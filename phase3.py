"""
phase3.py
---------
Autoregressive curriculum training (Phase 3).
Loads the Phase 1 checkpoint, then iterates over AR step counts
from phase3_ar_start to phase3_ar_end.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import xarray as xr
from tqdm import tqdm

import logger
import checkpointing
import metrics
from config import TrainConfig
from config import build_channel_names
from data import make_train_prefetcher, make_val_prefetcher
from model import build_spectral_criterion


def _lin_range(total_steps: int) -> np.ndarray:
    """Sqrt-spaced + tail indices used for val/checkpoint gating."""
    if total_steps <= 100:
        ll = (np.linspace(0, 1, 5) ** 0.5 * total_steps).astype(int)
    elif total_steps > 100 and total_steps <= 1000:
        ll = (np.linspace(0, 1, 50) ** 0.5 * total_steps).astype(int)
    else:
        ll = (np.linspace(0, 1, 30) ** 0.5 * total_steps).astype(int)
    tail = np.arange(total_steps - 10, total_steps)
    return np.sort(np.unique(np.append(ll, tail)))


def _autoregressive_loss(
    model: torch.nn.Module,
    initial_input: torch.Tensor,
    y_true: torch.Tensor,
    ar_steps: int,
    criterion,
    static_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Roll the model ar_steps times and accumulate loss.

    initial_input : (B, C_in, H, W) -- full channel set (dynamic + static),
                     the model's expected input shape.
    y_true        : (B, C_dyn, ar_steps, H, W) -- DYNAMIC-ONLY targets,
                     matching the model's out_chans (static fields aren't
                     predicted, so they're never part of the target either).
    static_mask   : (C_in,) bool tensor, True at the positions of static
                     (non-predicted) channels within the model's input
                     ordering. Static fields don't change over time, so the
                     values from `initial_input` are re-attached to the
                     model's dynamic-only prediction before every
                     subsequent AR step -- without this the model receives
                     too few input channels from step 2 onward (this was a
                     real bug: `current = pred` fed the model only its own
                     out_chans back in, but its Conv2d input layer expects
                     in_chans = out_chans + n_static).

    Returns
    -------
    total_loss  : scalar tensor (still has grad)
    spec_accum  : float (spectral component, detached)
    """
    current = initial_input
    static_channels = current[:, static_mask]  # constant across all AR steps

    total_loss = 0.0
    spec_accum = 0.0

    for t in range(ar_steps):
        pred                     = model(current)                      # (B, C_dyn, H, W)
        step_loss, step_spectral = criterion(pred, y_true[:, :, t, :, :])
        total_loss  = total_loss + step_loss + step_spectral
        spec_accum += step_spectral.item()
        current     = torch.cat([pred, static_channels], dim=1)        # reattach for next step

    return total_loss, spec_accum


def run(
    ds_train: xr.Dataset,
    ds_val: xr.Dataset,
    model: torch.nn.Module,
    cfg: TrainConfig,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    """Execute Phase 3: autoregressive curriculum training."""
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"\n### Phase 3: Autoregressive training on {world_size} GPU(s) ###")

    checkpointing.load(model, cfg.phase3_load_ckpt, device=device, is_main=is_main)

    criterion = build_spectral_criterion(ds_train, device, spectral=True, crop=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.phase3_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=1, eta_min=5e-7)
    lin_range = _lin_range(cfg.phase3_gradient_steps)

    if is_main:
        print(f" validation is performed on {lin_range}")

    # Global step counter spanning the ENTIRE phase (all curriculum stages
    # concatenated). Used ONLY for W&B's "phase3_step" x-axis. The local
    # `step` below still resets to 0 at the start of every ar_steps stage
    # and continues to drive tqdm, the val/checkpoint schedule, etc. --
    # this counter is what fixes the logging bug: previously every log
    # call used the local `step`, so each new curriculum stage restarted
    # "phase3_step" at 0 and its early points landed on top of (visually
    # "overwriting") the previous stage's points on the same W&B chart.
    global_step = 0

    # ── Dynamic / static channel mask ───────────────────────────────────────
    # The model's input channels are [dynamic..., static...] (see
    # build_channel_names); it predicts only the dynamic subset. `mask` is
    # True at dynamic positions, used both to slice targets down to
    # out_chans and (inverted) to know which input channels to re-attach
    # between AR steps.
    static_vars = cfg.data_config['static_variables']
    all_vars    = build_channel_names(cfg.data_config)
    mask        = np.array([v not in static_vars for v in all_vars])
    dyn_mask_t  = torch.as_tensor(mask, dtype=torch.bool)                 # CPU, for indexing batches
    static_mask_t = torch.as_tensor(~mask, dtype=torch.bool, device=device)  # for _autoregressive_loss

    for ar_steps in range(cfg.phase3_ar_start, cfg.phase3_ar_end + 1):
        if is_main:
            print(f"\n--- Phase 3 curriculum: {ar_steps} AR step(s) ---")

        train_pf = make_train_prefetcher(
            ds_train, sequence_length=ar_steps + 1,
            batch_size=cfg.batch_size, device=local_rank,
            config=cfg.data_config,
        )
        val_pf = make_val_prefetcher(
            ds_val, sequence_length=ar_steps + 1,
            batch_size=cfg.batch_size, device=local_rank,
            config=cfg.data_config,
        )
        train_pf.start()
        val_pf.start()

        # Best-of-this-curriculum-stage checkpoint. Reset per stage since
        # loss accumulates over `ar_steps` forward passes -- val loss at
        # ar_steps=2 and ar_steps=4 aren't on the same scale, so "best" is
        # only meaningful within a single stage.
        best_val_loss = None
        best_dir = cfg.phase3_best_ckpt_dir_template.format(ar_steps=ar_steps)

        running_loss = 0.0
        steps = tqdm(range(cfg.phase3_gradient_steps),
                     desc=f"Phase 3  AR={ar_steps}") if is_main \
                else range(cfg.phase3_gradient_steps)

        for step in steps:
            model.train()

            batch         = train_pf.get()
            initial_input = batch[:, :, 0, :, :].to(device)          # full channels (dyn + static)
            y_true        = batch[:, dyn_mask_t, 1:, :, :].to(device)  # dynamic-only targets

            optimizer.zero_grad()
            loss, spec_accum = _autoregressive_loss(
                model, initial_input, y_true, ar_steps, criterion, static_mask_t,
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  max_norm=float("inf"))
            optimizer.step()
            scheduler.step()

            ll           = loss.item()
            running_loss += ll

            # ── Train logging ────────────────────────────────────────────────
            # NOTE: logs against `global_step`, not the local `step`, so the
            # W&B x-axis is monotonic across the whole phase (see comment
            # above). `ar_steps` is still logged as its own field so you can
            # filter/color the W&B chart by curriculum stage if you want.
            if is_main and step % cfg.phase3_log_every == 0:
                logger.log_phase3_train(
                    step=global_step, total_loss=ll, grad_norm=norm,
                    avg_step_loss=running_loss / (step + 1),
                    ar_steps=ar_steps, lr=optimizer.param_groups[0]["lr"],
                    is_main=is_main,
                )
                if is_main and hasattr(steps, "set_postfix"):
                    steps.set_postfix(loss=f"{ll:.4f}",
                                      lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                    print(f"[P3 AR={ar_steps} {step}] loss={ll:.4f}  "
                          f"spectral_accum={spec_accum:.4f}  "
                          f"grad_norm={norm:.3f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")

            del batch

            # ── Validation (on sqrt-spaced steps) ───────────────────────────
            if is_main and (step % cfg.phase3_gradient_steps) in lin_range:
                model.eval()
                with torch.no_grad():
                    bv          = val_pf.get()
                    iv          = bv[:, :, 0, :, :].to(device)
                    yv          = bv[:, dyn_mask_t, 1:, :, :].to(device)
                    val_loss, _ = _autoregressive_loss(
                        model, iv, yv, ar_steps, criterion, static_mask_t,
                    )
                logger.log_phase3_val(step=global_step, val_loss=val_loss.item(),
                                      ar_steps=ar_steps, is_main=is_main)
                print(f"[P3 AR={ar_steps} {step}] VAL loss={val_loss.item():.4f}")

                # ── Best-of-stage checkpoint (dir + model.pth + YAML) ───────
                best_val_loss = checkpointing.save_best(
                    model, best_dir, is_main=is_main,
                    metric=val_loss.item(), best_metric=best_val_loss,
                    cfg=cfg, metric_name="val_loss",
                    extra_metrics={"step_within_stage": step, "global_step": global_step,
                                   "ar_steps": ar_steps},
                )
                model.train()

            global_step += 1

        train_pf.stop()
        val_pf.stop()

        # Per-curriculum checkpoint (final weights at end of this stage)
        ckpt_path = cfg.phase3_ckpt_name.format(ar_steps=ar_steps)
        checkpointing.save(model, ckpt_path, is_main=is_main)

    # Final model
    checkpointing.save(model, cfg.phase3_final_ckpt, is_main=is_main)
    if is_main:
        print("Phase 3 complete.")

    # ── Full RMSE / ACC evaluation over the whole validation set ─────────────
    # Run once, after the entire curriculum finishes, on the main rank only.
    # Uses the best checkpoint from the LAST (longest-horizon) curriculum
    # stage if one was saved, otherwise the final model.
    if is_main and cfg.phase3_run_metrics:
        print("\n### Phase 3: running RMSE/ACC evaluation ###")
        eval_model = model.module if hasattr(model, "module") else model

        final_best_dir = cfg.phase3_best_ckpt_dir_template.format(ar_steps=cfg.phase3_ar_end)
        final_best_ckpt = os.path.join(final_best_dir, "model.pth")
        if os.path.exists(final_best_ckpt):
            checkpointing.load(eval_model, final_best_ckpt, device=device, is_main=is_main)
        else:
            print("[metrics] No best checkpoint found for final AR stage; "
                  "evaluating the final model.")

        results = metrics.evaluate(
            eval_model, ds_val, cfg, device,
            rollout_len=cfg.phase3_metric_rollout_len,
            n_samples=cfg.phase3_metric_samples,
            clim_path=cfg.clim_path,
            is_main=is_main,
        )
        logger.log_phase3_metrics(results, is_main=is_main)

        # Restore final trained weights in case `model` is reused by a
        # subsequent phase in the same process.
        checkpointing.load(eval_model, cfg.phase3_final_ckpt, device=device, is_main=is_main)