
"""
phase4.py
---------
Cropped-region fine-tuning (Phase 4).
Loads from the Phase 1 (or Phase 3) checkpoint and trains on a spatial
crop of the globe.
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


def _crop(tensor: torch.Tensor,
          w_slice: tuple[int, int],
          h_slice: tuple[int, int]) -> torch.Tensor:
    """Spatial crop: [B, C, H, W] → [B, C, h, w]."""
    return tensor[:, :, h_slice[0]:h_slice[1], w_slice[0]:w_slice[1]]


def _val_schedule(total_steps: int, n_points: int = 30, tail: int = 10) -> np.ndarray:
    """
    Sqrt-spaced step indices at which to run validation, with a dense tail.

    Validation is expensive, and during cosine-annealed fine-tuning val
    loss is noisy and not very informative early on (LR is still high) --
    it only becomes meaningful for model selection once the LR has
    annealed down, near the end of the run. So this schedule is SPARSE
    early (large gaps between validation steps) and DENSE late (every
    step is validated for the final `tail` steps).

    Concretely: `linspace(0, 1, n_points) ** 0.5` compresses -- consecutive
    points are far apart near 0 and close together near 1, so mapping
    those through `* (total_steps - 1)` gives step indices that start
    sparse and end dense.
    """
    if total_steps <= 0:
        return np.array([], dtype=int)
    n_points = max(1, min(n_points, total_steps))
    sparse = (np.linspace(0, 1, n_points) ** 0.5 * (total_steps - 1)).astype(int)
    dense_tail = np.arange(max(total_steps - tail, 0), total_steps)
    return np.sort(np.unique(np.concatenate([sparse, dense_tail])))


def run(
    ds_train: xr.Dataset,
    ds_val: xr.Dataset,
    model: torch.nn.Module,
    cfg: TrainConfig,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    """Execute Phase 4: cropped-region fine-tuning."""
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"\n### Phase 4: Cropped fine-tuning on {world_size} GPU(s) ###")

    checkpointing.load(model, cfg.phase4_load_ckpt, device=device, is_main=is_main)

    criterion = build_spectral_criterion(ds_train, device, spectral=False, crop=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.phase4_lr)
    # Scheduler defined but commented-out in original; kept here for easy re-enabling.
    # NOTE: if you do enable a WarmRestarts schedule (multiple annealing
    # cycles), the sqrt+tail validation schedule below is dense only near
    # the very END of the whole run, not near the end of each restart
    # cycle. If you want validation to sharpen near the end of every
    # cycle instead, build a periodic version keyed on T_0 rather than
    # `cfg.phase4_gradient_steps`.
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=5e-5)

    # ── Dynamic / static channel mask ───────────────────────────────────────
    # Model predicts only the dynamic channels (out_chans = channels - n_static);
    # targets must be sliced down to match, or the loss shape-mismatches.
    static_vars = cfg.data_config['static_variables']
    all_vars    = build_channel_names(cfg.data_config)
    mask        = np.array([v not in static_vars for v in all_vars])
    dyn_mask_t  = torch.as_tensor(mask, dtype=torch.bool)  # CPU, for indexing batches

    train_pf = make_train_prefetcher(ds_train, sequence_length=2,
                                     batch_size=cfg.batch_size, device=local_rank,
                                     config=cfg.data_config)
    val_pf   = make_val_prefetcher(ds_val,     sequence_length=2,
                                   batch_size=cfg.batch_size, device=local_rank,
                                   config=cfg.data_config)

    val_schedule = _val_schedule(
        cfg.phase4_gradient_steps,
        n_points=cfg.phase4_val_schedule_points,
        tail=cfg.phase4_val_schedule_tail,
    )
    if is_main:
        print(f"[P4] validation scheduled at {len(val_schedule)} / "
              f"{cfg.phase4_gradient_steps} steps (sparse→dense)")

    train_pf.start()
    val_pf.start()

    best_val_loss = None

    model.train()
    steps = tqdm(range(cfg.phase4_gradient_steps), desc="Phase 4") if is_main \
            else range(cfg.phase4_gradient_steps)

    for step in steps:
        # ── Forward ─────────────────────────────────────────────────────────
        batch = train_pf.get()
        x     = batch[:, :, 0, :, :].to(device)             # full channels (dyn + static)
        y     = batch[:, dyn_mask_t, 1, :, :].to(device)    # dynamic-only target

        optimizer.zero_grad()
        pred  = model(x)
        loss  = criterion(
            _crop(pred, cfg.phase4_crop_w, cfg.phase4_crop_h),
            _crop(y,    cfg.phase4_crop_w, cfg.phase4_crop_h),
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                              max_norm=cfg.phase4_grad_clip)
        optimizer.step()
        # scheduler.step()

        # ── Train logging ────────────────────────────────────────────────────
        if step % cfg.phase4_log_every == 0:
            logger.log_phase4_train(
                step=step, loss=loss.item(), grad_norm=norm,
                lr=optimizer.param_groups[0]["lr"], is_main=is_main,
            )
            if is_main and hasattr(steps, "set_postfix"):
                steps.set_postfix(loss=f"{loss.item():.4f}",
                                  lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                print(f"[P4 {step}] loss={loss.item():.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                      f"grad_norm={norm:.3f}")

        # ── Validation (sparse early, dense late -- see _val_schedule) ────────
        if is_main and step in val_schedule:
            model.eval()
            with torch.no_grad():
                bv     = val_pf.get()
                xv     = bv[:, :, 0, :, :].to(device)
                yv     = bv[:, dyn_mask_t, 1, :, :].to(device)
                pv     = model(xv)
                lossv  = criterion(
                    _crop(pv, cfg.phase4_crop_w, cfg.phase4_crop_h),
                    _crop(yv, cfg.phase4_crop_w, cfg.phase4_crop_h),
                )
                logger.log_phase4_val(step=step, val_loss=lossv.item(), is_main=is_main)
                print(f"[P4 {step}] VAL loss={lossv.item():.4f}")

                # ── Best-model checkpoint (dir + model.pth + YAML) ────────────
                best_val_loss = checkpointing.save_best(
                    model, cfg.phase4_best_ckpt_dir, is_main=is_main,
                    metric=lossv.item(), best_metric=best_val_loss,
                    cfg=cfg, metric_name="val_loss",
                    extra_metrics={"step": step},
                )
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if is_main and step % cfg.phase4_ckpt_every == 0:
            ckpt_path = cfg.phase4_ckpt_name.format(step=step)
            checkpointing.save(model, ckpt_path, is_main=is_main)

    train_pf.stop()
    val_pf.stop()

    if is_main:
        print("Phase 4 complete.")

    # ── Full RMSE / ACC evaluation over the whole validation set ─────────────
    # Off by default (phase4_run_metrics=False): this is a full-globe,
    # multi-step rollout eval, which is a lot of extra compute to spend on
    # a model that was only fine-tuned on a regional crop. Turn it on if
    # you specifically want to check whether the regional fine-tune helped
    # or hurt global skill.
    if is_main and cfg.phase4_run_metrics:
        print("\n### Phase 4: running RMSE/ACC evaluation ###")
        eval_model = model.module if hasattr(model, "module") else model

        best_ckpt_path = os.path.join(cfg.phase4_best_ckpt_dir, "model.pth")
        if os.path.exists(best_ckpt_path):
            checkpointing.load(eval_model, best_ckpt_path, device=device, is_main=is_main)
        else:
            print("[metrics] No best checkpoint found; evaluating the final model.")

        results = metrics.evaluate(
            eval_model, ds_val, cfg, device,
            rollout_len=cfg.phase4_metric_rollout_len,
            n_samples=cfg.phase4_metric_samples,
            clim_path=cfg.clim_path,
            is_main=is_main,
        )
        logger.log_phase4_metrics(results, is_main=is_main)
