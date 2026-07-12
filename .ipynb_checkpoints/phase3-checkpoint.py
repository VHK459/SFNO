"""
phase3.py
---------
Autoregressive curriculum training (Phase 3).
Loads the Phase 1 checkpoint, then iterates over AR step counts
from phase3_ar_start to phase3_ar_end.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.optim as optim
import xarray as xr
from tqdm import tqdm

import logger
import checkpointing
from config import TrainConfig
from data import make_train_prefetcher, make_val_prefetcher
from model import build_spectral_criterion


def _lin_range(total_steps: int) -> np.ndarray:
    """Sqrt-spaced + tail indices used for val/checkpoint gating."""
    ll = (np.linspace(0, 1, 30) ** 0.5 * total_steps).astype(int)
    tail = np.arange(total_steps - 10, total_steps)
    return np.sort(np.unique(np.append(ll, tail)))


def _autoregressive_loss(
    model: torch.nn.Module,
    initial_input: torch.Tensor,
    y_true: torch.Tensor,
    ar_steps: int,
    criterion,
) -> tuple[torch.Tensor, float]:
    """
    Roll the model ar_steps times and accumulate loss.

    Returns
    -------
    total_loss  : scalar tensor (still has grad)
    spec_accum  : float (spectral component, detached)
    """
    current = initial_input
    total_loss  = 0.0
    spec_accum  = 0.0

    for t in range(ar_steps):
        pred                    = model(current)
        step_loss, step_spectral = criterion(pred, y_true[:, :, t, :, :])
        total_loss  = total_loss + step_loss + step_spectral
        spec_accum += step_spectral.item()
        current     = pred

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

    lin_range = _lin_range(cfg.phase3_gradient_steps)

    for ar_steps in range(cfg.phase3_ar_start, cfg.phase3_ar_end + 1):
        if is_main:
            print(f"\n--- Phase 3 curriculum: {ar_steps} AR step(s) ---")

        train_pf = make_train_prefetcher(
            ds_train, sequence_length=ar_steps + 1,
            batch_size=cfg.batch_size, device=local_rank,
        )
        # val_pf = make_val_prefetcher(
        #     ds_val, sequence_length=ar_steps + 1,
        #     batch_size=cfg.batch_size, device=rank,
        # )
        train_pf.start()
        # val_pf.start()

        running_loss = 0.0
        steps = tqdm(range(cfg.phase3_gradient_steps),
                     desc=f"Phase 3  AR={ar_steps}") if is_main \
                else range(cfg.phase3_gradient_steps)

        for step in steps:
            model.train()

            batch         = train_pf.get()
            initial_input = batch[:, :, 0, :, :].to(device)
            y_true        = batch[:, :, 1:, :, :].to(device)

            optimizer.zero_grad()
            loss, spec_accum = _autoregressive_loss(
                model, initial_input, y_true, ar_steps, criterion,
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  max_norm=float("inf"))
            optimizer.step()

            ll           = loss.item()
            running_loss += ll

            # ── Train logging ────────────────────────────────────────────────
            if is_main and step % cfg.phase3_log_every == 0:
                logger.log_phase3_train(
                    step=step, total_loss=ll, grad_norm=norm,
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
            # if is_main and (step % cfg.phase3_gradient_steps) in lin_range:
            #     model.eval()
            #     with torch.no_grad():
            #         bv            = val_pf.get()
            #         iv            = bv[:, :, 0, :, :].squeeze().to(device)
            #         yv            = bv[:, :, 1:, :, :].squeeze().to(device)
            #         val_loss, _   = _autoregressive_loss(model, iv, yv, ar_steps, criterion)
            #     logger.log_phase3_val(step=step, val_loss=val_loss.item(),
            #                           ar_steps=ar_steps, is_main=is_main)
            #     print(f"[P3 AR={ar_steps} {step}] VAL loss={val_loss.item():.4f}")
            #     model.train()

        train_pf.stop()
        # val_pf.stop()

        # Per-curriculum checkpoint
        ckpt_path = cfg.phase3_ckpt_name.format(ar_steps=ar_steps)
        checkpointing.save(model, ckpt_path, is_main=is_main)

    # Final model
    checkpointing.save(model, cfg.phase3_final_ckpt, is_main=is_main)
    if is_main:
        print("Phase 3 complete.")
