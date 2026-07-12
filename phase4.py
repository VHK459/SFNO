"""
phase4.py
---------
Cropped-region fine-tuning (Phase 4).
Loads from the Phase 1 checkpoint and trains on a spatial crop of the globe.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import xarray as xr
from tqdm import tqdm

import logger
import checkpointing
from config import TrainConfig
from data import make_train_prefetcher, make_val_prefetcher
from model import build_spectral_criterion


def _crop(tensor: torch.Tensor,
          w_slice: tuple[int, int],
          h_slice: tuple[int, int]) -> torch.Tensor:
    """Spatial crop: [B, C, H, W] → [B, C, h, w]."""
    return tensor[:, :, h_slice[0]:h_slice[1], w_slice[0]:w_slice[1]]


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
    # Scheduler defined but commented-out in original; kept here for easy re-enabling
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=5e-5)

    train_pf = make_train_prefetcher(ds_train, sequence_length=2,
                                     batch_size=cfg.batch_size, device=local_rank)
    val_pf   = make_val_prefetcher(ds_val,     sequence_length=2,
                                   batch_size=cfg.batch_size, device=rank)

    ll_range = (np.linspace(0, 1, 30) ** 0.5 * 1000).astype(int)
    lin_range = np.sort(np.append(ll_range, np.arange(990, 1000)))

    train_pf.start()
    val_pf.start()

    model.train()
    steps = tqdm(range(cfg.phase4_gradient_steps), desc="Phase 4") if is_main \
            else range(cfg.phase4_gradient_steps)

    for step in steps:
        # ── Forward ─────────────────────────────────────────────────────────
        batch = train_pf.get()
        x     = batch[:, :, 0, :, :].to(device)
        y     = batch[:, :, 1, :, :].to(device)

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

        # ── Validation ───────────────────────────────────────────────────────
        if is_main and step % cfg.phase4_val_every == 0:
            model.eval()
            with torch.no_grad():
                bv     = val_pf.get()
                xv     = bv[:, :, 0, :, :].to(device)
                yv     = bv[:, :, 1, :, :].to(device)
                pv     = model(xv)
                lossv  = criterion(
                    _crop(pv, cfg.phase4_crop_w, cfg.phase4_crop_h),
                    _crop(yv, cfg.phase4_crop_w, cfg.phase4_crop_h),
                )
                logger.log_phase4_val(step=step, val_loss=lossv.item(), is_main=is_main)
                print(f"[P4 {step}] VAL loss={lossv.item():.4f}")
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if is_main and step % cfg.phase4_ckpt_every == 0:
            ckpt_path = cfg.phase4_ckpt_name.format(step=step)
            checkpointing.save(model, ckpt_path, is_main=is_main)

    train_pf.stop()
    val_pf.stop()

    if is_main:
        print("Phase 4 complete.")
