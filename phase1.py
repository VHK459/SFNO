"""
phase1.py
---------
Single-step prediction training (Phase 1).
"""

from __future__ import annotations
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
import xarray as xr
from tqdm import tqdm

import logger
import checkpointing
from config import TrainConfig
from config import build_channel_names
from data import make_train_prefetcher, make_val_prefetcher
from model import build_spectral_criterion



def run(
    ds_train: xr.Dataset,
    ds_val: xr.Dataset,
    model: torch.nn.Module,
    cfg: TrainConfig,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    """Execute Phase 1: single-step supervised training."""
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"\n### Phase 1: Single-step training on {world_size} GPU(s) ###")

    criterion = build_spectral_criterion(ds_train, device, spectral=True, crop=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.phase1_lr)

    # Warmup → cosine-with-restarts
    warmup    = LinearLR(optimizer, start_factor=0.001, end_factor=1.0,
                         total_iters=cfg.phase1_gradient_steps)
    cosine    = CosineAnnealingWarmRestarts(optimizer, T_0=1_000, T_mult=1, eta_min=5e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.phase1_warmup_steps])

    train_pf = make_train_prefetcher(ds_train, sequence_length=2,
                                     batch_size=cfg.batch_size, device=local_rank, config = cfg.data_config)
    val_pf   = make_val_prefetcher(ds_val,     sequence_length=2,
                                   batch_size=1, device=rank, config = cfg.data_config)
    train_pf.start()
    val_pf.start()

    model.train()
    steps = tqdm(range(cfg.phase1_gradient_steps), desc="Phase 1") if is_main \
            else range(cfg.phase1_gradient_steps)


    static_vars = cfg.data_config['static_variables']
    all_vars = build_channel_names(cfg.data_config)
    mask = [not i in static_vars for i in all_vars]
    indices = [i for i,m in enumerate(mask) if m]
    
    # if is_main:
    #     print(f' The train variables are {all_vars}')
    #     print(f' The pred variables are {all_vars[indices]}')
    
    for step in steps:
        # ── Forward ─────────────────────────────────────────────────────────
        batch  = train_pf.get()
        x, y   = batch[:, :, 0, :, :].to(device), batch[:, mask, 1, :, :].to(device)
        if is_main: print(x.shape, y.shape)
        optimizer.zero_grad()
        pred              = model(x)
        loss, spectral    = criterion(pred, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                              max_norm=cfg.phase1_grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Train logging ────────────────────────────────────────────────────
        if step % cfg.phase1_log_every == 0:
            logger.log_phase1_train(
                step=step, loss=loss.item(), spectral_loss=spectral.item(),
                grad_norm=norm, lr=optimizer.param_groups[0]["lr"], is_main=is_main,
            )
            if is_main and hasattr(steps, "set_postfix"):
                steps.set_postfix(loss=f"{loss.item():.4f}",
                                  lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                print(f"[P1 {step}] loss={loss.item():.4f}  "
                      f"spectral={spectral.item():.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                      f"grad_norm={norm:.3f}")

        # ── Validation ───────────────────────────────────────────────────────
        if is_main and step % cfg.phase1_val_every == 0:
            model.eval()
            with torch.no_grad():
                bv        = val_pf.get()
                xv, yv    = bv[:, :, 0, :, :].to(device), bv[:, mask, 1, :, :].to(device)
                pv        = model(xv)
                lv, sv    = criterion(pv, yv)
                logger.log_phase1_val(step=step, val_loss=lv.item(),
                                      val_spectral_loss=sv.item(), is_main=is_main)
                print(f"[P1 {step}] VAL loss={lv.item():.4f}  spectral={sv.item():.4f}")
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if is_main and step % cfg.phase1_ckpt_every == 0:
            ckpt_path = cfg.phase1_ckpt_name.format(step=step)
            checkpointing.save(model, ckpt_path, is_main=is_main)

    train_pf.stop()
    val_pf.stop()

    # Final Phase 1 checkpoint (used as Phase 3 / Phase 4 starting point)
    checkpointing.save(model, cfg.phase1_final_ckpt, is_main=is_main)
    if is_main:
        print("Phase 1 complete.")
