"""
logger.py
---------
Thin wrapper around W&B so every log call is typed and in one place.
All functions are no-ops when called from non-main ranks
(pass is_main=False from worker processes).
"""

from __future__ import annotations
from typing import Any

import wandb


def init(cfg, world_size: int) -> None:
    """Initialise a W&B run (call only from rank 0)."""
    import dataclasses
    wandb.init(
        project=cfg.wandb_project,
        name=f"run_phase_{cfg.phase}_bs_{cfg.batch_size}_gelu_layer_norm",
        config=dataclasses.asdict(cfg),
        group=f"DDP_worldsize_{world_size}",
    )
    # Register custom x-axes so charts align correctly
    wandb.define_metric("phase1_step")
    wandb.define_metric("phase3_step")
    wandb.define_metric("phase4_step")
    wandb.define_metric("phase1/*", step_metric="phase1_step")
    wandb.define_metric("phase3/*", step_metric="phase3_step")
    wandb.define_metric("phase4/*", step_metric="phase4_step")


def finish() -> None:
    wandb.finish()


def watch(model, log_freq: int = 1) -> None:
    wandb.watch(model, log="all", log_freq=log_freq)


# ── Phase-specific log helpers ───────────────────────────────────────────────

def log_phase1_train(
    step: int,
    loss: float,
    spectral_loss: float,
    grad_norm: float,
    lr: float,
    is_main: bool,
) -> None:
    if not is_main:
        return
    wandb.log({
        "phase1/train_loss":    loss,
        "phase1/spectral_loss": spectral_loss,
        "phase1/grad_norm":     grad_norm,
        "phase1/learning_rate": lr,
        "phase1_step":          step,
    })


def log_phase1_val(
    step: int,
    val_loss: float,
    val_spectral_loss: float,
    is_main: bool,
) -> None:
    if not is_main:
        return
    wandb.log({
        "phase1/val_loss":          val_loss,
        "phase1/val_spectral_loss": val_spectral_loss,
        "phase1_step":              step,
    })


def log_phase3_train(
    step: int,
    total_loss: float,
    grad_norm: float,
    avg_step_loss: float,
    ar_steps: int,
    lr: float,
    is_main: bool,
) -> None:
    if not is_main:
        return
    wandb.log({
        "phase3/total_loss":    total_loss,
        "phase3/grad_norm":     grad_norm,
        "phase3/avg_step_loss": avg_step_loss,
        "phase3/ar_steps":      ar_steps,
        "phase3/learning_rate": lr,
        "phase3_step":          step,
    })


def log_phase3_val(
    step: int,
    val_loss: float,
    ar_steps: int,
    is_main: bool,
) -> None:
    if not is_main:
        return
    wandb.log({
        "phase3/val_loss": val_loss,
        "phase3/ar_steps": ar_steps,
        "phase3_step":     step,
    })


def log_phase4_train(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    is_main: bool,
) -> None:
    if not is_main:
        return
    wandb.log({
        "phase4/train_loss":    loss,
        "phase4/grad_norm":     grad_norm,
        "phase4/learning_rate": lr,
        "phase4_step":          step,
    })


def log_phase4_val(
    step: int,
    val_loss: float,
    is_main: bool,
) -> None:
    if not is_main:
        return
    wandb.log({
        "phase4/val_loss": val_loss,
        "phase4_step":     step,
    })
