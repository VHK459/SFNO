"""
logger.py
---------
Thin wrapper around W&B so every log call is typed and in one place.
All functions are no-ops when called from non-main ranks
(pass is_main=False from worker processes).
"""

from __future__ import annotations
from typing import Any

import numpy as np
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


def log_final_metrics(results: dict, is_main: bool, phase: str = "phase1") -> None:
    """
    Log end-of-phase RMSE/ACC metrics (as returned by metrics.evaluate) to
    W&B as proper line charts against rollout step, using
    wandb.plot.line_series. Produces four charts under a
    `{phase}_metrics/...` prefix:

      - rmse_mean_vs_rollout       : one line, mean-over-variables RMSE vs
                                      rollout step (1..T)
      - acc_mean_vs_rollout        : one line, mean-over-variables ACC vs
                                      rollout step
      - rmse_per_variable_vs_rollout : one line per variable, RMSE vs
                                         rollout step
      - acc_per_variable_vs_rollout  : one line per variable, ACC vs
                                         rollout step

    (Logging 28 separate flat scalars, one per lead time, gives W&B no way
    to draw a lead-time axis -- line_series builds the actual x/y chart in
    one shot instead.)
    """
    if not is_main:
        return

    prefix = f"{phase}_metrics"

    rmse_mean_curve = np.asarray(results["rmse_mean_over_vars"], dtype=float)  # (T,)
    acc_mean_curve = np.asarray(results["acc_mean_over_vars"], dtype=float)    # (T,)
    rollout_steps = list(range(1, len(rmse_mean_curve) + 1))

    var_names = list(results["rmse"].keys())
    rmse_per_var = [np.asarray(results["rmse"][v], dtype=float).tolist() for v in var_names]
    acc_per_var = [np.asarray(results["acc"][v], dtype=float).tolist() for v in var_names]

    log_dict = {
        # Single-number rollout+variable-averaged summaries, handy for
        # sorting/filtering runs in the W&B table view.
        f"{prefix}/rmse_overall": float(rmse_mean_curve.mean()),
        f"{prefix}/acc_overall": float(acc_mean_curve.mean()),

        # Mean-over-variables RMSE / ACC vs rollout step
        f"{prefix}/rmse_mean_vs_rollout": wandb.plot.line_series(
            xs=rollout_steps, ys=[rmse_mean_curve.tolist()], keys=["mean_rmse"],
            title=f"{prefix}: mean RMSE vs rollout step", xname="rollout step",
        ),
        f"{prefix}/acc_mean_vs_rollout": wandb.plot.line_series(
            xs=rollout_steps, ys=[acc_mean_curve.tolist()], keys=["mean_acc"],
            title=f"{prefix}: mean ACC vs rollout step", xname="rollout step",
        ),

        # Per-variable RMSE / ACC vs rollout step (one line per variable,
        # sharing the same x-axis)
        f"{prefix}/rmse_per_variable_vs_rollout": wandb.plot.line_series(
            xs=rollout_steps, ys=rmse_per_var, keys=var_names,
            title=f"{prefix}: RMSE per variable vs rollout step", xname="rollout step",
        ),
        f"{prefix}/acc_per_variable_vs_rollout": wandb.plot.line_series(
            xs=rollout_steps, ys=acc_per_var, keys=var_names,
            title=f"{prefix}: ACC per variable vs rollout step", xname="rollout step",
        ),
    }

    wandb.log(log_dict)


def log_phase1_metrics(results: dict, is_main: bool) -> None:
    """End-of-phase-1 RMSE/ACC metrics. Thin wrapper around log_final_metrics."""
    log_final_metrics(results, is_main=is_main, phase="phase1")


def log_phase3_metrics(results: dict, is_main: bool) -> None:
    """End-of-phase-3 RMSE/ACC metrics. Thin wrapper around log_final_metrics."""
    log_final_metrics(results, is_main=is_main, phase="phase3")


def log_phase4_metrics(results: dict, is_main: bool) -> None:
    """End-of-phase-4 RMSE/ACC metrics. Thin wrapper around log_final_metrics."""
    log_final_metrics(results, is_main=is_main, phase="phase4")


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