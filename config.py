"""
config.py
---------
Central configuration: all hyperparameters and paths live here.
Parsed once in main() and threaded through every module.
"""

import argparse
from dataclasses import dataclass, field
from typing import Tuple

def build_channel_names(config: dict, ignore_static = False) -> list:
    """
    Deterministic channel ordering:
      surface vars  ->  vertical vars x levels (var-major)  ->  static vars
    This order MUST match the order used to look up means/stds per channel.
    """
    names = list(config["surface_variables"])
    for var in config["vertical_variables"]:
        for lev in config["levels"]:
            names.append(f"{var}_{lev}")
    if ignore_static:
        return names
    names += list(config["static_variables"])
    return names

@dataclass
class TrainConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    data_path: str = "/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

    # ── Phase selection ──────────────────────────────────────────────────────
    phase: str = "all"            # choices: '1', '3', '4', 'all'

    data_config = dict(
    surface_variables=[
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "total_column_water_vapour",
        "surface_pressure",
    ],
    vertical_variables=[
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "geopotential",
        "specific_humidity",
    ],
    levels=[250, 500, 850],           # subset (or all) of ds['level'].values
    static_variables=[
        "land_sea_mask",
        "soil_type",
    ],
)
    # ── Shared ──────────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 15

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    phase1_gradient_steps: int = 10001
    phase1_lr: float = 25e-4
    phase1_warmup_steps: int = 1000
    phase1_grad_clip: float = 32.0
    phase1_log_every: int = 10
    phase1_val_every: int = 10
    phase1_ckpt_every: int = 1000

  
    

    # ── Phase 1 metrics / best-model saving ────────────────────────────────────
    phase1_run_metrics: bool = True          # run full RMSE/ACC eval after phase1 finishes
    phase1_metric_rollout_len: int = 14      # autoregressive rollout length (in 6h steps)
    phase1_metric_samples: int = 100           # number of random rollout windows to average over
    phase1_best_ckpt_dir: str = "checkpoints/phase1_best_new"

    # ── Phase 3 ─────────────────────────────────────────────────────────────
    phase3_gradient_steps: int = 1001
    phase3_ar_start: int = 2
    phase3_ar_end: int = 5
    phase3_lr: float = 1e-2
    phase3_log_every: int = 10
    phase3_val_every: int = 10          # checked against lin_range inside loop

    # ── Phase 3 metrics / best-model saving ────────────────────────────────────
    phase3_run_metrics: bool = True          # run full RMSE/ACC eval after phase3 finishes
    phase3_metric_rollout_len: int = 28      # autoregressive rollout length (in 6h steps)
    phase3_metric_samples: int = 100           # number of random rollout windows to average over
    phase3_best_ckpt_dir_template: str = "checkpoints/phase3_best_new_ar{ar_steps}"

    # ── Phase 4 ─────────────────────────────────────────────────────────────
    phase4_gradient_steps: int = 10
    phase4_lr: float = 5e-5
    phase4_grad_clip: float = 32.0
    phase4_log_every: int = 10
    phase4_val_every: int = 10          # legacy/unused now -- kept for backward compat; see phase4_val_schedule_*
    phase4_ckpt_every: int = 10
    phase4_crop_w: Tuple[int, int] = field(default_factory=lambda: (40, 65))
    phase4_crop_h: Tuple[int, int] = field(default_factory=lambda: (64, 85))

    # ── Phase 4 validation schedule ─────────────────────────────────────────
    # Validation is expensive, and early in cosine-annealed fine-tuning the
    # LR is still high enough that val loss is noisy and not very
    # informative for model selection -- it only becomes meaningful once
    # the LR has annealed down near the end. So instead of a flat interval,
    # validate on a sqrt-spaced schedule (sparse early) with a dense tail
    # (frequent right at the end, where it actually matters).
    phase4_val_schedule_points: int = 30   # number of sqrt-spaced checkpoints across the full run
    phase4_val_schedule_tail: int = 10     # always validate every step for the final N steps

    # ── Phase 4 metrics / best-model saving ────────────────────────────────────
    phase4_run_metrics: bool = False  # off by default: expensive full-globe rollout eval on
                                       # a model that was only fine-tuned on a regional crop
    phase4_metric_rollout_len: int = 28
    phase4_metric_samples: int = 100
    phase4_best_ckpt_dir: str = "checkpoints/phase4_best_new"

    # ── Checkpoints ─────────────────────────────────────────────────────────
    phase1_ckpt_name: str = "sfno_phase1SPHERE_layernorm_{step}_new.pth"
    phase1_final_ckpt: str = "sfno_phase1.pth"
    phase3_load_ckpt: str = "checkpoints/phase1_best_new.pth"
    phase3_ckpt_name: str = "sfno_phase3_ar_{ar_steps}.pth"
    phase3_final_ckpt: str = "sfno_phase3_final.pth"
    phase4_load_ckpt: str = "sfno_phase1SPHERE_layernorm_10000_new.pth"
    phase4_ckpt_name: str = "sfno_phase4SPHERE_layernorm_{step}_final_new.pth"

    # ── Model ────────────────────────────────────────────────────────────────
    num_layers: int = 8
    scale_factor: int = 2
    embed_dim: int = 384
    hard_thresholding_fraction: float = 1.0
    channels: int = 22

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_project: str = "ERA5-train-distributed"

    # ── Metrics (RMSE / ACC evaluation) ────────────────────────────────────────
    clim_path: str = (
        "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/"
        "era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
    )
    means_path: str = "/storage/vishnu/era5_mean_all.zarr"
    stds_path: str = "/storage/vishnu/era5_std_all.zarr"


def parse_args() -> TrainConfig:
    """Parse CLI args and return a populated TrainConfig."""
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Distributed SFNO Training (torchrun)")

    parser.add_argument("--data_path",               type=str,   default=cfg.data_path)
    parser.add_argument("--phase",                   type=str,   default=cfg.phase,
                        choices=["1", "3", "4", "all"])
    parser.add_argument("--batch_size",              type=int,   default=cfg.batch_size)
    parser.add_argument("--num_workers",             type=int,   default=cfg.num_workers)

    # Phase 1
    parser.add_argument("--phase1_gradient_steps",  type=int,   default=cfg.phase1_gradient_steps)
    parser.add_argument("--phase1_lr",              type=float, default=cfg.phase1_lr)
    parser.add_argument("--phase1_run_metrics",     type=int,   default=int(cfg.phase1_run_metrics),
                        choices=[0, 1], help="1 to run RMSE/ACC eval after phase1, 0 to skip")
    parser.add_argument("--phase1_metric_rollout_len", type=int, default=cfg.phase1_metric_rollout_len)
    parser.add_argument("--phase1_metric_samples",  type=int,   default=cfg.phase1_metric_samples)

    # Phase 3
    parser.add_argument("--phase3_gradient_steps",  type=int,   default=cfg.phase3_gradient_steps)
    parser.add_argument("--phase3_ar_start",        type=int,   default=cfg.phase3_ar_start)
    parser.add_argument("--phase3_ar_end",          type=int,   default=cfg.phase3_ar_end)
    parser.add_argument("--phase3_lr",              type=float, default=cfg.phase3_lr)
    parser.add_argument("--phase3_run_metrics",     type=int,   default=int(cfg.phase3_run_metrics),
                        choices=[0, 1], help="1 to run RMSE/ACC eval after phase3, 0 to skip")
    parser.add_argument("--phase3_metric_rollout_len", type=int, default=cfg.phase3_metric_rollout_len)
    parser.add_argument("--phase3_metric_samples",  type=int,   default=cfg.phase3_metric_samples)

    # Phase 4
    parser.add_argument("--phase4_gradient_steps",  type=int,   default=cfg.phase4_gradient_steps)
    parser.add_argument("--phase4_val_schedule_points", type=int, default=cfg.phase4_val_schedule_points)
    parser.add_argument("--phase4_val_schedule_tail",    type=int, default=cfg.phase4_val_schedule_tail)
    parser.add_argument("--phase4_run_metrics",     type=int,   default=int(cfg.phase4_run_metrics),
                        choices=[0, 1], help="1 to run RMSE/ACC eval after phase4, 0 to skip")
    parser.add_argument("--phase4_metric_rollout_len", type=int, default=cfg.phase4_metric_rollout_len)
    parser.add_argument("--phase4_metric_samples",  type=int,   default=cfg.phase4_metric_samples)

    args = parser.parse_args()

    # Merge parsed values back into the dataclass
    cfg.data_path              = args.data_path
    cfg.phase                  = args.phase
    cfg.batch_size             = args.batch_size
    cfg.num_workers            = args.num_workers
    cfg.phase1_gradient_steps  = args.phase1_gradient_steps
    cfg.phase1_lr              = args.phase1_lr
    cfg.phase1_run_metrics     = bool(args.phase1_run_metrics)
    cfg.phase1_metric_rollout_len = args.phase1_metric_rollout_len
    cfg.phase1_metric_samples  = args.phase1_metric_samples
    cfg.phase3_gradient_steps  = args.phase3_gradient_steps
    cfg.phase3_ar_start        = args.phase3_ar_start
    cfg.phase3_ar_end          = args.phase3_ar_end
    cfg.phase3_lr              = args.phase3_lr
    cfg.phase3_run_metrics     = bool(args.phase3_run_metrics)
    cfg.phase3_metric_rollout_len = args.phase3_metric_rollout_len
    cfg.phase3_metric_samples  = args.phase3_metric_samples
    cfg.phase4_gradient_steps  = args.phase4_gradient_steps
    cfg.phase4_val_schedule_points = args.phase4_val_schedule_points
    cfg.phase4_val_schedule_tail   = args.phase4_val_schedule_tail
    cfg.phase4_run_metrics     = bool(args.phase4_run_metrics)
    cfg.phase4_metric_rollout_len = args.phase4_metric_rollout_len
    cfg.phase4_metric_samples  = args.phase4_metric_samples

    return cfg