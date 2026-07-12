"""
train.py
--------
Entry point. Parses config, sets up DDP, loads datasets,
builds the model, then dispatches to the requested phase(s).

Launch with:
    torchrun --nproc_per_node=<N_GPUS> train.py [--phase 1|3|4|all] [...]
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import distributed
import logger
import checkpointing
from config import parse_args
from data import get_train_dataset, get_val_dataset, get_test_dataset
from model import build_model
import phase1
import phase3
import phase4


def main() -> None:
    # ── Distributed setup ───────────────────────────────────────────────────
    rank, world_size, local_rank = distributed.setup()
    is_main = distributed.is_main(rank)
    device  = torch.device(f"cuda:{local_rank}")

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = parse_args()

    # ── W&B ─────────────────────────────────────────────────────────────────
    if is_main:
        logger.init(cfg, world_size)

    # ── Datasets ────────────────────────────────────────────────────────────
    ds_train = get_train_dataset(cfg.data_path)
    ds_val   = get_val_dataset(cfg.data_path)
    # ds_test  = get_test_dataset(cfg.data_path)   # held-out; not used during training

    if is_main:
        print("Datasets opened successfully")
        print(f"  Train : {ds_train.time.values[0]} → {ds_train.time.values[-1]}")
        print(f"  Val   : {ds_val.time.values[0]}   → {ds_val.time.values[-1]}")
        # print(f"  Test  : {ds_test.time.values[0]}  → {ds_test.time.values[-1]}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(ds_train, cfg).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    if is_main:
        logger.watch(model, log_freq=1)

    # ── Training phases ──────────────────────────────────────────────────────
    if cfg.phase in ("1", "all"):
        phase1.run(ds_train, ds_val, model, cfg,
                   rank=rank, local_rank=local_rank, world_size=world_size)
        # Ensure all ranks see the Phase 1 checkpoint before Phase 3 loads it
        distributed.barrier()

    if cfg.phase in ("3", "all"):
        phase3.run(ds_train, ds_val, model, cfg,
                   rank=rank, local_rank=local_rank, world_size=world_size)

    if cfg.phase in ("4", "all"):
        phase4.run(ds_train, ds_val, model, cfg,
                   rank=rank, local_rank=local_rank, world_size=world_size)

    # ── Teardown ─────────────────────────────────────────────────────────────
    if is_main:
        logger.finish()

    distributed.cleanup()


if __name__ == "__main__":
    main()
