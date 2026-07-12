"""
distributed.py
--------------
Thin wrappers around torch.distributed so the rest of the codebase
never touches os.environ directly.
"""

import os
import torch
import torch.distributed as dist


def setup() -> tuple[int, int, int]:
    """
    Initialise the NCCL process group (torchrun sets env vars automatically).

    Returns
    -------
    rank        : global rank of this process
    world_size  : total number of processes
    local_rank  : rank on the current node (= GPU index to bind to)
    """
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup() -> None:
    """Tear down the process group."""
    dist.destroy_process_group()


def barrier() -> None:
    """Thin wrapper so callers don't import dist directly."""
    dist.barrier()


def is_main(rank: int) -> bool:
    return rank == 0
