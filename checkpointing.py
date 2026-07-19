"""
checkpointing.py
----------------
Save and load model checkpoints. All functions guard against non-main
ranks by accepting an is_main flag.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn


def save(model: nn.Module, path: str, is_main: bool) -> None:
    """
    Save model.module.state_dict() (unwraps DDP wrapper).
    No-op on non-main ranks.
    """
    if not is_main:
        return
    # Support both DDP-wrapped and bare modules
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)
    print(f"[ckpt] Saved → {path}")


def load(model: nn.Module, path: str, device: torch.device, is_main: bool) -> None:
    """
    Load state dict into model.module (unwraps DDP wrapper).
    Skips silently if the file does not exist.
    """
    if not os.path.exists(path):
        if is_main:
            print(f"[ckpt] WARNING: checkpoint '{path}' not found – skipping load.")
        return
    state = torch.load(path, map_location=device)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state)
    if is_main:
        print(f"[ckpt] Loaded ← {path}")
