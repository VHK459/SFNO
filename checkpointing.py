"""
checkpointing.py
----------------
Save and load model checkpoints. All functions guard against non-main
ranks by accepting an is_main flag.
"""

from __future__ import annotations

import os
import datetime
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml


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


# ── YAML-safe serialization helpers ─────────────────────────────────────────

def _yaml_safe(obj: Any) -> Any:
    """
    Recursively coerce an object into something yaml.safe_dump can handle:
    dicts/lists/tuples/sets -> dicts/lists, primitives pass through,
    anything else (functions, custom objects, etc.) -> str(obj).
    """
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_yaml_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _config_to_dict(cfg: Any) -> dict:
    """
    Turn a TrainConfig (or any config object) into a plain dict.

    `asdict()` only picks up annotated dataclass *fields* -- attributes
    like `data_config = dict(...)` that are assigned without a type
    annotation are class-level attributes, not dataclass fields, and are
    silently skipped by asdict(). We add those back in explicitly so the
    saved YAML reflects the exact config actually used for this run.
    """
    if is_dataclass(cfg):
        d = asdict(cfg)
    else:
        d = dict(vars(cfg))

    for extra_attr in ("data_config",):
        if hasattr(cfg, extra_attr) and extra_attr not in d:
            d[extra_attr] = getattr(cfg, extra_attr)

    return _yaml_safe(d)


def save_best(
    model: nn.Module,
    save_dir: str,
    is_main: bool,
    metric: float,
    best_metric: Optional[float],
    cfg: Any = None,
    metric_name: str = "val_loss",
    ckpt_name: str = "model.pth",
    extra_metrics: Optional[dict] = None,
) -> Optional[float]:
    """
    Save a checkpoint + metadata YAML into `save_dir`, only if `metric`
    improves on `best_metric` (lower is better, e.g. validation loss).
    No-op on non-main ranks.

    Writes:
        save_dir/model.pth              -- model.state_dict()
        save_dir/best_model_info.yaml    -- metric, previous best, timestamp,
                                             any extra_metrics, and the full
                                             training config (if `cfg` given)

    Parameters
    ----------
    save_dir      : directory to create/overwrite with the best checkpoint.
    metric        : the current metric value.
    best_metric   : the best metric value seen so far, or None if this is
                    the first comparison.
    cfg           : optional config object (e.g. TrainConfig) to dump into
                    the YAML alongside the metric.
    metric_name   : label used for `metric` in the YAML (default "val_loss").
    ckpt_name     : filename for the checkpoint inside save_dir.
    extra_metrics : optional dict of additional scalars/info to record.

    Returns
    -------
    The updated best_metric (== metric if this call improved on it,
    otherwise the unchanged best_metric).
    """
    if not is_main:
        return best_metric
    if best_metric is not None and metric >= best_metric:
        return best_metric

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, ckpt_name)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, ckpt_path)

    info = {
        metric_name: float(metric),
        "previous_best": float(best_metric) if best_metric is not None else None,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if extra_metrics:
        info.update(_yaml_safe(extra_metrics))
    if cfg is not None:
        info["config"] = _config_to_dict(cfg)

    yaml_path = os.path.join(save_dir, "best_model_info.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(info, f, sort_keys=False, default_flow_style=False)

    print(f"[ckpt] New best ({metric_name}={metric:.4f}, prev={best_metric}) → {ckpt_path}")
    print(f"[ckpt] Wrote metadata → {yaml_path}")
    return metric


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