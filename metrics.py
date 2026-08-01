"""
metrics.py
----------
Latitude-weighted RMSE / ACC evaluation for a trained SFNO model.

This module is import-safe (nothing runs at import time). The main entry
point is `evaluate`, which:

  1. spins up an `ERA5Prefetcher` in `metric=True` mode (batch_size=1,
     one random rollout window per `.get()` call),
  2. autoregressively rolls the model out `rollout_len` steps,
  3. de-normalizes both the forecast and the ground truth back to
     physical units,
  4. aligns the WeatherBench2 hourly climatology to the forecast's valid
     times (pointwise on day-of-year / hour),
  5. computes latitude-weighted RMSE and ACC per channel and per lead
     time, averaged over `n_samples` random rollout windows.

Can also be run standalone:

    python metrics.py --ckpt sfno_phase1.pth --rollout_len 28 --n_samples 4
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import xarray as xr

from config import TrainConfig, build_channel_names
from bounded_prefetcher import ERA5Prefetcher

DEFAULT_CLIM_PATH = (
    "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/"
    "era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)


# ── Latitude weighting ────────────────────────────────────────────────────────

def latitude_weights(lat: np.ndarray) -> np.ndarray:
    """Cosine(latitude) weights, normalized to mean 1."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    w = w / w.mean()
    return w


def _broadcast_weights(w: np.ndarray, ndim: int, lat_axis: int) -> np.ndarray:
    """Reshape a 1-D lat-weight vector to broadcast against an ndim array."""
    lat_axis = lat_axis % ndim
    shape = [1] * ndim
    shape[lat_axis] = w.shape[0]
    return w.reshape(shape)


def weighted_rmse_np(forecast, truth, lat, lat_axis=-2, mean_axes=None):
    """
    forecast, truth : numpy arrays; lat_axis points to the latitude dim.
    lat             : 1-D latitude values matching that axis's size.
    mean_axes       : axes to average over, e.g. (-2, -1) to reduce
                       (lat, lon) and leave (channel, time).
    """
    w = latitude_weights(lat)
    w_b = _broadcast_weights(w, forecast.ndim, lat_axis)
    sq_err = (forecast - truth) ** 2
    return np.sqrt((sq_err * w_b).mean(axis=mean_axes))


def weighted_acc_np(forecast, truth, climatology, lat, lat_axis=-2, mean_axes=(-2, -1)):
    """
    Latitude-weighted anomaly correlation coefficient.
    forecast, truth, climatology : numpy arrays, same shape.
    lat                          : 1-D latitude values matching lat_axis.
    """
    w = latitude_weights(lat)
    w_b = _broadcast_weights(w, forecast.ndim, lat_axis)

    f_anom = forecast - climatology
    t_anom = truth - climatology

    num = (f_anom * t_anom * w_b).mean(axis=mean_axes)
    denom_f = (f_anom ** 2 * w_b).mean(axis=mean_axes)
    denom_t = (t_anom ** 2 * w_b).mean(axis=mean_axes)

    return num / np.sqrt(denom_f * denom_t)


# ── Climatology ────────────────────────────────────────────────────────────────

def load_climatology(clim_path: str, data_config: dict) -> xr.Dataset:
    """
    Open the WeatherBench2 hourly climatology zarr and subset/rename it to
    the same dynamic channel set produced by
    `build_channel_names(data_config, ignore_static=True)`.

    Each variable in the returned Dataset has dims (dayofyear, hour,
    latitude, longitude).
    """
    clim = xr.open_zarr(clim_path, chunks={})

    data_vars = {}
    for var in data_config["surface_variables"]:
        data_vars[var] = clim[var]
    for var in data_config["vertical_variables"]:
        for lev in data_config["levels"]:
            # .sel(level=lev) leaves behind a scalar, non-dimension 'level'
            # coordinate. Since each (var, lev) pair becomes its own
            # data_vars entry keyed by name (e.g. 'temperature_250'), that
            # leftover coord conflicts across levels once combined into a
            # single Dataset (xarray tries to merge 'level' == 250 vs 500
            # vs 850 as if it were one shared coordinate) -> drop it here.
            data_vars[f"{var}_{lev}"] = clim[var].sel(level=lev).drop_vars("level")

    return xr.Dataset(data_vars)


def align_climatology(clim_ds: xr.Dataset, times: np.ndarray, channel_names: List[str]) -> np.ndarray:
    """
    Point-wise select the climatology at each timestamp in `times` (matched
    on day-of-year / hour-of-day), and stack channels in `channel_names`
    order.

    times : 1-D array-like of np.datetime64 (forecast valid times).
    Returns a numpy array shaped (channel, time, lat, lon).
    """
    times_da = xr.DataArray(np.asarray(times), dims="time")
    dayofyear = times_da.dt.dayofyear
    hour = times_da.dt.hour

    aligned = clim_ds.sel(dayofyear=dayofyear, hour=hour)
    arr = aligned[channel_names].to_array(dim="channel")
    arr = arr.transpose("channel", "time", "latitude", "longitude")
    return arr.values.astype(np.float32)


# ── Rollout ──────────────────────────────────────────────────────────────────

def rollout(model, batch, rollout_len, device, means=None, stds=None, static_mask=None):
    """
    Autoregressive rollout -> numpy array shaped (channel, time, lat, lon),
    ready to pass directly into weighted_rmse_np / weighted_acc_np.

    batch       : torch tensor (B, C_in, step, H, W) -- prefetcher output.
                  Only the initial condition (step 0) is used; B must be 1.
    means, stds : optional, shape (C_out,) -- if given, predictions are
                  DE-NORMALIZED back to physical units before returning
                  (RMSE/ACC should be computed in physical units, not
                  z-scored units).
    static_mask : optional boolean array, shape (C_in,), True at the
                  positions of static (non-predicted) channels within the
                  model's input ordering. The model outputs only the
                  dynamic channels (out_chans = C_in - n_static), so those
                  static channels must be re-attached to `out` before it
                  can be fed back in as the next step's input -- static
                  fields don't change over time, so the values from the
                  initial condition are reused at every step. If None,
                  the model's out_chans is assumed to equal its in_chans
                  and `out` is fed back in directly.
    """
    model.eval()
    preds = []
    current = batch[:, :, 0].to(device)  # (B, C_in, H, W) -- initial condition

    static_channels = None
    if static_mask is not None:
        static_mask_t = torch.as_tensor(static_mask, dtype=torch.bool, device=device)
        # Static fields are constant across time, so the slice taken from
        # the initial condition is valid to reuse at every rollout step.
        static_channels = current[:, static_mask_t]  # (B, C_static, H, W)

    with torch.no_grad():
        for _ in range(rollout_len):
            out = model(current)  # (B, C_out, H, W) -- dynamic channels only
            preds.append(out.detach().cpu().numpy())
            if static_channels is not None:
                # Reconstruct the full (dynamic + static) input ordering
                # the model expects for the next step.
                current = torch.cat([out, static_channels], dim=1)
            else:
                current = out

    preds_arr = np.stack(preds, axis=0)  # (T, B, C, H, W)
    preds_arr = preds_arr[:, 0]          # squeeze batch -> (T, C, H, W)
    preds_arr = np.moveaxis(preds_arr, 0, 1)  # -> (C, T, H, W)

    if means is not None and stds is not None:
        means_b = np.asarray(means).reshape(-1, 1, 1, 1)
        stds_b = np.asarray(stds).reshape(-1, 1, 1, 1)
        preds_arr = preds_arr * stds_b + means_b  # back to physical units

    return preds_arr  # (C, T, H, W)


# ── Full evaluation pipeline ────────────────────────────────────────────────────

def evaluate(
    model: torch.nn.Module,
    ds_eval: xr.Dataset,
    cfg: TrainConfig,
    device: torch.device,
    rollout_len: int = 28,
    n_samples: int = 4,
    clim_path: str = None,
    is_main: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate `model` with `n_samples` random autoregressive rollouts of
    length `rollout_len`, sampled from `ds_eval`.

    Returns a dict:
        {
          "rmse": {channel_name: np.ndarray of shape (rollout_len,), ...},
          "acc":  {channel_name: np.ndarray of shape (rollout_len,), ...},
          "rmse_mean_over_vars": np.ndarray (rollout_len,),
          "acc_mean_over_vars":  np.ndarray (rollout_len,),
        }
    """
    clim_path = clim_path or DEFAULT_CLIM_PATH
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    data_config = cfg.data_config
    all_vars = build_channel_names(data_config)               # includes static
    static_vars = set(data_config["static_variables"])
    dyn_mask = np.array([v not in static_vars for v in all_vars])
    dyn_names = [v for v, m in zip(all_vars, dyn_mask) if m]   # matches model out_chans order

    prefetcher = ERA5Prefetcher(
        ds_eval,
        batch_size=1,
        queue_size=1,
        sequence_length=rollout_len + 1,
        device=device,
        check_nans=False,
        normalize=True,
        config=data_config,
        metric=True,
        means_path=cfg.means_path,
        stds_path=cfg.stds_path,
    )
    prefetcher.start()

    means = prefetcher.means  # (C_total,)
    stds = prefetcher.stds
    dyn_means = means[dyn_mask]
    dyn_stds = stds[dyn_mask]
    lat = ds_eval.latitude.values

    clim_ds = load_climatology(clim_path, data_config)

    rmse_samples, acc_samples = [], []
    try:
        for i in range(n_samples):
            batch, times = prefetcher.get()  # batch: (1,C,T+1,H,W); times: (1,T+1)

            unnorm_batch = (
                batch.cpu().numpy() * stds.reshape(1, -1, 1, 1, 1)
                + means.reshape(1, -1, 1, 1, 1)
            )
            truth_dyn = unnorm_batch[0][dyn_mask][:, 1:]  # (C_dyn, T, H, W)

            forecast = rollout(raw_model, batch, rollout_len, device,
                                means=dyn_means, stds=dyn_stds,
                                static_mask=~dyn_mask)  # (C_dyn, T, H, W)

            target_times = times[0][1:]
            clim_aligned = align_climatology(clim_ds, target_times, dyn_names)

            rmse = weighted_rmse_np(forecast, truth_dyn, lat, lat_axis=-2, mean_axes=(-2, -1))
            acc = weighted_acc_np(forecast, truth_dyn, clim_aligned, lat, lat_axis=-2, mean_axes=(-2, -1))

            rmse_samples.append(rmse)  # (C, T)
            acc_samples.append(acc)    # (C, T)

            if is_main:
                print(f"[metrics] sample {i + 1}/{n_samples} done "
                      f"(rollout_len={rollout_len})")
    finally:
        prefetcher.stop()

    rmse_mean = np.mean(np.stack(rmse_samples, axis=0), axis=0)  # (C, T)
    acc_mean = np.mean(np.stack(acc_samples, axis=0), axis=0)    # (C, T)

    results: Dict[str, Dict[str, np.ndarray]] = {"rmse": {}, "acc": {}}
    for ci, name in enumerate(dyn_names):
        results["rmse"][name] = rmse_mean[ci]
        results["acc"][name] = acc_mean[ci]
    results["rmse_mean_over_vars"] = rmse_mean.mean(axis=0)
    results["acc_mean_over_vars"] = acc_mean.mean(axis=0)

    if is_main:
        print(f"[metrics] mean RMSE (over vars, per lead step): {results['rmse_mean_over_vars']}")
        print(f"[metrics] mean ACC  (over vars, per lead step): {results['acc_mean_over_vars']}")

    return results


# ── Standalone CLI ──────────────────────────────────────────────────────────────

def _build_model_for_eval(ds: xr.Dataset, cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    from model import build_model
    return build_model(ds, cfg).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone SFNO metric evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_path", type=str, default=None,
                         help="Overrides TrainConfig.data_path")
    parser.add_argument("--clim_path", type=str, default=DEFAULT_CLIM_PATH)
    parser.add_argument("--rollout_len", type=int, default=28)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--eval_start", type=str, default="2019-01-01")
    parser.add_argument("--eval_end", type=str, default="2019-12-31")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional .npz path to save the raw results")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.data_path:
        cfg.data_path = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = xr.open_zarr(cfg.data_path, chunks={})
    ds_eval = ds.sel(time=slice(args.eval_start, args.eval_end))

    model = _build_model_for_eval(ds_eval, cfg, device)

    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    results = evaluate(
        model, ds_eval, cfg, device,
        rollout_len=args.rollout_len,
        n_samples=args.n_samples,
        clim_path=args.clim_path,
        is_main=True,
    )

    if args.out:
        np.savez(args.out, **{
            f"rmse_{k}": v for k, v in results["rmse"].items()
        }, **{
            f"acc_{k}": v for k, v in results["acc"].items()
        }, rmse_mean_over_vars=results["rmse_mean_over_vars"],
           acc_mean_over_vars=results["acc_mean_over_vars"])
        print(f"Saved results -> {args.out}")


if __name__ == "__main__":
    main()