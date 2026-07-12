"""
data.py
-------
Dataset construction and prefetcher factory functions.
All time-slice logic is centralised here.
"""

import xarray as xr

from bounded_prefetcher import ERA5Prefetcher

from config import TrainConfig as cfg
# ── Dataset factories ────────────────────────────────────────────────────────

def get_train_dataset(data_path: str) -> xr.Dataset:
    """ERA5 data up to end of 2016 (training split)."""
    ds = xr.open_zarr(data_path, chunks={})
    return ds.sel(time=slice(None, "2000-12-31"))


def get_val_dataset(data_path: str) -> xr.Dataset:
    """ERA5 data for 2017–2018 (validation split)."""
    ds = xr.open_zarr(data_path, chunks={})
    return ds.sel(time=slice("2017-01-01", "2018-12-31"))


def get_test_dataset(data_path: str) -> xr.Dataset:
    """ERA5 data from 2019 onwards (test split)."""
    ds = xr.open_zarr(data_path, chunks={})
    return ds.sel(time=slice("2019-01-01", None))


# ── Prefetcher factories ─────────────────────────────────────────────────────

def make_train_prefetcher(
    ds: xr.Dataset,
    sequence_length: int,
    batch_size: int,
    device: int,
    config: dict,
    queue_size: int = 4,
) -> ERA5Prefetcher:
    return ERA5Prefetcher(
        ds,
        batch_size=batch_size,
        queue_size=queue_size,
        sequence_length=sequence_length,
        device=device,
        config=config,
    )

def make_val_prefetcher(
    ds: xr.Dataset,
    sequence_length: int,
    batch_size: int,
    device: int,
    config: dict,
    queue_size: int = 4,
) -> ERA5Prefetcher:
    return ERA5Prefetcher(
        ds,
        batch_size=batch_size,
        queue_size=queue_size,
        sequence_length=sequence_length,
        device=device,
        config=config,
    )
