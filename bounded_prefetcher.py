import threading
import queue
import numpy as np
import xarray as xr
import torch
import time
import dask
import os

from config import build_channel_names
# Prevent CPU lockups by restricting Dask to a single thread per worker
# dask.config.set(scheduler='synchronous')

# ─────────────────────────────────────────────
# VARIABLE CONFIG — edit this to add/remove variables or levels
# ─────────────────────────────────────────────






"""
era5_prefetcher_v2.py
======================
Prefetcher that builds training batches directly from the RAW ERA5 zarr
(surface vars: dims (time, latitude, longitude); vertical vars: dims
(time, level, latitude, longitude); static vars: dims (latitude, longitude),
no time dim) — no need to pre-flatten levels and re-save a stacked zarr.

Add/remove a variable or level -> just edit `config`. No data pipeline
re-run required.
"""



# ─────────────────────────────────────────────
# VARIABLE CONFIG — edit this to add/remove variables or levels
# ─────────────────────────────────────────────

# DEFAULT_CONFIG = dict(
#     surface_variables=[
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "2m_temperature",
#         "total_column_water_vapour",
#         "surface_pressure",
#     ],
#     vertical_variables=[
#         "u_component_of_wind",
#         "v_component_of_wind",
#         "temperature",
#         "geopotential",
#         "specific_humidity",
#     ],
#     levels=[250, 500, 850],           # subset (or all) of ds['level'].values
#     static_variables=[
#         "land_sea_mask",
#         "soil_type",
#     ],
# )


# def build_channel_names(config: dict) -> list:
#     """
#     Deterministic channel ordering:
#       surface vars  ->  vertical vars x levels (var-major)  ->  static vars
#     This order MUST match the order used to look up means/stds per channel.
#     """
#     names = list(config["surface_variables"])
#     for var in config["vertical_variables"]:
#         for lev in config["levels"]:
#             names.append(f"{var}_{lev}")
#     names += list(config["static_variables"])
#     return names


# ─────────────────────────────────────────────
# PREFETCHER
# ─────────────────────────────────────────────

class ERA5Prefetcher:
    """
    Same interface as the old ERA5Prefetcher (start/get/stop), but reads
    directly from a RAW xarray Dataset instead of a pre-flattened one.

    ds must contain:
        - surface variables  : dims (time, latitude, longitude)
        - vertical variables : dims (time, level, latitude, longitude)
        - static variables   : dims (latitude, longitude)  [NO time dim]

    Which variables/levels get used is controlled entirely by `config` —
    change it and the prefetcher pulls a different channel set, with no
    re-saving of a stacked zarr required.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        batch_size: int,
        queue_size: int = 4,
        sequence_length: int = 14,
        device: str = "cpu",
        check_nans: bool = False,
        normalize: bool = True,
        config: dict = None,
        metric = False,
        means_path: str = "/storage/vishnu/era5_mean_all.zarr",
        stds_path: str = "/storage/vishnu/era5_std_all.zarr",
    ):
        self.ds = ds
        self.B = batch_size
        self.seq_len = sequence_length
        self.device = device
        self.check_nans = check_nans
        self.normalize = normalize
        self.config = config 
        self.metric = metric

        self.channel_names = build_channel_names(self.config)
        self.n_channels = len(self.channel_names)

        self.surface_vars  = self.config["surface_variables"]
        self.vertical_vars = self.config["vertical_variables"]
        self.levels        = self.config["levels"]
        self.static_vars   = self.config["static_variables"]

        self.time_max = self.ds.time.size - sequence_length
        self.valid_indices = self.time_max

        self.queue = queue.Queue(maxsize=queue_size = 1) if metric == True else queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)

        # ── Static channels: load ONCE, they never vary with time ────
        self._static_cache = None
        if self.static_vars:
            static_arrs = [self.ds[v].transpose('latitude', 'longitude').values.astype(np.float32) for v in self.static_vars]
            self._static_cache = np.stack(static_arrs, axis=0)   # (n_static, lat, lon)

        # ── Normalization stats, ordered to match self.channel_names ──
        if self.normalize:
            means_ds = xr.open_zarr(means_path)
            stds_ds  = xr.open_zarr(stds_path)
            self.means = self._extract_stats(means_ds)
            self.stds = self._extract_stats(stds_ds)


    # ── control ────────────────────────────────────────────────────

    def _extract_stats(self, ds):
        arrays = []
         # surface variables — (B, step, lat, lon) each
        for var in self.surface_vars:
            arr = (
                ds[var]
                .values
            )
            # print(arr, 'surface')
            arrays.append(arr)
    
        # vertical variables x levels — (B, step, lat, lon) each
        for var in self.vertical_vars:
            for lev in self.levels:
                arr = (
                    ds[var]
                    .sel(level=lev)
                    .values
                )
                # print(arr, ' ', lev, ' vert',)
                arrays.append(arr)
                
        for var in self.static_vars:
            arr = (
                ds[var]
                .values
            )
            # print(arr,' static')
            arrays.append(arr)
        # print(arr)
        batch = np.asarray(arrays).astype(np.float32)
        # print(batch)

        return batch

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join()

    def get(self):
        return self.queue.get(timeout=600)

    # ── batch construction ───────────────────────────────────────

    def _extract_batch(self, time_idx: np.ndarray) -> np.ndarray:
        """
        time_idx : (B, seq_len) int array of time indices into self.ds

        Returns: (B, C, seq_len, lat, lon) float32 numpy array,
        channel order == self.channel_names.
        """
        time_da = xr.DataArray(time_idx, dims=("batch", "step"))
        arrays = []

        # surface variables — (B, step, lat, lon) each
        for var in self.surface_vars:
            arr = (
                self.ds[var]
                .isel(time=time_da)
                .transpose("batch", "step", "latitude", "longitude")
                .values
            )
            arrays.append(arr)

        # vertical variables x levels — (B, step, lat, lon) each
        for var in self.vertical_vars:
            for lev in self.levels:
                arr = (
                    self.ds[var]
                    .sel(level=lev)
                    .isel(time=time_da)
                    .transpose("batch", "step", "latitude", "longitude")
                    .values
                )
                arrays.append(arr)

        # stack -> (B, C_dynamic, step, lat, lon)
        batch = np.stack(arrays, axis=1).astype(np.float32)

        # static variables — broadcast to (B, C_static, step, lat, lon)
        if self._static_cache is not None:
            n_static = self._static_cache.shape[0]
            lat, lon = self._static_cache.shape[-2:]
            static_batch = np.broadcast_to(
                self._static_cache[None, :, None, :, :],
                (time_idx.shape[0], n_static, self.seq_len, lat, lon),
            )
            batch = np.concatenate([batch, static_batch], axis=1)

        return batch  # (B, C, step, lat, lon)

    # ── worker loop ────────────────────────────────────────────────

    def _worker(self):
        worker_id = threading.get_ident()
        print(f"[Worker {worker_id}] started — {self.n_channels} channels: "
              f"{self.channel_names}")

        while not self.stop_event.is_set():
            if self.queue.full():
                time.sleep(0.01)
                continue

            loc = np.random.choice(self.valid_indices, size=self.B)
            time_idx = loc[:, None] + np.arange(self.seq_len)[None, :]

            try:
                numpy_batch = self._extract_batch(time_idx)  # (B,C,step,lat,lon)

                if self.normalize:
                    means = self.means.reshape(1, -1, 1, 1, 1)
                    stds  = self.stds.reshape(1, -1, 1, 1, 1)
                    numpy_batch = (numpy_batch - means) / stds

                if self.check_nans and np.isnan(numpy_batch).any():
                    print("!!!! Error nan found.")
                    np.save("time_ids_for_nans.npy", time_idx)

                tensor = torch.from_numpy(numpy_batch).contiguous().float()
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()
                if self.metric == True:
                    self.queue.put((tensor, batch_view.time.values))
                else:
                    self.queue.put(tensor)

            except Exception as e:
                print(f"Worker Error: {e}")
                self.stop_event.set()

