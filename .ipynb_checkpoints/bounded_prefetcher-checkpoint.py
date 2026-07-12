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







class ERA5PrefetcherV2:
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
        means_path: str = "/storage/vishnu/means_1979.zarr",
        stds_path: str = "/storage/vishnu/std_1979.zarr",
    ):
        self.ds = ds
        self.B = batch_size
        self.seq_len = sequence_length
        self.device = device
        self.check_nans = check_nans
        self.normalize = normalize
        self.config = config 

        self.channel_names = build_channel_names(self.config)
        self.n_channels = len(self.channel_names)

        self.surface_vars  = self.config["surface_variables"]
        self.vertical_vars = self.config["vertical_variables"]
        self.levels        = self.config["levels"]
        self.static_vars   = self.config["static_variables"]

        self.time_max = self.ds.time.size - sequence_length
        self.valid_indices = self.time_max

        self.queue = queue.Queue(maxsize=queue_size)
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
            self.means = np.array(
                [float(np.asarray(means_ds[name].values).squeeze())
                 for name in self.channel_names],
                dtype=np.float32,
            )
            self.stds = np.array(
                [float(np.asarray(stds_ds[name].values).squeeze())
                 for name in self.channel_names],
                dtype=np.float32,
            )

    # ── control ────────────────────────────────────────────────────

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

                self.queue.put(tensor)

            except Exception as e:
                print(f"Worker Error: {e}")
                self.stop_event.set()


