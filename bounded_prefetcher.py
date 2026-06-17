import threading
import queue
import numpy as np
import xarray as xr
import torch
import time
import dask
import os
# Prevent CPU lockups by restricting Dask to a single thread per worker
# dask.config.set(scheduler='synchronous')

class ERA5Prefetcher:
    def __init__(self, ds, batch_size, queue_size=4, sequence_length=14, device="cpu",
                 check_nans=False, normalize=True,
                 means='/storage/vishnu/means_1979.zarr', stds='/storage/vishnu/std_1979.zarr'):
        self.ds = ds
        self.B = batch_size
        self.time_max = self.ds.time.size - sequence_length
        self.queue = queue.Queue(maxsize=queue_size)
        self.device = device
        self.seq_len = sequence_length
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.valid_indices = self.time_max
        self.check_nans = check_nans
        self.normalize = normalize
        if self.normalize:
            self.means = xr.open_zarr(means).to_array().values
            self.stds =  xr.open_zarr(means).to_array().values

    # def get_valid_indices(self):  ## There are no nans to all indices are valid
    #     time_max = self.time_max
    #     nan_times = np.load('nan_times.npy')
    #     offsets = np.arange(-self.seq_len - 1, self.seq_len + 1)
    #     masks = np.unique((nan_times[:, None] + offsets[None, :]).flatten())
    #     arr = np.arange(time_max)
    #     masks = masks[(masks <= time_max)]
    #     arr[masks] = -1
    #     arr = arr[(arr >= 0) & (arr < time_max)]
    #     return np.unique(arr)

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

    def _worker(self):
        worker_id = threading.get_ident()
        print(f"[Worker {worker_id}] started")
        while not self.stop_event.is_set():
            if self.queue.full():
                time.sleep(0.01)
                continue
            loc = np.random.choice(self.valid_indices, size=self.B)
            # print(loc, end='')
            time_idx = loc[:, None] + np.arange(self.seq_len)[None, :]
            try:
                batch_view = self.ds['era5_features'].isel(
                    time=xr.DataArray(time_idx, dims=("batch", "step"))
                ).transpose("batch", "channel", "step", "latitude", "longitude")

                if self.normalize:
                    numpy_batch = batch_view.values
                    means = np.expand_dims(self.means, axis=(0, 2, 3, 4))
                    stds = np.expand_dims(self.stds, axis=(0, 2, 3, 4))
                    numpy_batch = (numpy_batch - means) / stds
                else:
                    numpy_batch = batch_view.values

                if self.check_nans:
                    if np.isnan(numpy_batch).any():
                        print('!!!! Error nan found.')
                        np.save('time_ids_for_nans.npy', time_idx)

                tensor = torch.from_numpy(numpy_batch).contiguous().float()
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()
                self.queue.put(tensor)

            except Exception as e:
                print(f"Worker Error: {e}")
                self.stop_event.set()

    def get(self):
        return self.queue.get(timeout=600)

# --- SETUP SCRIPT ---

# DATA_PATH = "/storage/vishnu/era5_stacked.zarr/"
# # Hint: If your Zarr chunks are huge (e.g. whole month), set chunks={'time': 1} here
# ds = xr.open_zarr(DATA_PATH, chunks={}) 

# print("Graph prepared. Starting Prefetcher...")

# # 3. Initialize Prefetcher with the PRE-STACKED dataset
# prefetcher = ERA5Prefetcher(
#     ds,
#     batch_size=32,
#     queue_size=4,
#     sequence_length=14,
#     device="cuda"
# )

# prefetcher.start()

# # --- TRAINING LOOP ---
# num_steps = 3
# t0 = time.time()

# try:
#     for step in range(num_steps):
#         t1 = time.time()
        
#         batch = prefetcher.get()
        
#         # Move to GPU here (main thread)
#         # batch = batch.to("cuda", non_blocking=True)
        
#         print(f"Step {step}: shape={batch.shape}, wait_time={t1 - t0:.4f}s")
        
#         # Simulate training
       
#         t0 = time.time()
        
#         del batch
# finally:
#     prefetcher.stop()
