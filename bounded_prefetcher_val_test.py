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

class ERA5PrefetcherValTest:
    def __init__(self, ds, batch_size, queue_size=4, sequence_length=14, device="cpu",check_nans=False,normalize=True,means='means_era5_dataset.npy',stds='std_era5_dataset.npy'):
        """
        ds: A PRE-STACKED xarray DataArray with shape (time, lat, lon, channel)
        """
        self.ds = ds
        self.B = batch_size
        self.time_max = self.ds.time.size - sequence_length
        self.queue = queue.Queue(maxsize=queue_size)
        self.device = device
        self.seq_len = sequence_length
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        # self.valid_indices = self.get_valid_indices()
        self.check_nans = check_nans
        self.normalize = normalize
        if self.normalize:
            self.means = np.load(means)
            self.stds = np.load(stds)
        
    # def get_valid_indices(self):
    #     time_max = self.time_max
    #     nan_times = np.load('nan_times.npy')
    #     offsets = np.arange(- self.seq_len - 1, self.seq_len + 1)
    #     masks = np.unique((nan_times[:,None] + offsets[None,:]).flatten())
    #     arr = np.arange(time_max)
    #     masks = masks[(masks <= time_max)]
    #     arr[masks] = -1
    #     arr = arr[(arr >= 0) & (arr < time_max)] 
    #     return np.unique(arr)
        
    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        # Drain queue to allow thread to exit if stuck on put()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join()

    def _worker(self):
        # Calculate valid start indices
        # Assuming ds structure is (time, lat, lon, channel) or similar
        total_time = self.time_max
        # Valid starts: ensure we don't go out of bounds
        # max_start = total_time - (self.seq_len * 6) 
        worker_id = threading.get_ident()
        seed = (worker_id + os.getpid()) % (2**32 - 1)
        np.random.seed(seed)
        
        print(f"[Worker {worker_id}] started")

        while not self.stop_event.is_set():
            # Throttle slightly to let main thread breathe if queue is full
            if self.queue.full():
                time.sleep(0.01)
                continue

            # t1 = time.time()
            
            # 1. Generate Indices (Vectorized)
            loc = np.random.choice(total_time, size=self.B)
            # Create batch indices: (Batch, Step)
            time_idx = loc[:, None] + np.arange(self.seq_len)[None, :]

            try:
                # 2. Slice (Single efficient command)
                # We expect ds to already be (time, lat, lon, channel)
                batch_view = self.ds['era5_features'].isel(
    time=xr.DataArray(time_idx, dims=("batch", "step"))
).transpose("batch", "channel", "step", "latitude","longitude")

                
                # We use .values to trigger the Dask computation synchronously
                if self.normalize:
                    numpy_batch = batch_view.values
                    means = np.expand_dims(self.means,axis=(2,3,4))
                    stds = np.expand_dims(self.stds, axis=(2,3,4))
                    numpy_batch = (numpy_batch - means)/stds
                else:
                    numpy_batch = batch_view.values
                
                # 4. To Torch
                # Transpose to (Batch, Channel, Step, Lat, Lon) standard DL format
                # Current: (batch, step, lat, lon, channel)
                # Target:  (batch, channel, step, lat, lon)
                if self.check_nans:
                    if np.isnan(numpy_batch).any():
                        print(f'!!!! Error nan found. Values used stored in file named time_ids_for_nans.npy . Good luck!!!')
                        np.save('time_ids_for_nans.npy',time_idx)
                
                tensor = torch.from_numpy(numpy_batch).contiguous().float()
                
                # Pin memory for faster CPU->GPU transfer later
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()

                self.queue.put(tensor)
                
                # print(f'[Worker {worker_id}] Batch ready: {time.time() - t1:.2f}s')

            except Exception as e:
                print(f"Worker Error: {e}")
                self.stop_event.set()

    def get(self):
        # Blocking get with timeout to allow clean exit on error
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
