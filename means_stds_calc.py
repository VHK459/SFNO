import xarray as xr
import dask
from dask.distributed import Client

def main():
    # 1. Initialize the Dask Distributed Cluster
    # Adjust memory/workers based on your actual machine specs
    client = Client(n_workers=8, threads_per_worker=4, memory_limit='25GB')
    
    print(f"Dask Dashboard is available at: {client.dashboard_link}")
    print("Open this link in your browser to monitor the compute live!\n")

    # 2. Load the Zarr dataset
    data_path = "/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    
    print("Loading Zarr metadata...")
    ds = xr.open_zarr(data_path,chunks={"time": 500}, consolidated=True)
    ds = ds.sel(time=slice('1979-01-01', '2018-12-31'))
    
    # 3. Apply chunking
    # Time chunk of 500 is good (~180MB per chunk for a 3D float32 variable)
    # ds = ds.chunk({"time": 500, "level": -1, "latitude": -1, "longitude": -1})

    # 4. Define lazy computations
    # We only reduce over time, lat, and lon, leaving 'level' intact
    reduce_dims = ["time", "latitude", "longitude"]
    
    print("Building Dask task graph for Mean and Std...")
    mean_lazy = ds.mean(dim=reduce_dims, skipna=True)
    std_lazy  = ds.std(dim=reduce_dims, skipna=True)

    # 5. Execute in a SINGLE pass using dask.compute
    # By passing both lazy objects to a single compute call, Dask knows 
    # to load the chunks from disk exactly once, compute both stats, and discard the chunk.
    print("Computing stats... (Check your Dask dashboard for progress)")
    mean, std = dask.compute(mean_lazy, std_lazy)

    print("\nComputation Complete!")
    
    # 6. Save the results so you never have to do this again
    mean.to_zarr("/storage/vishnu/era5_mean_all.zarr", mode="w", zarr_format=2)
    std.to_zarr("/storage/vishnu/era5_std_all.zarr", mode="w", zarr_format=2)
    
    # Close the client
    client.close()

if __name__ == "__main__":
    main()