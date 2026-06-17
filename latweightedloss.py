
import os
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import argparse
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperatorNet as SFNO



class LatWeightedLoss(nn.Module):
    def __init__(self, ds: xr.Dataset, p: int = 2, device: str = "cpu", pool = 'sum'):
        super().__init__()
        self.p = p
        self.pool = pool
        weights = self._get_lat_weights(ds)  # numpy
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.register_buffer("weights", weights.view(1, 1, -1, 1))

    def _get_lat_weights(self, ds: xr.Dataset) -> np.ndarray:
        def _assert_increasing(x: np.ndarray):
            if not (np.diff(x) > 0).all():
                raise ValueError("array is not increasing")
        
        def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
            pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
            return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])
        
        def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
            bounds = _latitude_cell_bounds(points)
            _assert_increasing(bounds)
            upper = bounds[1:]
            lower = bounds[:-1]
            return np.sin(upper) - np.sin(lower)
        
        weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
        weights /= np.mean(weights)
        return weights

    def forward(self, prediction, target) -> torch.Tensor:
        diff = (target - prediction) ** self.p
        numerator = torch.sum(self.weights * diff, dim=(-2, -1))
        denominator = torch.sum(self.weights * prediction**self.p, dim=(-2,-1) )
        loss_per_channel = (numerator / denominator).clamp(1e-12) ** (1.0 / self.p)
        if self.pool == 'sum':
            return loss_per_channel.sum()
        else:
            return loss_per_channel.mean()
