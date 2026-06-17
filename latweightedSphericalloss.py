
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

import torch
import torch.nn as nn
import xarray as xr
import numpy as np

class LatWeightedSpectralLoss(nn.Module):
    def __init__(
        self,
        ds: xr.Dataset,
        var_names: list,  # <--- NEW: Pass your list of variable names here
        sht,              # torch_harmonics RealSHT
        p: int = 2,
        alpha: float = 2.0,
        lambda_spec: float = 0.3,
        l_min: int = 20,
        device: str = "cpu",
        pool: str = "sum",
        spectral = True
    ):
        super().__init__()

        self.p = p
        self.pool = pool
        self.alpha = alpha
        self.lambda_spec = lambda_spec
        self.l_min = l_min
        self.sht = sht
        self.spectral = spectral

        # 1. LATITUDE WEIGHTS (Spatial Area Correction)
        lat_weights = self._get_lat_weights(ds)
        lat_weights = torch.tensor(lat_weights, dtype=torch.float32, device=device)
        self.register_buffer("lat_weights", lat_weights.view(1, 1, -1, 1))

        # 2. VARIABLE CHANNEL WEIGHTS (FCN3 Logic)
        channel_weights = self._get_channel_weights(var_names)
        channel_weights = torch.tensor(channel_weights, dtype=torch.float32, device=device)
        # Shape (1, C, 1, 1) for broadcasting against (B, C, H, W)
        self.register_buffer("channel_weights", channel_weights.view(1, -1, 1, 1))

    def _get_channel_weights(self, var_names):
        """Parses variable names to assign FCN3 weights."""
        weights = []
        for name in var_names:
            # Case 1: 2m Temperature 
            if '2m_temperature' in name:
                weights.append(1.0)
            
            # Case 2: Other Surface Variables
            elif any(x in name for x in ['10m_u', '10m_v', 'mean_sea_level', 'total_column', 'surface']):
                weights.append(0.1)
            
            # Case 3: Atmospheric Variables (Weight = Pressure / 1000)
            else:
                try:
                    # Extract the pressure level (last part of string, e.g., 'wind_250' -> 250)
                    pressure = int(name.split('_')[-1])
                    weights.append(pressure / 1000.0)
                except ValueError:
                    # Fallback if naming convention fails
                    print(f"Warning: Could not parse pressure for {name}, defaulting to 1.0")
                    weights.append(1.0)
        
        print(f"Assigned Channel Weights: {weights}")
        return np.array(weights)

    def _get_lat_weights(self, ds: xr.Dataset) -> np.ndarray:
        # ... (Your existing latitude weight code remains exactly the same) ...
        def _assert_increasing(x):
            if not (np.diff(x) > 0).all(): raise ValueError("array is not increasing")
        def _latitude_cell_bounds(x):
            pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
            return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])
        def _cell_area_from_latitude(points):
            bounds = _latitude_cell_bounds(points)
            _assert_increasing(bounds)
            upper, lower = bounds[1:], bounds[:-1]
            return np.sin(upper) - np.sin(lower)
        
        w = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
        w /= np.mean(w)
        return w

    def sht_loss(self, prediction, target):
        # (B,C,L,M) complex
        pred_hat = self.sht(prediction)
        gt_hat   = self.sht(target)

        B, C, L, M = pred_hat.shape
        l = torch.arange(L, device=prediction.device).float()
        
        # Spectral frequency weighting (High frequency emphasis)
        raw_weight = (l + 1) ** self.alpha
        spec_weight = raw_weight[None, None, :, None] # (1, 1, L, 1)

        # 1. Calculate absolute spectral difference
        diff = torch.abs(pred_hat - gt_hat)
        
        # 2. Apply Spectral Weighting (alpha)
        weighted_diff = spec_weight * diff

        # 3. Apply Channel Weighting (FCN3 weights)
        # self.channel_weights is (1, C, 1, 1), so it broadcasts correctly here
        total_weighted_diff = weighted_diff * self.channel_weights

        # Average over batch and spatial modes, Sum over Channels (or mean depending on preference)
        # FCN3 sums the weighted components, but usually we mean over batch
        return total_weighted_diff.mean()

    def forward(self, prediction, target) -> torch.Tensor:
        # ---- Spatial Loss ----
        diff = (target - prediction) ** self.p
        
        # weighted spatial average (Latitude correction)
        numerator = torch.sum(self.lat_weights * diff, dim=(-2, -1))
        denominator = torch.sum(self.lat_weights)
        
        # Loss per channel (B, C)
        loss_per_channel = (numerator / denominator)

        # Apply FCN3 Channel Weights
        # channel_weights is (1, C, 1, 1) -> squeeze to (1, C)
        w = self.channel_weights.view(1, -1)
        loss_per_channel_weighted = loss_per_channel * w

        if self.pool == 'sum':
            L_spatial = loss_per_channel_weighted.sum()
        else:
            L_spatial = loss_per_channel_weighted.mean()

        # ---- Spectral Loss ----
        # (Already includes channel weighting inside the method now)

        if self.spectral:
            L_spec = self.sht_loss(prediction, target)
            return (L_spatial + self.lambda_spec * L_spec), L_spec
        else:
            return L_spatial

# class LatWeightedSpectralLoss(nn.Module):
#     def __init__(
#         self,
#         ds: xr.Dataset,
#         sht,                 # torch_harmonics RealSHT
#         p: int = 2,
#         alpha: float = 2.0,  # high-frequency emphasis
#         lambda_spec: float = 0.3,
#         l_min: int = 20,     # restrict to high-l
#         device: str = "cpu",
#         pool: str = "sum"
#     ):
#         super().__init__()

#         self.p = p
#         self.pool = pool
#         self.alpha = alpha
#         self.lambda_spec = lambda_spec
#         self.l_min = l_min
#         self.sht = sht
        
#         weights = self._get_lat_weights(ds)
#         weights = torch.tensor(weights, dtype=torch.float32, device=device)
#         self.register_buffer("weights", weights.view(1, 1, -1, 1))

#     # --------- unchanged ----------
#     def _get_lat_weights(self, ds: xr.Dataset) -> np.ndarray:

#         def _assert_increasing(x):
#             if not (np.diff(x) > 0).all():
#                 raise ValueError("array is not increasing")

#         def _latitude_cell_bounds(x):
#             pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
#             return np.concatenate(
#                 [-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2]
#             )

#         def _cell_area_from_latitude(points):
#             bounds = _latitude_cell_bounds(points)
#             _assert_increasing(bounds)
#             upper = bounds[1:]
#             lower = bounds[:-1]
#             return np.sin(upper) - np.sin(lower)

#         weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
#         weights /= np.mean(weights)
#         return weights

#     # --------- SHT spectral loss ----------
#     def sht_loss(self, prediction, target):

#         # (B,C,L,M) complex
#         pred_hat = self.sht(prediction)
#         gt_hat   = self.sht(target)

#         B, C, L, M = pred_hat.shape

#         l = torch.arange(L, device=prediction.device).float()
#         print(f" B, C, L, M is {B, C, L, M} and l_min is {self.l_min} pred_hat.shape")
#         # high-l mask
#         mask = (l >= self.l_min).float()

#         raw_weight = (l + 1) ** self.alpha
#         norm_weight = raw_weight / raw_weight.max()
#         # weight = mask * norm_weight
#         weight = raw_weight[None, None, :, None]

#         loss = weight * torch.abs(pred_hat - gt_hat) 

#         # return loss.sum() / (mask.sum() + 1e-8)

#         return loss.mean()
        
#         # We sum over L and M, but average over B and C to match spatial loss behavior
#         # spectral_diff = torch.abs(pred_hat - gt_hat) ** 2
#         # weighted_diff = weight * spectral_diff
        
#         # Sum over spectral modes (L, M), mean over batch/channel
#         # We divide by the number of active modes to keep scale consistent
#         # active_modes = mask.sum() * M 
#         # loss = weighted_diff.sum(dim=(-2, -1)) / (active_modes + 1e-8)
        
#         # return spectral_diff.mean()

#     # --------- forward ----------
#     def forward(self, prediction, target) -> torch.Tensor:

#         # ---- spatial (your original) ----
#         diff = (target - prediction) ** self.p
#         numerator = torch.sum(self.weights * diff, dim=(-2, -1))
#         denominator = torch.sum(self.weights)

#         loss_per_channel = (numerator / denominator).clamp(1e-12)# ** (1.0 / self.p)

#         if self.pool == 'sum':
#             L_spatial = loss_per_channel.sum()
#         else:
#             L_spatial = loss_per_channel.mean()

#         # ---- spectral ----
#         L_spec = self.sht_loss(prediction, target)
        
#         return (L_spatial + self.lambda_spec * L_spec), L_spec

