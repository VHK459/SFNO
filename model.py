"""
model.py
--------
Model construction and loss factory functions.
"""

import xarray as xr
import torch
import torch.nn as nn
from torch_harmonics import RealSHT

from sfno import SphericalFourierNeuralOperatorNet as SFNO
from latweightedSphericalloss import LatWeightedSpectralLoss
from latweightedSphericallossCrop import LatWeightedSpectralLossCrop
from config import TrainConfig
from config import build_channel_names
# ── Variables used by the spectral loss ─────────────────────────────────────

# ERA5_VAR_LIST_old = ['10m_u_component_of_wind', '10m_v_component_of_wind',
#         '2m_temperature', 'geopotential_250', 'geopotential_500',
#         'geopotential_850', 'land_sea_mask', 'soil_type',
#         'specific_humidity_250', 'specific_humidity_500',
#         'specific_humidity_850', 'surface_pressure', 'temperature_250',
#         'temperature_500', 'temperature_850', 'total_column_water_vapour',
#         'u_component_of_wind_250', 'u_component_of_wind_500',
#         'u_component_of_wind_850', 'v_component_of_wind_250',
#         'v_component_of_wind_500', 'v_component_of_wind_850']

ERA5_VAR_LIST = ['10m_u_component_of_wind', '10m_v_component_of_wind',
        '2m_temperature', 'geopotential_250', 'geopotential_500',
        'geopotential_850', 
        'specific_humidity_250', 'specific_humidity_500',
        'specific_humidity_850', 'temperature_250',
        'temperature_500', 'temperature_850', 'total_column_water_vapour',
        'u_component_of_wind_250', 'u_component_of_wind_500',
        'u_component_of_wind_850', 'v_component_of_wind_250',
        'v_component_of_wind_500', 'v_component_of_wind_850']



# ── Model ────────────────────────────────────────────────────────────────────

def build_model(ds: xr.Dataset, cfg: TrainConfig) -> nn.Module:
    """Instantiate SFNO from dataset grid and config."""
    nlat = ds.latitude.size
    nlon = ds.longitude.size
    return SFNO(
        operator_type="driscoll-healy",
        img_size=(nlat, nlon),
        num_layers=cfg.num_layers,
        scale_factor=cfg.scale_factor,
        embed_dim=cfg.embed_dim,
        pos_embed="latlon",
        use_mlp=True,
        activation_function="gelu",
        normalization_layer="layer_norm",
        hard_thresholding_fraction=cfg.hard_thresholding_fraction,
        in_chans=cfg.channels ,
        out_chans=cfg.channels - 2,
    )


# ── Loss ─────────────────────────────────────────────────────────────────────

def build_spectral_criterion(
    ds: xr.Dataset,
    device: torch.device,
    spectral: bool = True,
    crop: bool = False,
) -> nn.Module:
    """
    Build the latitude-weighted (optionally spectral) loss.

    Parameters
    ----------
    spectral : if True the spectral regularisation term is active.
    crop     : if True return the *crop* variant (no spectral term forced off).
    """
    H, W = ds.latitude.size, ds.longitude.size
    lmax = H - 1
    sht = RealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax).to(device)

    shared_kwargs = dict(
        ds=ds,
        sht=sht,
        alpha=2,
        lambda_spec=10,
        l_min=20,
        var_names=build_channel_names(TrainConfig.data_config, ignore_static = True),
        pool="mean",
        device=device,
    )

    if not crop:
        return LatWeightedSpectralLoss(**shared_kwargs, spectral=spectral)
    else:
        # Crop variant always has spectral=False (matches original behaviour)
        return LatWeightedSpectralLossCrop(**shared_kwargs, spectral=False)
