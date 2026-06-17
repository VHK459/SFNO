import os
import wandb
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR , CosineAnnealingLR , CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import argparse
from torch_harmonics import RealSHT
# from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperatorNet as SFNO
from sfno import SphericalFourierNeuralOperatorNet as SFNO


from bounded_prefetcher import ERA5Prefetcher
# torch.autograd.set_detect_anomaly(True)
from bounded_prefetcher_val_test import ERA5PrefetcherValTest
from latweightedloss import LatWeightedLoss
from latweightedSphericalloss import LatWeightedSpectralLoss
from latweightedlossCrop import LatWeightedLossCrop
from latweightedSphericallossCrop import LatWeightedSpectralLossCrop


def setup_distributed():
    """Initialize the distributed environment using torchrun."""
    # torchrun sets these environment variables automatically
    dist.init_process_group(backend="nccl")
    
    # Get rank and world_size from the process group
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set the device for this process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_model(ds, channels):
    nlat, nlon = ds.latitude.size, ds.longitude.size
    # Create model
    model = SFNO(
        operator_type='driscoll-healy',
        img_size=(nlat, nlon),
        num_layers=8,
        scale_factor=2,
        embed_dim=384,
        pos_embed="latlon",
        use_mlp=True,
        activation_function = "gelu",
       normalization_layer = "layer_norm",
        hard_thresholding_fraction=1,
        in_chans=channels, 
        out_chans=channels 
    )
    return model


def get_validation_dataset(data_path):
    """
    Returns a validation xarray dataset covering 2017-01-01 to 2018-12-31.
    Mirrors the same zarr-opening pattern as the training dataset.
    """
    ds_val = xr.open_zarr(data_path, chunks={})
    ds_val = ds_val.sel(time=slice("2017-01-01", "2018-12-31"))
    return ds_val


def get_test_dataset(data_path):
    """
    Returns a test xarray dataset covering 2019-01-01 onwards.
    Mirrors the same zarr-opening pattern as the training dataset.
    """
    ds_test = xr.open_zarr(data_path, chunks={})
    ds_test = ds_test.sel(time=slice("2017-01-01", "2018-12-31"))
    return ds_test


def make_spectral_criterion(ds, device, spectral = True, crop = False):
    """
    Builds the LatWeightedSpectralLoss used in Phase 1 (and Phase 3).
    Centralised here so both phases share identical loss configuration.
    """
    var_list = ['u_component_of_wind_250', 'u_component_of_wind_500',
       'u_component_of_wind_850', 'v_component_of_wind_250',
       'v_component_of_wind_500', 'v_component_of_wind_850', 'temperature_250',
       'temperature_500', 'temperature_850', 'geopotential_250',
       'geopotential_500', 'geopotential_850', 'vertical_velocity_250',
       'vertical_velocity_500', 'vertical_velocity_850',
       'specific_humidity_250', 'specific_humidity_500',
       'specific_humidity_850', '10m_u_component_of_wind',
       '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure',
       'total_column_water_vapour']

    H, W = ds.latitude.size, ds.longitude.size
    lmax = H - 1
    sht = RealSHT(nlat=H, nlon=W, lmax=lmax, mmax=lmax).to(device)

    if crop == False:
        criterion = LatWeightedSpectralLoss(
            ds,
            sht=sht,
            alpha=2,
            lambda_spec=10,
            l_min=20,
            var_names=var_list,
            pool="mean",
            device=device,
            spectral = spectral
        )
    else:
        criterion = LatWeightedSpectralLossCrop(
            ds,
            sht=sht,
            alpha=2,
            lambda_spec=10,
            l_min=20,
            var_names=var_list,
            pool="mean",
            device=device,
            spectral = False
        )
    return criterion


def train_phase1(ds, args, rank, world_size, local_rank, model , ds_val):
    """Phase 1: Single-step prediction training with distributed setup."""
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process:
        print(f"Training on {world_size} GPUs")

    criterion = make_spectral_criterion(ds, device)

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    prefetcher = ERA5Prefetcher(
        ds,
        batch_size=32,
        queue_size=4,
        sequence_length=2,
        device=rank,
        check_nans=True
    )

    prefetcher_val = ERA5PrefetcherValTest(
        ds_val,
        batch_size=32,
        queue_size=4,
        sequence_length=2,
        device=rank,
        check_nans=True
    )

    prefetcher.start()
    prefetcher_val.start()
    total_steps = args.phase1_gradient_steps
    if is_main_process:
        print(f"Phase 1: {total_steps} samples, batch size {args.batch_size} (per GPU: {args.batch_size}).")
    
    # Schedulers
    scheduler1 = LinearLR(optimizer, start_factor=.001, end_factor=1.0, total_iters=args.phase1_gradient_steps)
    scheduler2 = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=1000,
        T_mult=1,
        eta_min=5e-5
    )
    main_scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler1, scheduler2], 
        milestones=[1000]
    )
    
    model.train()
    
    if is_main_process:
        gradient_steps = tqdm(range(total_steps), desc="Phase 1 Training")
    else:
        gradient_steps = range(total_steps)
    
    for step in gradient_steps:
    
        batch = prefetcher.get()
        x, y = batch[:,:,0,:,:].to(device), batch[:,:,1,:,:].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss, spectral = criterion(pred, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
                         model.parameters(),
                         max_norm=32 
                         )
        optimizer.step()

        if is_main_process and step % 10 == 0:
            wandb.log({
                "phase1/train_loss": loss.item(),
                "phase1/spectral_loss": spectral.item(),
                "phase1/grad_norm": norm,
                "phase1/learning_rate": optimizer.param_groups[0]['lr'],
                "phase1_step": step
            })
            if hasattr(gradient_steps, 'set_postfix'):
                gradient_steps.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                print(f"loss = {loss.item():.4f},  lr = {optimizer.param_groups[0]['lr']:.2e}")
    
        main_scheduler.step()
        
        # Save checkpoints (only main process)
        if is_main_process:
            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    batch_v = prefetcher_val.get()
                    x, y = batch_v[:,:,0,:,:].to(device), batch_v[:,:,1,:,:].to(device)
                    pred = model(x)
                    lossv, spectralv = criterion(pred, y)
                    wandb.log({
                "phase1/val_loss": lossv.item(),
                "phase1/val_spectral_loss": spectralv.item(),
                "phase1_step": step
            })
                model.train()
                    # del batch_v, x, y, pred
            if step % 1000 == 0:
                torch.save(model.module.state_dict(), f"sfno_phase1SPHERE_layernorm_{step}.pth")
                print(f"\nSaved Phase 1 checkpoint at step {step}")
        # del batch, x, y
    prefetcher.stop()
    prefetcher_val.stop()


def train_phase3(ds, args, rank, world_size, local_rank, model,ds_val):
    """Phase 3: Autoregressive training with distributed setup.
    Uses the same LatWeightedSpectralLoss as Phase 1.
    Loads from the latest Phase 1 checkpoint.
    """
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process:
        print(f"\n### Starting Phase 3: Autoregressive Training on {world_size} GPUs ###")

    # Load Phase 1 checkpoint
    phase1_ckpt = "sfno_phase1SPHERE_layernorm_10000.pth"
    if os.path.exists(phase1_ckpt):
        model.module.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        if is_main_process:
            print(f"Loaded Phase 1 checkpoint from '{phase1_ckpt}'")

    # Same loss as Phase 1
    criterion = make_spectral_criterion(ds, device)

    optimizer = optim.Adam(model.parameters(), lr=args.phase3_lr)
    
    # Autoregressive curriculum loop
    for ar_steps in range(args.phase3_ar_start, args.phase3_ar_end + 1):
        if is_main_process:
            print(f"\n--- Phase 3: Training with {ar_steps} autoregressive steps ---")
        
        prefetcher = ERA5Prefetcher(
            ds,
            batch_size=32,
            queue_size=4,
            sequence_length=ar_steps + 1,
            device=local_rank,
            check_nans=True
        )

        prefetcher_val = ERA5PrefetcherValTest(
        ds_val,
        batch_size=32,
        queue_size=4,
        sequence_length=ar_steps + 1,
        device=rank,
        check_nans=True
    )
        ll = (np.linspace(0,1,30)**(1/2)*1000).astype(int)
        lin_range = np.sort(np.append(ll,np.arange(990,1000)))
        prefetcher.start()
        prefetcher_val.start()
        if is_main_process:
            print(f"Phase 3 ({ar_steps} steps): {args.phase3_gradient_steps} samples.")
            gradient_steps = tqdm(range(args.phase3_gradient_steps), desc=f"AR Steps: {ar_steps}")
        else:
            gradient_steps = range(args.phase3_gradient_steps)
        
        running_loss = 0.0
        
        for step in gradient_steps:
            model.train()
            
            batch = prefetcher.get()
            initial_input = batch[:,:,0,:,:].squeeze().to(device)
            y_true = batch[:,:,1:,:,:].squeeze().to(device)
            
            # Autoregressive forward pass
            current_input = initial_input
            loss = 0.0
            spectral = 0.0
            optimizer.zero_grad()

            for t in range(ar_steps):
                pred = model(current_input)
                # criterion returns (total_loss, spectral_loss); accumulate total_loss
                step_loss, step_spectral = criterion(pred, y_true[:,:,t,:,:].squeeze())
                loss = loss + step_loss + step_spectral
                # spectral = spectral + step_spectral
                current_input = pred

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                         model.parameters(),
                         max_norm=float("inf")
                         )
            optimizer.step()

            ll = loss.item()
            running_loss += ll
            del batch

            if is_main_process and step % 10 == 0:
                wandb.log({
                    "phase3/total_loss": ll,
                    "phase3/grad_norm": norm,
                    "phase3/avg_step_loss": running_loss / (step + 1),
                    "phase3/ar_steps": ar_steps,
                    "phase3/learning_rate": optimizer.param_groups[0]['lr'],
                    
                })
                if hasattr(gradient_steps, 'set_postfix'):
                    gradient_steps.set_postfix(loss=f"{ll:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                    print(f'loss = {ll:.4f}')

        running_loss = 0.0
        prefetcher.stop()
    
        # Save checkpoint per AR curriculum stage
        if is_main_process:
            if (step % 1000) in lin_range:
                model.eval()
                with torch.no_grad():
                    batch_v = prefetcher_val.get()
                    initial_input = batch_v[:,:,0,:,:].squeeze().to(device)
                    y_true = batch_v[:,:,1:,:,:].squeeze().to(device)
                    
                    # Autoregressive forward pass
                    current_input = initial_input
                    lossv = 0.0
                    spectralv = 0.0
        
                    for t in range(ar_steps):
                        pred = model(current_input)
                        # criterion returns (total_loss, spectral_loss); accumulate total_loss
                        step_lossv, step_spectralv = criterion(pred, y_true[:,:,t,:,:].squeeze())
                        lossv = lossv + step_lossv + step_spectralv
                        # spectral = spectral + step_spectral
                        current_input = pred
        
                    ll = lossv.item()                
                    wandb.log({
                "phase3/val_loss": ll,
                
            })
                model.train()
            ckpt_name = f"sfno_phase3_ar{ar_steps}.pth"
            torch.save(model.module.state_dict(), ckpt_name)
            print(f"Saved checkpoint: {ckpt_name}")
    
    # Save final model
    if is_main_process:
        torch.save(model.module.state_dict(), "sfno_phase3_final.pth")
        print("\nTraining complete. Final model saved to 'sfno_phase3_final.pth'")

def train_phase4(ds, args, rank, world_size, local_rank, model , ds_val):
    """Phase 1: Single-step prediction training with distributed setup."""
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    

    if is_main_process:
        print(f"\n### Starting Phase 3: Autoregressive Training on {world_size} GPUs ###")

    # Load Phase 1 checkpoint
    phase3_ckpt = "sfno_phase1SPHERE_layernorm_10000.pth"
    if os.path.exists(phase3_ckpt):
        model.module.load_state_dict(torch.load(phase3_ckpt, map_location=device))
        if is_main_process:
            print(f"Loaded Phase 1 checkpoint from '{phase3_ckpt}'")
    # criterion = LatWeightedLossCrop(ds,p=2, device=device)
    criterion = make_spectral_criterion(ds, device, crop = True)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=200,
        T_mult=1,
        eta_min=5e-5
    )
    prefetcher = ERA5Prefetcher(
        ds,
        batch_size=32,
        queue_size=4,
        sequence_length=2,
        device=rank,
        check_nans=True
    )

    prefetcher_val = ERA5PrefetcherValTest(
        ds_val,
        batch_size=32,
        queue_size=4,
        sequence_length=2,
        device=rank,
        check_nans=True
    )
    ll = (np.linspace(0,1,30)**(1/2)*1000).astype(int)
    lin_range = np.sort(np.append(ll,np.arange(990,1000)))
    prefetcher.start()
    prefetcher_val.start()
    total_steps = args.phase4_gradient_steps
    if is_main_process:
        print(f"Phase 1: {total_steps} samples, batch size {args.batch_size} (per GPU: {args.batch_size}).")
    

    
    model.train()
    
    def crop(tensor, w_slice=(40, 65), h_slice=(64, 85)):
        return tensor[:, :, h_slice[0]:h_slice[1], w_slice[0]:w_slice[1]]
    
    if is_main_process:
        gradient_steps = tqdm(range(total_steps), desc="Phase 4 Training")
    else:
        gradient_steps = range(total_steps)
    
    for step in gradient_steps:
        batch = prefetcher.get()
        x, y = batch[:,:,0,:,:].to(device), batch[:,:,1,:,:].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(crop(pred), crop(y))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
                         model.parameters(),
                         max_norm=32 
                         )
        optimizer.step()

        if is_main_process:
            if step % 10 == 0:
                wandb.log({
                    "phase4/train_loss": loss.item(),
                    # "phase4/spectral_loss": spectral.item(),
                    "phase4/grad_norm": norm,
                    "phase4/learning_rate": optimizer.param_groups[0]['lr'],
                    "phase4_step": step
                })
                if hasattr(gradient_steps, 'set_postfix'):
                    gradient_steps.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                    print(f"loss = {loss.item():.4f},  lr = {optimizer.param_groups[0]['lr']:.2e}")
        
        # scheduler.step()
        
        # Save checkpoints (only main process)
        if is_main_process:
            if (step % 10) == 0:
                model.eval()
                with torch.no_grad():
                    batch_v = prefetcher_val.get()
                    x, y = batch_v[:,:,0,:,:].to(device), batch_v[:,:,1,:,:].to(device)
                    pred = model(x)
                    lossv = criterion(crop(pred), crop(y))
                    wandb.log({
                "phase4/val_loss": lossv.item(),
                "phase4_step": step
                })
                model.train()
                    # del batch_v, x, y, pred
            if step % 10 == 0:
                torch.save(model.module.state_dict(), f"sfno_phase4SPHERE_layernorm_{step}_final.pth")
                print(f"\nSaved Phase 1 checkpoint at step {step}")
        # del batch, x, y
    prefetcher.stop()
    prefetcher_val.stop()

def main():
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    parser = argparse.ArgumentParser(description='Distributed SFNO Training with torchrun')
    parser.add_argument('--data_path', type=str, 
                        default='/storage/vishnu/era5_stacked.zarr/',
                        help='Path to the dataset')
    
    parser.add_argument('--phase', type=str, choices=['1', '3', 'all' ,'4'], default='all',
                        help='Which training phase to run (phase 2 has been removed)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Total batch size across all GPUs')
    
    parser.add_argument('--num_workers', type=int, default=15,
                        help='Number of data loading workers per GPU')
    
    parser.add_argument('--phase1_gradient_steps', type=int, default=2,
                        help='Number of steps for Phase 1')
    
    parser.add_argument('--phase3_gradient_steps', type=int, default=3,
                        help='Number of steps per AR block in Phase 3')
    
    parser.add_argument('--phase3_ar_start', type=int, default=2,
                        help='Starting AR steps for Phase 3')
    
    parser.add_argument('--phase3_ar_end', type=int, default=4,
                        help='Ending AR steps for Phase 3')
    
    parser.add_argument('--phase3_lr', type=float, default=3e-5,
                        help='lr for phase 3')

    parser.add_argument('--phase4_gradient_steps', type=int, default=2,
                        help = 'Number of steps in phase4')
    
    args = parser.parse_args()




    if is_main_process:
        wandb.init(
            project='ERA5-train-distributed',
            name=f'run_phase_{args.phase}_bs_{args.batch_size}_gelu_layer_norm',
            config=vars(args),
            group=f'DDP_worldsize_{world_size}'
        )
        wandb.define_metric("phase1_step")
        wandb.define_metric("phase2_step")
        wandb.define_metric("phase3_step")

        wandb.define_metric("phase1/*", step_metric="phase1_step")
        wandb.define_metric("phase2/*", step_metric="phase2_step")
        wandb.define_metric("phase3/*", step_metric="phase3_step")
    
    # --- Training dataset: up to end of 2016 (original behaviour) ---
    ds_train = xr.open_zarr(args.data_path, chunks={})
    ds_train = ds_train.sel(time=slice(None, "2016-12-31"))

    # --- Validation dataset: 2017-01-01 to 2018-12-31 ---
    ds_val = get_validation_dataset(args.data_path)

    # --- Test dataset: 2019-01-01 onwards ---
    ds_test = get_test_dataset(args.data_path)

    if is_main_process:
        print('Datasets opened successfully')
        print(f"  Train : {ds_train.time.values[0]} → {ds_train.time.values[-1]}")
        print(f"  Val   : {ds_val.time.values[0]} → {ds_val.time.values[-1]}")
        print(f"  Test  : {ds_test.time.values[0]} → {ds_test.time.values[-1]}")

    device = torch.device(f'cuda:{local_rank}')
    model = get_model(ds=ds_train, channels=23).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    if is_main_process:
        wandb.watch(model, log="all", log_freq=1)

    if args.phase in ['1', 'all']:
        train_phase1(ds_train, args, rank, world_size, local_rank, model, ds_val)
        # Save a named final Phase 1 checkpoint so Phase 3 can load it
        if is_main_process:
            torch.save(model.module.state_dict(), "sfno_phase1.pth")
            print("Saved final Phase 1 checkpoint.")
        dist.barrier()  # Ensure all ranks wait before Phase 3 loads the checkpoint

    if args.phase in ['3', 'all']:
        train_phase3(ds_train, args, rank, world_size, local_rank, model, ds_val)

    if args.phase in ['4', 'all']:
        train_phase4(ds_train, args, rank, world_size, local_rank, model, ds_val)

    if is_main_process:
        wandb.finish()
    
    cleanup_distributed()

if __name__ == '__main__':
    main()