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

from latweightedloss import LatWeightedLoss
from latweightedSphericalloss import LatWeightedSpectralLoss



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

def train_phase1(ds, args,rank, world_size, local_rank,model):
    """Phase 1 : Single-step prediction training with distributed setup."""
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
    
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process:
        print(f"Training on {world_size} GPUs")
    

    H, W = ds.latitude.size, ds.longitude.size
    lmax = H - 1
    
    sht = RealSHT(
        nlat=H,
        nlon=W,
        lmax=lmax,
        mmax=lmax
    ).to(device)
    
    # Loss and optimizer
    # criterion = LatWeightedLoss(ds, p=2, device=device)
    criterion = LatWeightedSpectralLoss(
    ds,
    sht=sht,
    alpha=2,
    lambda_spec=10,
    l_min=20,
    var_names = var_list,
    pool="mean",
    device=device
)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    # Dataset and distributed sampler

    prefetcher = ERA5Prefetcher(
    ds,
    batch_size=32,
    queue_size=4,
    sequence_length=2,
    device=rank,
    check_nans=True
    )

    prefetcher.start()
    total_steps = args.phase1_gradient_steps
    if is_main_process:
        print(f"Phase 1 & 2: {total_steps}  samples, batch size {args.batch_size} (per GPU: {args.batch_size }).")
    
    # Schedulers
    scheduler1 = LinearLR(optimizer, start_factor=.001, end_factor=1.0, total_iters=args.phase1_gradient_steps)
       # 2. Define the Cosine Annealing Scheduler
    # T_max: Number of iterations (epochs) for a half-cycle (high -> low)
    # eta_min: The minimum learning rate (usually 0 or very small, e.g. 1e-6)
    scheduler2 = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=1000,      # Cycle length (epochs)
    T_mult=1,      # Factor to increase cycle length (1 = constant 1000)
    eta_min=5e-5   # Minimum LR at bottom of curve
    )

    main_scheduler = SequentialLR(
    optimizer, 
    schedulers=[scheduler1, scheduler2], 
    milestones=[1000] # Switch happens exactly at step 1000
)
    
    # Training loop
    model.train()
    
    
    if is_main_process:
        gradient_steps = tqdm(range(total_steps), desc="Phase 1 Training")
       
    else:
        gradient_steps = range(total_steps)
    
    for step in gradient_steps:
        # train_sampler.set_epoch(step)  # Shuffle data differently at each epoch
        
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

        
        if is_main_process:
            wandb.log({
                "phase1/train_loss": loss.item(),
                "phase1/spectral_loss": spectral.item(),
                "phase1/grad_norm": norm,
                "phase1/learning_rate": optimizer.param_groups[0]['lr'],
                "phase1/global_step": step
            })
            if hasattr(gradient_steps, 'set_postfix'):
                gradient_steps.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                print(f"loss = {loss.item():.4f},  lr = {optimizer.param_groups[0]['lr']:.2e}")
    
        main_scheduler.step()
        
        # Save checkpoints (only main process)
        if is_main_process:
            if step % 1000 == 0:
                torch.save(model.module.state_dict(), f"sfno_phase1SPHERE_layernorm_{step}.pth")
                print(f"\nSaved Phase 1 checkpoint at step {step}")
        del batch,x,y
        # torch.cuda.empty_cache()
    prefetcher.stop()




def train_phase2(ds, args,rank, world_size, local_rank,model):
    """Phase 2: Single-step prediction training with distributed setup."""
    
    
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process:
        print(f"Training on {world_size} GPUs")
        
    if os.path.exists("sfno_phase1.pth"):
        model.module.load_state_dict(torch.load("sfno_phase1_0.pth", map_location=device))
        if is_main_process:
            print("Loaded Phase 2 checkpoint")
    
    # Loss and optimizer
    criterion = LatWeightedLoss(ds, p=2, device=device)
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Dataset and distributed sampler

    prefetcher = ERA5Prefetcher(
    ds,
    batch_size=32,
    queue_size=4,
    sequence_length=2,
    device=rank,
    check_nans=True
    )

    prefetcher.start()
    total_steps =  args.phase2_gradient_steps
    if is_main_process:
        print(f"Phase 1 & 2: {total_steps}  samples, batch size {args.batch_size} (per GPU: {args.batch_size }).")
    
    # Schedulers
    scheduler1 = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=args.phase2_gradient_steps)
   
    
    # Training loop
    model.train()
    
    
    if is_main_process:
        gradient_steps = tqdm(range(total_steps), desc="Phase 2 Training")

    else:
        gradient_steps = range(total_steps)
    
    for step in gradient_steps:
        # train_sampler.set_epoch(step)  # Shuffle data differently at each epoch
        
        batch = prefetcher.get()
        x, y = batch[:,:,0,:,:].to(device), batch[:,:,1,:,:].to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
                         model.parameters(),
                         max_norm=32  # no clipping, just measure
                         )
        optimizer.step()

        
        if is_main_process:
            wandb.log({
                "phase2/train_loss": loss.item(),
                "phase2/learning_rate": optimizer.param_groups[0]['lr'],
                "phase2/global_step": step
            })
            if hasattr(gradient_steps, 'set_postfix'):
                gradient_steps.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                print(f"loss = {loss.item():.4f},  lr = {optimizer.param_groups[0]['lr']:.2e}")
    
        scheduler1.step()
        
        # Save checkpoints (only main process)
        if is_main_process:
            if step == args.phase2_gradient_steps - 1:
                torch.save(model.module.state_dict(), "sfno_phase2.pth")
                print(f"\nSaved Phase 1 checkpoint at step {step}")
                
        del batch,x,y
        # torch.cuda.empty_cache()
    prefetcher.stop()


def get_lat_loss(ds , p , device):
    return LatWeightedLoss(ds, p=2, device=device)

def train_phase3(ds, args,rank, world_size, local_rank,model):
    """Phase 3: Autoregressive training with distributed setup."""
   
    
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process:
        print(f"\n### Starting Phase 3: Autoregressive Training on {world_size} GPUs ###")


    # Load Phase 2 checkpoint
    if os.path.exists("sfno_phase2.pth"):
        model.module.load_state_dict(torch.load("sfno_phase2.pth", map_location=device))
        if is_main_process:
            print("Loaded Phase 2 checkpoint")
    
    # Wrap with DDP
 
    # Loss and optimizer
    criterion = LatWeightedLoss(ds, p=2, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.phase3_lr)  # Fine-tuning LR
    
    # Autoregressive training loop
    for ar_steps in range(args.phase3_ar_start, args.phase3_ar_end + 1):
        if is_main_process:
            print(f"\n--- Phase 3: Training with {ar_steps} autoregressive steps ---")
        
        # Dataset
        prefetcher = ERA5Prefetcher(
            ds,
            batch_size=32,
            queue_size=15,
            sequence_length=ar_steps + 1,
            device=local_rank,
            check_nans=True
            )

        prefetcher.start()
       
        
        if is_main_process:
            print(f"Phase 3 ({ar_steps} steps): {args.phase3_gradient_steps}  samples .")
            gradient_steps = tqdm(range(args.phase3_gradient_steps), desc=f"AR Steps: {ar_steps}")
        else:
            gradient_steps = range(args.phase3_gradient_steps)
        
        running_loss = 0.0
        
        for step in gradient_steps:
            # train_sampler.set_epoch(step + ar_steps * 1000)  # Ensure different shuffling
            model.train()
            
            batch = prefetcher.get()

            
           
            initial_input = batch[:,:,0,:,:].squeeze().to(device)
            y_true =  batch[:,:,1:,:,:].squeeze().to(device)
            
            # Autoregressive forward pass
            current_input = initial_input
            # loss = torch.tensor(0.0, requires_grad=True, device=device, dtype=initial_input.dtype)
            loss = 0.0
            optimizer.zero_grad()
            for t in range(ar_steps):
               
                pred = model(current_input)
               
                loss = loss + criterion(pred, y_true[:,:, t, :, :].squeeze())
                current_input = pred
                
                # del pred
                  
                
                # Backward
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                         model.parameters(),
                         max_norm=float("inf")  # no clipping, just measure
                         )


            optimizer.step()

            ll = loss.item()
            running_loss += loss.item()
            del batch
            # torch.cuda.empty_cache()
        
            if is_main_process:
                wandb.log({
                    "phase3/total_loss": ll,
                    "phase3/grad_norm": norm,
                    "phase3/avg_step_loss": running_loss / ar_steps, # Average loss per time step
                    "phase3/ar_steps": ar_steps, # Track which curriculum stage we are in
                    "phase3/learning_rate": optimizer.param_groups[0]['lr']
                })
                if hasattr(gradient_steps, 'set_postfix'):
                    gradient_steps.set_postfix(loss=f"{running_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                    print(f'loss = {running_loss:.4f}')
        running_loss = 0.0
        prefetcher.stop()
        # Save checkpoint (only main process)
        if is_main_process:
            ckpt_name = f"sfno_phase3_ar{ar_steps}.pth"
            torch.save(model.module.state_dict(), ckpt_name)
            print(f"Saved checkpoint: {ckpt_name}")
    
    # Save final model
    if is_main_process:
        torch.save(model.module.state_dict(), "sfno_phase3_final.pth")
        print("\nTraining complete. Final model saved to 'sfno_phase3_final.pth'")
    
    


def main():
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    parser = argparse.ArgumentParser(description='Distributed SFNO Training with torchrun')
    parser.add_argument('--data_path', type=str, 
                        default='/storage/vishnu/era5_stacked.zarr/',
                        help='Path to the dataset')
    
    parser.add_argument('--phase', type=str, choices=['1', '3', 'all','2'], default='all',
                        help='Which training phase to run')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Total batch size across all GPUs')
    
    parser.add_argument('--num_workers', type=int, default=15,
                        help='Number of data loading workers per GPU')
    
    parser.add_argument('--phase1_gradient_steps', type=int, default=2,
                        help='Number of steps for Phase 1')
    
    parser.add_argument('--phase2_gradient_steps', type=int, default=2,
                        help='Number of steps for Phase 2')
    
    parser.add_argument('--phase3_gradient_steps', type=int, default=3,
                        help='Number of steps per AR block in Phase 3')
    
    parser.add_argument('--phase3_ar_start', type=int, default=2,
                        help='Starting AR steps for Phase 3')
    
    parser.add_argument('--phase3_ar_end', type=int, default=5,
                        help='Ending AR steps for Phase 3')
    
    parser.add_argument('--phase3_lr', type=float, default=3e-7,
                        help='lr for phase 3')
    
    args = parser.parse_args()

    if is_main_process:
        wandb.init(
            project='ERA5-train-distributed',
            name=f'run_phase_{args.phase}_bs_{args.batch_size}_gelu_layer_norm',
            config=vars(args),
            group=f'DDP_worldsize_{world_size}'
        )
    
    ds = xr.open_zarr(args.data_path, chunks={})
    ds = ds.sel(time = slice(None,"2017-12-31"))
    if is_main_process:
        print('dataset opened successfully')
    device = torch.device(f'cuda:{local_rank}')
    model = get_model(ds=ds, channels=23).to(device)
    

    model = DDP(model, device_ids=[local_rank],broadcast_buffers=False)
    # Run training phases (torchrun handles process spawning)
    if is_main_process:
        wandb.watch(model, log="all", log_freq=10)
    if args.phase in ['1', 'all']:
        train_phase1(ds, args,rank, world_size, local_rank,model)

    if args.phase in ['2', 'all']:
        train_phase2(ds, args,rank, world_size, local_rank,model)

        
    if args.phase in ['3', 'all']:
        train_phase3(ds, args,rank, world_size, local_rank,model)

    if is_main_process:
        wandb.finish()
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
