#!/bin/bash
# =============================================================================
# SLURM job script — SFNO training
# Cluster   : GPU-AI partition, node gpu1
# Hardware  : 3 GPUs · 64 GB RAM · 32 CPUs
# Launcher  : torchrun (NCCL backend)
# =============================================================================

#SBATCH --job-name=sfno
#SBATCH --output=logs/sfno_%j.out
#SBATCH --error=logs/sfno_%j.err

# --- Resources (mirrors your srun command exactly) ---------------------------
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1                   # single task; torchrun spawns the 3 processes

# --- Time --------------------------------------------------------------------


# =============================================================================
# Environment
# =============================================================================

echo "========================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $SLURM_JOB_GPUS"
echo "Start time  : $(date)"
echo "Working dir : $SLURM_SUBMIT_DIR"
echo "========================================================"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

# --- Conda -------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vishnu

echo "Python      : $(which python)"
echo "Torch       : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail  : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count   : $(python -c 'import torch; print(torch.cuda.device_count())')"

# =============================================================================
# Launch
# =============================================================================

torchrun --nproc_per_node=3 --master_port=29501 \
    train.py \
        --phase 3 \

EXIT_CODE=$?

echo "========================================================"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
echo "========================================================"

exit $EXIT_CODE
