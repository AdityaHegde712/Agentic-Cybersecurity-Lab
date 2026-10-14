#!/bin/bash
#SBATCH -J surrogate_bakeoff_vista
#SBATCH -o results/surrogates/vista_bakeoff/logs/%x_%j.out
#SBATCH -e results/surrogates/vista_bakeoff/logs/%x_%j.err
#SBATCH -p gh
#SBATCH -A TG-NAIRR260371
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 02:00:00
#SBATCH --mail-type=FAIL,END

# ==============================================================================
# TACC Vista GH200 Production Batch Script
# Task 8: Dynamics Surrogate Model Bake-Off Suite
# Architecture: NVIDIA GH200 Grace Hopper Superchip (72 Arm cores, 96GB HBM3 GPU)
# Target Deliverable: November 18 Report
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Nodes Allocated:  $SLURM_JOB_NODELIST"
echo "Submit Directory: $SLURM_SUBMIT_DIR"
echo "Start Time:       $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================================"

# Navigate to submission directory
cd "$SLURM_SUBMIT_DIR"

# Create log directory
mkdir -p results/surrogates/vista_bakeoff/logs

# Environment initialization for TACC Vista GH200
module purge || true
module load gcc || true
module load cuda || true
module load python3 || true

# Activate Python environment
if [ -f "$SCRATCH/venvs/sengupta_cyber/bin/activate" ]; then
    echo "Activating Scratch venv ($SCRATCH/venvs/sengupta_cyber)..."
    source "$SCRATCH/venvs/sengupta_cyber/bin/activate"
elif [ -d ".venv" ]; then
    echo "Activating local .venv..."
    source .venv/bin/activate
elif [ -d "$WORK/venvs/vista-gh200" ]; then
    echo "Activating Vista work venv..."
    source "$WORK/venvs/vista-gh200/bin/activate"
fi

# Log hardware and CUDA state
echo "--- GPU Architecture & Driver Status ---"
nvidia-smi || echo "nvidia-smi unavailable"

echo "--- Python & PyTorch CUDA Sanity Check ---"
python3 -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Execute surrogate bake-off training suite
echo "--- Launching Surrogate Model Bake-Off Suite ---"
python3 -u scripts/train_surrogates_vista.py \
    --model all \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --context-length 24 \
    --gamma-mass 0.1 \
    --device cuda \
    --output-dir results/surrogates/vista_bakeoff \
    --train-path data/processed/batadal/train.parquet \
    --val-path data/processed/batadal/val.parquet \
    --manifest-path configs/manifests/batadal-channel-manifest-v1.json

echo "========================================================================"
echo "Bake-Off Complete: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Artifacts written to results/surrogates/vista_bakeoff/"
echo "========================================================================"
