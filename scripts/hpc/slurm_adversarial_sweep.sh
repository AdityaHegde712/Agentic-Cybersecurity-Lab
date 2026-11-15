#!/bin/bash
#SBATCH -J adv_imperceptible_sweep
#SBATCH -o logs/slurm_adv_%j.out
#SBATCH -e logs/slurm_adv_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-NAIRR260371

# ==============================================================================
# TACC Vista HPC Batch Script: Imperceptible & Adversarial Attack Sweep
# Hardware: NVIDIA Grace-Hopper GH200 (1 Node, 1x GH200 GPU, 1hr Walltime)
# Allocation: TG-NAIRR260371
# ==============================================================================

set -euo pipefail

mkdir -p logs
mkdir -p results/experiments/adversarial_imperceptible

echo "================================================================================"
echo "STARTING ADVERSARIAL & IMPERCEPTIBLE SWEEP ON TACC VISTA GH200"
echo "Job ID: ${SLURM_JOB_ID:-LOCAL}"
echo "Node: $(hostname)"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "================================================================================"

# 1. Environment & Module Initialization
if command -v module &> /dev/null; then
    echo "Loading Lmod module stack..."
    module purge || true
    module load gcc/15.1.0 cuda/12.9 python3/3.11.8 || true
    module load nvpl || true
fi

# Activate virtual environment
if [ -d "$SCRATCH/venvs/agentic_cyber" ]; then
    echo "Activating virtualenv from \$SCRATCH..."
    source "$SCRATCH/venvs/agentic_cyber/bin/activate"
elif [ -d "$WORK/venvs/agentic_cyber" ]; then
    echo "Activating virtualenv from \$WORK..."
    source "$WORK/venvs/agentic_cyber/bin/activate"
fi

export PYTHONPATH=".:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "Executing Imperceptible & Adversarial Experiment..."
python scripts/test_imperceptible_adversarial.py

echo "================================================================================"
echo "EXPERIMENT COMPLETE. ARTIFACTS IN results/experiments/adversarial_imperceptible/"
echo "================================================================================"
