#!/bin/bash
#SBATCH -J surrogate_master_pipeline
#SBATCH -o logs/slurm_master_%j.out
#SBATCH -e logs/slurm_master_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-NAIRR260371

# ==============================================================================
# Master HPC Pipeline: Multi-Dataset Plant Surrogate Training and Evaluation
# Hardware Target: TACC Vista NVIDIA Grace-Hopper (GH200, 1x H200 96GB GPU)
# Orchestrates:
#   Step 1: Train & Evaluate CausalTCN on BATADAL
#   Step 2: Train & Evaluate CausalPatchTST on BATADAL
#   Step 3: Train & Evaluate CausalTCN on SWaT
#   Step 4: Train & Evaluate CausalPatchTST on SWaT
#   Step 5: Train & Evaluate CausalTCN on WADI
#   Step 6: Train & Evaluate CausalPatchTST on WADI
#
# Idempotent: Skips any step whose evaluation summary.json already exists.
# ==============================================================================

set -euo pipefail

mkdir -p logs
mkdir -p results/surrogates
mkdir -p results/surrogate_eval

echo "================================================================================"
echo "STARTING MASTER SURROGATE PIPELINE ON TACC VISTA GH200"
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

# Activate virtual environment if present
if [ -d "$SCRATCH/venvs/agentic_cyber" ]; then
    echo "Activating virtualenv from \$SCRATCH..."
    source "$SCRATCH/venvs/agentic_cyber/bin/activate"
elif [ -d "$WORK/venvs/agentic_cyber" ]; then
    echo "Activating virtualenv from \$WORK..."
    source "$WORK/venvs/agentic_cyber/bin/activate"
elif [ -d ".venv" ]; then
    echo "Activating local .venv..."
    source .venv/bin/activate || source .venv/Scripts/activate
fi

export PYTHONPATH=".:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Helper function to print banner
print_banner() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "================================================================================"
}

# Ensure processed parquets exist
if [ ! -f "data/processed/swat/train.parquet" ] || [ ! -f "data/processed/wadi/train.parquet" ]; then
    echo "Generating SWaT and WADI processed Parquets..."
    python scripts/prepare_swat_wadi_parquets.py || true
fi

EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-10}

# ------------------------------------------------------------------------------
# STEP 1: BATADAL CausalTCN (Train & Eval)
# ------------------------------------------------------------------------------
print_banner "STEP 1/6: BATADAL - CausalTCN"
STEP1_CKPT="results/surrogates/batadal_tcn/checkpoint_best.pt"
STEP1_SUMMARY="results/surrogate_eval/batadal_tcn_5family/summary.json"

# Fallback to existing champion checkpoint if present
if [ ! -f "$STEP1_CKPT" ] && [ -f "results/surrogates/vista_tcn_48h_150ep/tcn.pt" ]; then
    mkdir -p results/surrogates/batadal_tcn
    cp results/surrogates/vista_tcn_48h_150ep/tcn.pt "$STEP1_CKPT"
    echo "Copied existing champion checkpoint to $STEP1_CKPT"
fi

if [ -f "$STEP1_SUMMARY" ]; then
    echo "[SKIPPED] Step 1 already completed: $STEP1_SUMMARY exists."
else
    if [ ! -f "$STEP1_CKPT" ]; then
        echo "Training CausalTCN on BATADAL ($EPOCHS epochs, patience $PATIENCE, onecycle)..."
        python scripts/train_surrogate.py --dataset batadal --model tcn --epochs "$EPOCHS" --patience "$PATIENCE" --scheduler onecycle --device cuda
    fi
    echo "Evaluating CausalTCN on BATADAL (5-family benchmark)..."
    python scripts/run_surrogate_eval.py --dataset batadal --model tcn --checkpoint "$STEP1_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 2: BATADAL CausalPatchTST (Train & Eval)
# ------------------------------------------------------------------------------
print_banner "STEP 2/6: BATADAL - CausalPatchTST"
STEP2_CKPT="results/surrogates/batadal_patch_tst/checkpoint_best.pt"
STEP2_SUMMARY="results/surrogate_eval/batadal_patch_tst_5family/summary.json"

if [ -f "$STEP2_SUMMARY" ]; then
    echo "[SKIPPED] Step 2 already completed: $STEP2_SUMMARY exists."
else
    if [ ! -f "$STEP2_CKPT" ]; then
        echo "Training CausalPatchTST on BATADAL ($EPOCHS epochs, patience $PATIENCE, onecycle)..."
        python scripts/train_surrogate.py --dataset batadal --model patch_tst --epochs "$EPOCHS" --patience "$PATIENCE" --scheduler onecycle --device cuda
    fi
    echo "Evaluating CausalPatchTST on BATADAL (5-family benchmark)..."
    python scripts/run_surrogate_eval.py --dataset batadal --model patch_tst --checkpoint "$STEP2_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 3: SWaT CausalTCN (Train & Eval)
# ------------------------------------------------------------------------------
print_banner "STEP 3/6: SWaT - CausalTCN"
STEP3_CKPT="results/surrogates/swat_tcn/checkpoint_best.pt"
STEP3_SUMMARY="results/surrogate_eval/swat_tcn_5family/summary.json"

if [ -f "$STEP3_SUMMARY" ]; then
    echo "[SKIPPED] Step 3 already completed: $STEP3_SUMMARY exists."
else
    if [ ! -f "$STEP3_CKPT" ]; then
        echo "Training CausalTCN on SWaT ($EPOCHS epochs, patience $PATIENCE, onecycle)..."
        python scripts/train_surrogate.py --dataset swat --model tcn --epochs "$EPOCHS" --patience "$PATIENCE" --scheduler onecycle --device cuda
    fi
    echo "Evaluating CausalTCN on SWaT (5-family benchmark)..."
    python scripts/run_surrogate_eval.py --dataset swat --model tcn --checkpoint "$STEP3_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 4: SWaT CausalPatchTST (Train & Eval)
# ------------------------------------------------------------------------------
print_banner "STEP 4/6: SWaT - CausalPatchTST"
STEP4_CKPT="results/surrogates/swat_patch_tst/checkpoint_best.pt"
STEP4_SUMMARY="results/surrogate_eval/swat_patch_tst_5family/summary.json"

if [ -f "$STEP4_SUMMARY" ]; then
    echo "[SKIPPED] Step 4 already completed: $STEP4_SUMMARY exists."
else
    if [ ! -f "$STEP4_CKPT" ]; then
        echo "Training CausalPatchTST on SWaT ($EPOCHS epochs, patience $PATIENCE, onecycle)..."
        python scripts/train_surrogate.py --dataset swat --model patch_tst --epochs "$EPOCHS" --patience "$PATIENCE" --scheduler onecycle --device cuda
    fi
    echo "Evaluating CausalPatchTST on SWaT (5-family benchmark)..."
    python scripts/run_surrogate_eval.py --dataset swat --model patch_tst --checkpoint "$STEP4_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 5: WADI CausalTCN (Train & Eval)
# ------------------------------------------------------------------------------
print_banner "STEP 5/6: WADI - CausalTCN"
STEP5_CKPT="results/surrogates/wadi_tcn/checkpoint_best.pt"
STEP5_SUMMARY="results/surrogate_eval/wadi_tcn_5family/summary.json"

if [ -f "$STEP5_SUMMARY" ]; then
    echo "[SKIPPED] Step 5 already completed: $STEP5_SUMMARY exists."
else
    if [ ! -f "$STEP5_CKPT" ]; then
        echo "Training CausalTCN on WADI ($EPOCHS epochs, patience $PATIENCE, onecycle)..."
        python scripts/train_surrogate.py --dataset wadi --model tcn --epochs "$EPOCHS" --patience "$PATIENCE" --scheduler onecycle --device cuda
    fi
    echo "Evaluating CausalTCN on WADI (5-family benchmark)..."
    python scripts/run_surrogate_eval.py --dataset wadi --model tcn --checkpoint "$STEP5_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 6: WADI CausalPatchTST (Train & Eval)
# ------------------------------------------------------------------------------
print_banner "STEP 6/6: WADI - CausalPatchTST"
STEP6_CKPT="results/surrogates/wadi_patch_tst/checkpoint_best.pt"
STEP6_SUMMARY="results/surrogate_eval/wadi_patch_tst_5family/summary.json"

if [ -f "$STEP6_SUMMARY" ]; then
    echo "[SKIPPED] Step 6 already completed: $STEP6_SUMMARY exists."
else
    if [ ! -f "$STEP6_CKPT" ]; then
        echo "Training CausalPatchTST on WADI ($EPOCHS epochs, patience $PATIENCE, onecycle)..."
        python scripts/train_surrogate.py --dataset wadi --model patch_tst --epochs "$EPOCHS" --patience "$PATIENCE" --scheduler onecycle --device cuda
    fi
    echo "Evaluating CausalPatchTST on WADI (5-family benchmark)..."
    python scripts/run_surrogate_eval.py --dataset wadi --model patch_tst --checkpoint "$STEP6_CKPT" --eval-mode 5family --device cuda
fi

print_banner "ALL 6 PIPELINE STEPS COMPLETED SUCCESSFULLY!"
echo "Master execution finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
