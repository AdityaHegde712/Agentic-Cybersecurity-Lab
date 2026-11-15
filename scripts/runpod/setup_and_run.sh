#!/usr/bin/env bash
# ==============================================================================
# RunPod Cloud Offload Pipeline: Multi-Dataset Plant Surrogate Training & Eval
# Target Environment: RunPod Pod (RTX 4090 / 5090 or GH200 / A100)
# Orchestrates:
#   0. Environment bootstrap, dependencies installation, data generation
#   Pre-flight: Fast 1-epoch reduced-batch smoke test
#   Step 1: Train CausalTCN on SWaT (150 epochs, patience 10, OneCycleLR)
#   Step 2: Evaluate CausalTCN on SWaT (5-family benchmark across 50 scenarios)
#   Step 3: Train CausalTCN on WADI (150 epochs, patience 10, OneCycleLR)
#   Step 4: Evaluate CausalTCN on WADI (5-family benchmark across 50 scenarios)
#   Step 5: 10,000-scenario high-throughput benchmark across BATADAL, SWaT, WADI
#   Step 6: Git commit and push checkpoints & evaluation artifacts
#
# Idempotent: Automatically skips any stage whose summary.json or checkpoint exists.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p results/surrogates
mkdir -p results/surrogate_eval

print_banner() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "================================================================================"
}

print_banner "INITIALIZING RUNPOD TRAINING & EVALUATION WORKFLOW"
echo "Project Root: ${PROJECT_ROOT}"
echo "Hostname: $(hostname)"
echo "OS / Architecture: $(uname -s) $(uname -m)"

# ------------------------------------------------------------------------------
# 0. ENVIRONMENT SETUP & DEPENDENCY VERIFICATION
# ------------------------------------------------------------------------------
print_banner "ENVIRONMENT SETUP & DEPENDENCIES"

# Locate python binary
if command -v python3 &> /dev/null; then
    PY_BIN="python3"
elif command -v python &> /dev/null; then
    PY_BIN="python"
else
    echo "ERROR: Neither python3 nor python was found in PATH." >&2
    exit 1
fi
echo "Using base Python binary: $(${PY_BIN} --version)"

# Manage virtual environment
if [ -d "venv" ]; then
    echo "Activating existing venv..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating existing .venv..."
    source .venv/bin/activate
else
    echo "Creating virtual environment 'venv'..."
    ${PY_BIN} -m venv venv
    source venv/bin/activate
fi

# Verify / install required packages
echo "Verifying Python dependencies..."
python -m pip install --upgrade pip
python -m pip install torch pandas pyarrow scipy matplotlib tqdm

# Check CUDA GPU availability
echo "Checking CUDA device availability..."
python -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
'

export PYTHONPATH=".:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Ensure SWaT and WADI processed parquets exist
if [ ! -f "data/processed/swat/train.parquet" ] || [ ! -f "data/processed/wadi/train.parquet" ]; then
    print_banner "GENERATING SWAT AND WADI PARQUETS"
    python scripts/prepare_swat_wadi_parquets.py
fi

# ------------------------------------------------------------------------------
# PRE-FLIGHT SMOKE TEST
# ------------------------------------------------------------------------------
print_banner "PRE-FLIGHT SMOKE TEST: SWaT & WADI PIPELINE CHECK"
echo "Running fast 1-epoch reduced-batch smoke training on SWaT..."
python scripts/train_surrogate.py --dataset swat --model tcn --smoke --device cuda

echo "Running fast 1-epoch reduced-batch smoke training on WADI..."
python scripts/train_surrogate.py --dataset wadi --model tcn --smoke --device cuda

echo "Running fast evaluator smoke check on SWaT..."
python scripts/run_surrogate_eval.py --dataset swat --model tcn --checkpoint results/surrogates/swat_tcn/checkpoint_best.pt --smoke --device cuda

echo "Pre-flight smoke tests passed successfully."

# ------------------------------------------------------------------------------
# STEP 1: TRAIN CAUSALTCN ON SWAT
# ------------------------------------------------------------------------------
print_banner "STEP 1/6: TRAIN CAUSALTCN ON SWAT (150 Epochs, Patience 10, OneCycleLR)"
SWAT_CKPT="results/surrogates/swat_tcn/checkpoint_best.pt"

if [ -f "$SWAT_CKPT" ] && [ -f "results/surrogates/swat_tcn/training_history.json" ]; then
    # Check if this was a smoke run (1 epoch) or full run
    EPOCH_COUNT=$(python -c "import json; h=json.load(open('results/surrogates/swat_tcn/training_history.json')); print(len(h))" 2>/dev/null || echo "0")
    if [ "$EPOCH_COUNT" -gt 1 ]; then
        echo "[SKIPPED] SWaT CausalTCN training already completed ($EPOCH_COUNT epochs)."
    else
        echo "Overwriting previous smoke checkpoint with full 150-epoch training run..."
        python scripts/train_surrogate.py --dataset swat --model tcn --epochs 150 --patience 10 --scheduler onecycle --device cuda
    fi
else
    python scripts/train_surrogate.py --dataset swat --model tcn --epochs 150 --patience 10 --scheduler onecycle --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 2: EVALUATE CAUSALTCN ON SWAT (5-FAMILY BENCHMARK)
# ------------------------------------------------------------------------------
print_banner "STEP 2/6: EVALUATE CAUSALTCN ON SWAT (5-FAMILY BENCHMARK)"
SWAT_5FAM_SUMMARY="results/surrogate_eval/swat_tcn_5family/summary.json"

if [ -f "$SWAT_5FAM_SUMMARY" ]; then
    SCENARIO_COUNT=$(python -c "import json; s=json.load(open('$SWAT_5FAM_SUMMARY')); print(s.get('n_scenarios', 0))" 2>/dev/null || echo "0")
    if [ "$SCENARIO_COUNT" -ge 50 ]; then
        echo "[SKIPPED] SWaT 5-family evaluation already completed ($SCENARIO_COUNT scenarios)."
    else
        echo "Re-running SWaT 5-family benchmark across full 50 scenarios..."
        python scripts/run_surrogate_eval.py --dataset swat --model tcn --checkpoint "$SWAT_CKPT" --eval-mode 5family --device cuda
    fi
else
    python scripts/run_surrogate_eval.py --dataset swat --model tcn --checkpoint "$SWAT_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 3: TRAIN CAUSALTCN ON WADI
# ------------------------------------------------------------------------------
print_banner "STEP 3/6: TRAIN CAUSALTCN ON WADI (150 Epochs, Patience 10, OneCycleLR)"
WADI_CKPT="results/surrogates/wadi_tcn/checkpoint_best.pt"

if [ -f "$WADI_CKPT" ] && [ -f "results/surrogates/wadi_tcn/training_history.json" ]; then
    EPOCH_COUNT=$(python -c "import json; h=json.load(open('results/surrogates/wadi_tcn/training_history.json')); print(len(h))" 2>/dev/null || echo "0")
    if [ "$EPOCH_COUNT" -gt 1 ]; then
        echo "[SKIPPED] WADI CausalTCN training already completed ($EPOCH_COUNT epochs)."
    else
        echo "Overwriting previous smoke checkpoint with full 150-epoch training run..."
        python scripts/train_surrogate.py --dataset wadi --model tcn --epochs 150 --patience 10 --scheduler onecycle --device cuda
    fi
else
    python scripts/train_surrogate.py --dataset wadi --model tcn --epochs 150 --patience 10 --scheduler onecycle --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 4: EVALUATE CAUSALTCN ON WADI (5-FAMILY BENCHMARK)
# ------------------------------------------------------------------------------
print_banner "STEP 4/6: EVALUATE CAUSALTCN ON WADI (5-FAMILY BENCHMARK)"
WADI_5FAM_SUMMARY="results/surrogate_eval/wadi_tcn_5family/summary.json"

if [ -f "$WADI_5FAM_SUMMARY" ]; then
    SCENARIO_COUNT=$(python -c "import json; s=json.load(open('$WADI_5FAM_SUMMARY')); print(s.get('n_scenarios', 0))" 2>/dev/null || echo "0")
    if [ "$SCENARIO_COUNT" -ge 50 ]; then
        echo "[SKIPPED] WADI 5-family evaluation already completed ($SCENARIO_COUNT scenarios)."
    else
        echo "Re-running WADI 5-family benchmark across full 50 scenarios..."
        python scripts/run_surrogate_eval.py --dataset wadi --model tcn --checkpoint "$WADI_CKPT" --eval-mode 5family --device cuda
    fi
else
    python scripts/run_surrogate_eval.py --dataset wadi --model tcn --checkpoint "$WADI_CKPT" --eval-mode 5family --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 5: 10,000-SCENARIO HIGH-THROUGHPUT EVALUATION ACROSS BATADAL, SWAT, WADI
# ------------------------------------------------------------------------------
print_banner "STEP 5/6: 10,000-SCENARIO HIGH-THROUGHPUT EVALUATION"

# Ensure BATADAL champion checkpoint is present
BATADAL_CKPT="results/surrogates/batadal_tcn/checkpoint_best.pt"
if [ ! -f "$BATADAL_CKPT" ]; then
    if [ -f "results/surrogates/vista_tcn_48h_150ep/tcn.pt" ]; then
        mkdir -p results/surrogates/batadal_tcn
        cp "results/surrogates/vista_tcn_48h_150ep/tcn.pt" "$BATADAL_CKPT"
        echo "Copied BATADAL champion checkpoint to $BATADAL_CKPT"
    else
        echo "Training BATADAL CausalTCN surrogate (150 epochs, patience 10)..."
        python scripts/train_surrogate.py --dataset batadal --model tcn --epochs 150 --patience 10 --scheduler onecycle --device cuda
    fi
fi

# 5A: BATADAL 10k
BATADAL_10K_SUMMARY="results/surrogate_eval/batadal_tcn_10k/summary.json"
if [ -f "$BATADAL_10K_SUMMARY" ]; then
    echo "[SKIPPED] BATADAL 10,000 evaluation already completed."
else
    echo "Running 10,000-scenario evaluation on BATADAL..."
    python scripts/run_surrogate_eval.py --dataset batadal --model tcn --checkpoint "$BATADAL_CKPT" --eval-mode 10k --device cuda
fi

# 5B: SWaT 10k
SWAT_10K_SUMMARY="results/surrogate_eval/swat_tcn_10k/summary.json"
if [ -f "$SWAT_10K_SUMMARY" ]; then
    echo "[SKIPPED] SWaT 10,000 evaluation already completed."
else
    echo "Running 10,000-scenario evaluation on SWaT..."
    python scripts/run_surrogate_eval.py --dataset swat --model tcn --checkpoint "$SWAT_CKPT" --eval-mode 10k --device cuda
fi

# 5C: WADI 10k
WADI_10K_SUMMARY="results/surrogate_eval/wadi_tcn_10k/summary.json"
if [ -f "$WADI_10K_SUMMARY" ]; then
    echo "[SKIPPED] WADI 10,000 evaluation already completed."
else
    echo "Running 10,000-scenario evaluation on WADI..."
    python scripts/run_surrogate_eval.py --dataset wadi --model tcn --checkpoint "$WADI_CKPT" --eval-mode 10k --device cuda
fi

# ------------------------------------------------------------------------------
# STEP 6: GIT COMMIT AND PUSH RESULTS
# ------------------------------------------------------------------------------
print_banner "STEP 6/6: GIT COMMIT AND PUSH ARTIFACTS"

BRANCH_NAME="feature/narrow-additive-inverse-research"

if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "Staging surrogate checkpoints, histories, loss curves, and evaluation artifacts..."
    git add -f results/surrogates/*_tcn/checkpoint_best.pt \
               results/surrogates/*_tcn/training_history.json \
               results/surrogates/*_tcn/loss_curve.png \
               results/surrogate_eval/ || true

    if git status --porcelain | grep -q 'results/'; then
        echo "Committing artifacts..."
        git commit -m "feat(runpod): offload training checkpoints and evaluation benchmarks for SWaT, WADI, BATADAL" || true

        echo "Pushing to remote branch ${BRANCH_NAME}..."
        git push origin "${BRANCH_NAME}" || {
            echo "WARNING: Git push failed. Please verify SSH credentials or branch permissions."
        }
    else
        echo "No changes in results/ to commit."
    fi
else
    echo "Git repository not found or git command not available. Skipping git commit."
fi

print_banner "RUNPOD WORKFLOW COMPLETED SUCCESSFULLY!"
echo "Finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
