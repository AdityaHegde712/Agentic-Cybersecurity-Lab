#!/bin/bash
# ==============================================================================
# Script: tail_job_logs.sh
# Purpose: Live tail stdout and stderr log outputs for a Slurm job on Vista.
# Usage:
#   bash scripts/hpc/tail_job_logs.sh [JOB_ID] [LINES] [MODE: out|err|both]
#
# Examples:
#   bash scripts/hpc/tail_job_logs.sh 1144267           # Tails stdout of job 1144267
#   bash scripts/hpc/tail_job_logs.sh 1144267 100 err   # Tails last 100 lines of stderr
#   bash scripts/hpc/tail_job_logs.sh 1144267 50 both   # Tails both stdout and stderr
#   bash scripts/hpc/tail_job_logs.sh                   # Auto-tails most recent log
# ==============================================================================

set -euo pipefail

JOB_ID="${1:-}"
LINES="${2:-50}"
MODE="${3:-out}"

# Auto-detect latest log file if JOB_ID is omitted
if [ -z "$JOB_ID" ]; then
    LATEST_LOG=$(ls -t logs/slurm_*.out 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_LOG" ]; then
        echo "[INFO] No Job ID supplied. Auto-tailing most recent log: $LATEST_LOG"
        tail -f -n "$LINES" "$LATEST_LOG"
        exit 0
    else
        echo "[ERROR] No log files found in logs/ directory."
        echo "Usage: bash scripts/hpc/tail_job_logs.sh <JOB_ID> [LINES] [out|err|both]"
        exit 1
    fi
fi

# Locate stdout and stderr files for the given JOB_ID
OUT_FILE=$(ls logs/*"${JOB_ID}"*.out 2>/dev/null | head -n 1 || true)
ERR_FILE=$(ls logs/*"${JOB_ID}"*.err 2>/dev/null | head -n 1 || true)

# Fallback: check scontrol for paths
if [ -z "$OUT_FILE" ] && command -v scontrol &>/dev/null; then
    RESOLVED_OUT=$(scontrol show job "$JOB_ID" 2>/dev/null | grep -o 'StdOut=[^ ]*' | cut -d= -f2 || true)
    if [ -n "$RESOLVED_OUT" ] && [ -f "$RESOLVED_OUT" ]; then
        OUT_FILE="$RESOLVED_OUT"
    fi
fi

if [ -z "$ERR_FILE" ] && command -v scontrol &>/dev/null; then
    RESOLVED_ERR=$(scontrol show job "$JOB_ID" 2>/dev/null | grep -o 'StdErr=[^ ]*' | cut -d= -f2 || true)
    if [ -n "$RESOLVED_ERR" ] && [ -f "$RESOLVED_ERR" ]; then
        ERR_FILE="$RESOLVED_ERR"
    fi
fi

echo "================================================================================"
echo "TAILING LOGS FOR SLURM JOB: $JOB_ID (Mode: $MODE, Lines: $LINES)"
echo "================================================================================"

if [ "$MODE" = "out" ]; then
    if [ -n "$OUT_FILE" ] && [ -f "$OUT_FILE" ]; then
        echo "Tailing stdout: $OUT_FILE"
        echo "--------------------------------------------------------------------------------"
        tail -f -n "$LINES" "$OUT_FILE"
    else
        echo "[PENDING] Stdout log for Job $JOB_ID has not been created yet."
        echo "The job is likely still PENDING in queue. Check queue rank with:"
        echo "  bash scripts/hpc/check_queue_rank.sh $JOB_ID"
    fi
elif [ "$MODE" = "err" ]; then
    if [ -n "$ERR_FILE" ] && [ -f "$ERR_FILE" ]; then
        echo "Tailing stderr: $ERR_FILE"
        echo "--------------------------------------------------------------------------------"
        tail -f -n "$LINES" "$ERR_FILE"
    else
        echo "[INFO] Stderr log for Job $JOB_ID is empty or not created yet."
    fi
elif [ "$MODE" = "both" ]; then
    FILES=()
    [ -n "$OUT_FILE" ] && [ -f "$OUT_FILE" ] && FILES+=("$OUT_FILE")
    [ -n "$ERR_FILE" ] && [ -f "$ERR_FILE" ] && FILES+=("$ERR_FILE")

    if [ ${#FILES[@]} -gt 0 ]; then
        echo "Tailing: ${FILES[*]}"
        echo "--------------------------------------------------------------------------------"
        tail -f -n "$LINES" "${FILES[@]}"
    else
        echo "[PENDING] No log files created yet for Job $JOB_ID."
    fi
fi
