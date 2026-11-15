#!/bin/bash
# ==============================================================================
# Script: check_queue_rank.sh
# Purpose: Check priority score, pending rank, and cluster queue position on Vista.
# Usage:
#   bash scripts/hpc/check_queue_rank.sh [JOB_ID] [PARTITION]
#
# Examples:
#   bash scripts/hpc/check_queue_rank.sh 1144267
#   bash scripts/hpc/check_queue_rank.sh 1144268 gh
# ==============================================================================

set -euo pipefail

JOB_ID="${1:-}"
PARTITION="${2:-gh}"

# If no JOB_ID is provided, find user's first pending job
if [ -z "$JOB_ID" ]; then
    USER_JOB=$(squeue -u "$USER" -p "$PARTITION" -t PD -h -o "%i" | head -n 1 || true)
    if [ -n "$USER_JOB" ]; then
        JOB_ID="$USER_JOB"
        echo "[INFO] No Job ID supplied. Auto-selected your pending Job ID: $JOB_ID"
    else
        echo "[WARN] No pending jobs found for user $USER in partition $PARTITION."
        echo "Usage: bash scripts/hpc/check_queue_rank.sh <JOB_ID> [PARTITION]"
        exit 1
    fi
fi

echo "================================================================================"
echo "TACC VISTA QUEUE RANK & PARTITION STATUS: $PARTITION"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%SZ")"
echo "================================================================================"

# Partition totals
TOTAL_PD=$(squeue -p "$PARTITION" -t PD -h | wc -l)
TOTAL_R=$(squeue -p "$PARTITION" -t R -h | wc -l)
TOTAL_CG=$(squeue -p "$PARTITION" -t CG -h | wc -l)

echo "Partition '$PARTITION' State:"
echo "  - Running Jobs:    $TOTAL_R"
echo "  - Pending Jobs:    $TOTAL_PD"
echo "  - Completing Jobs: $TOTAL_CG (releasing nodes)"
echo "--------------------------------------------------------------------------------"

# Inspect target job
JOB_LINE=$(squeue -j "$JOB_ID" -h -o "%i %P %j %u %t %M %L %D %R" 2>/dev/null || true)

if [ -z "$JOB_LINE" ]; then
    echo "[STATUS] Job $JOB_ID is NOT in the active queue (Completed, Cancelled, or Invalid ID)."
    echo ""
    echo "Querying accounting history (sacct):"
    sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
    exit 0
fi

JOB_STATE=$(echo "$JOB_LINE" | awk '{print $5}')
JOB_REASON=$(echo "$JOB_LINE" | awk '{print $9}')

echo "Target Job: $JOB_ID"
echo "  - State:           $JOB_STATE"
echo "  - Reason:          $JOB_REASON"

if [ "$JOB_STATE" = "PD" ]; then
    # Calculate queue rank
    RANK=$(squeue -p "$PARTITION" -t PD -S p -h -o "%i" | grep -n "^${JOB_ID}$" | cut -d: -f1 || true)
    PRIORITY=$(scontrol show job "$JOB_ID" 2>/dev/null | grep -o 'Priority=[0-9]*' | cut -d= -f2 || echo "N/A")
    TIME_LIMIT=$(scontrol show job "$JOB_ID" 2>/dev/null | grep -o 'TimeLimit=[0-9:-]*' | cut -d= -f2 || echo "N/A")

    echo "  - Priority Score:  $PRIORITY"
    echo "  - Time Limit:      $TIME_LIMIT"
    if [ -n "$RANK" ]; then
        echo "  - Queue Position:  Rank $RANK of $TOTAL_PD pending jobs"
    else
        echo "  - Queue Position:  Not ranked in priority order"
    fi
elif [ "$JOB_STATE" = "R" ]; then
    TIME_USED=$(echo "$JOB_LINE" | awk '{print $6}')
    TIME_LEFT=$(echo "$JOB_LINE" | awk '{print $7}')
    NODES=$(echo "$JOB_LINE" | awk '{print $8}')
    echo "  - Status:          RUNNING"
    echo "  - Elapsed Time:    $TIME_USED (Remaining: $TIME_LEFT)"
    echo "  - Nodes Allocated: $NODES"
fi

echo "--------------------------------------------------------------------------------"
echo "Top 5 Running Jobs Finishing Soonest in '$PARTITION' (Backfill Openings):"
echo "JOBID      USER         TIME       TIME_LEFT  NODES"
squeue -p "$PARTITION" -t R -o "%10i %12u %10M %10L %6D" -S L | head -n 6
echo "================================================================================"
