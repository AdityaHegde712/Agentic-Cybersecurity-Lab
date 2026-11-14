#!/bin/bash
# ==============================================================================
# Script: check_job_status.sh
# Purpose: Display detailed Slurm state, elapsed time, allocation, and diagnostics.
# Usage:
#   bash scripts/hpc/check_job_status.sh [JOB_ID]
#
# Examples:
#   bash scripts/hpc/check_job_status.sh           # Lists all active jobs for user
#   bash scripts/hpc/check_job_status.sh 1144267    # Shows details for job 1144267
# ==============================================================================

set -euo pipefail

JOB_ID="${1:-}"

if [ -z "$JOB_ID" ]; then
    echo "================================================================================"
    echo "ACTIVE JOBS FOR USER: $USER"
    echo "================================================================================"
    squeue -u "$USER" -o "%.10i %.10P %.24j %.8u %.4t %.10M %.10L %.6D %R"
    echo "================================================================================"
    echo "Pass a specific Job ID for detailed diagnostics:"
    echo "  bash scripts/hpc/check_job_status.sh <JOB_ID>"
    exit 0
fi

echo "================================================================================"
echo "SLURM JOB STATUS DIAGNOSTIC: $JOB_ID"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%SZ")"
echo "================================================================================"

# 1. Try scontrol for active/recent job details
SCONTROL_OUTPUT=$(scontrol show job "$JOB_ID" 2>/dev/null || true)

if [ -n "$SCONTROL_OUTPUT" ]; then
    echo "$SCONTROL_OUTPUT" | grep -E "JobId|JobName|UserId|Account|QOS" | sed 's/^ */  /'
    echo "$SCONTROL_OUTPUT" | grep -E "JobState|Reason|Dependency|Restarts" | sed 's/^ */  /'
    echo "$SCONTROL_OUTPUT" | grep -E "RunTime|TimeLimit|TimeMin|SubmitTime|StartTime|EndTime" | sed 's/^ */  /'
    echo "$SCONTROL_OUTPUT" | grep -E "NodeList|NumNodes|NumCPUs|Gres|Partition" | sed 's/^ */  /'
    echo "$SCONTROL_OUTPUT" | grep -E "WorkDir|StdOut|StdErr" | sed 's/^ */  /'
else
    echo "[INFO] Job $JOB_ID is not currently active in scontrol memory."
fi

echo "--------------------------------------------------------------------------------"
echo "Accounting Record (sacct):"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,NodeList
echo "================================================================================"
