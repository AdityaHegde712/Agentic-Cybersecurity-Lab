---
title: TACC Vista HPC Usage & Autonomous Agent Execution Guide
version: 1.0.0
last_updated: 2026-08-30
system: TACC Vista Supercomputer
architecture:
  cpu: ARMv9.0-A Neoverse-V2 (NVIDIA Grace Superchip)
  gpu: NVIDIA H200 Tensor Core GPU (96 GB HBM3, Hopper Architecture)
  interconnect: Mellanox Quantum-2 InfiniBand NDR (400 Gb/s GH, 200 Gb/s GG)
storage_tiers:
  home: { path: "$HOME", type: "VAST Flash NFS", quota: "23 GB / 500k files", backup: true, purge_policy: "none" }
  work: { path: "$WORK", type: "Global Lustre", quota: "1 TB / 3M files", backup: false, purge_policy: "none" }
  scratch: { path: "$SCRATCH", type: "VAST Flash NFS (~10 PB)", quota: "unlimited", backup: false, purge_policy: "10-day access purge" }
  tmp: { path: "/tmp", type: "Node-local NVMe (286 GB)", quota: "ephemeral", backup: false, purge_policy: "purged on job exit" }
queues:
  gh: { type: "Grace-Hopper (GH)", nodes_max: 64, gpus_max: 64, walltime_max: "48:00:00", su_rate: 1.00 }
  gh-dev: { type: "Grace-Hopper Dev", nodes_max: 8, gpus_max: 8, walltime_max: "02:00:00", su_rate: 1.00 }
  gg: { type: "Grace-Grace (GG)", nodes_max: 32, cores_max: 4608, walltime_max: "48:00:00", su_rate: 0.33 }
---

# TACC Vista HPC Usage & Autonomous Agent Execution Guide

This document is the definitive operational reference for running data processing, statistical detection pipelines, and deep learning training on the **TACC Vista** supercomputer. It is structured for both human engineers and autonomous AI agents.

---

## 1. System Architecture & Node Types

TACC Vista is an ARM-centric, AI-focused supercomputer funded by the NSF (Award #1818253). It serves as a bridge from Frontera to Horizon (LCCF).

```
                              ┌──────────────────────────────────────┐
                              │          TACC Vista Cluster          │
                              └──────────────────┬───────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    ┌─────────────────────────────┐                             ┌─────────────────────────────┐
    │   Grace-Grace (GG) Nodes    │                             │  Grace-Hopper (GH) Nodes    │
    │      (256 compute nodes)    │                             │     (600 compute nodes)     │
    ├─────────────────────────────┤                             ├─────────────────────────────┤
    │ • 2x Grace CPUs (144 cores) │                             │ • 1x Grace CPU (72 cores)   │
    │ • 237 GB LPDDR5X (850 GB/s) │                             │ • 1x NVIDIA H200 (96 GB HBM)│
    │ • Pure CPU workloads        │                             │ • NVLink-C2C Unified Memory │
    │ • 200 Gb/s NDR InfiniBand   │                             │ • 400 Gb/s NDR InfiniBand   │
    │ • Charge: 0.33 SU / node-hr │                             │ • Charge: 1.00 SU / node-hr │
    └─────────────────────────────┘                             └─────────────────────────────┘
```

### 1.1 Grace-Grace (GG) Compute Nodes (CPU Workload Tier)
- **CPU**: Dual-socket NVIDIA Grace Superchip (144 ARM Neoverse-V2 cores @ 3.4 GHz, 1 thread/core).
- **Vector Acceleration**: Scalable Vector Extension v2 (SVE2) + NEON (4x 128-bit functional units, 8x FP64 FMA/cycle/core).
- **Memory**: 237 GB LPDDR5X RAM with >850 GB/s bandwidth across 2 NUMA nodes.
- **Node Storage**: 286 GB NVMe `/tmp` partition.
- **Use Cases**: Fast tabular data preprocessing, Parquet validation, parallel CUSUM/BOCPD statistical runs, multi-core CPU baselines.

### 1.2 Grace-Hopper (GH) Compute Nodes (GPU Acceleration Tier)
- **Superchip**: NVIDIA GH200 Grace Hopper Superchip.
- **GPU**: NVIDIA H200 GPU with 96 GB HBM3 memory (34 TFlops FP64, 1979 TFlops FP16/Tensor Core).
- **CPU**: Single-socket Grace CPU with 72 cores @ 3.1 GHz, 116 GB LPDDR5X RAM.
- **Interconnect**: NVLink-C2C connects CPU and GPU directly at 900 GB/s (7x standard PCIe Gen 5 bandwidth), presenting CPU and GPU as unified NUMA domains.
- **Node Storage**: 286 GB NVMe `/tmp` partition.
- **Use Cases**: Deep learning model training (LSTM Autoencoder, Anomaly Transformer, GNN), high-throughput inference, embedding generation.

### 1.3 Login Nodes
- Hardware: Grace-Grace nodes (144 cores, 237 GB RAM).
- **CRITICAL RESTRICTION**: Login nodes are strictly for editing, git operations, file transfers, and job submission. Running heavy computations, dataset conversions, or model training on login nodes violates TACC policy and will lead to account deactivation.

---

## 2. File Systems, Storage Tiers & Purge Rules

```
┌───────────┬──────────────┬──────────────────┬────────────┬───────────────────────────────┐
│ System    │ Type         │ Quota            │ Backup     │ Primary Purpose & Policy      │
├───────────┼──────────────┼──────────────────┼────────────┼───────────────────────────────┤
│ $HOME     │ VAST Flash   │ 23 GB / 500k     │ Daily      │ Dotfiles, small scripts, code │
│ $WORK     │ Global Lustre│ 1 TB / 3M        │ None       │ Persistent datasets, models   │
│ $SCRATCH  │ VAST Flash   │ No Quota (~10 PB)│ None       │ High-speed I/O (10-day purge) │
│ /tmp      │ Local NVMe   │ 286 GB / node    │ None       │ Ephemeral run scratch, MPS    │
└───────────┴──────────────┴──────────────────┴────────────┴───────────────────────────────┘
```

### 2.1 The $SCRATCH 10-Day Purge Policy
> [!WARNING]
> `$SCRATCH` is **temporary execution storage**. Files not accessed within **10 days** are automatically deleted.
> - Compute node reads/writes update access time (`ls -ul`).
> - Login node reads/commands (e.g., `tar`, `scp`, `ls`) **DO NOT** update access time.
> - **Prohibited**: Artificially updating access timestamps (e.g., via automated `touch` loops) violates acceptable use.
> - **Safe Pattern**: Keep raw archive datasets and final model weights in `$WORK`. Stage active Parquet batches to `$SCRATCH` during training runs.

### 2.2 File System I/O Discipline
1. **No Lustre Striping on `$HOME` and `$SCRATCH`**: These are VAST Flash NFS filesystems. Commands like `lfs setstripe` only apply to `$WORK`.
2. **Avoid Small-File Floods**: High file counts (>100,000 loose files) degrade filesystem metadata servers. Package raw datasets into Parquet stores or HDF5 containers.
3. **Use Compute Node `/tmp`**: For per-worker temporary scratch or logging, use `/tmp` (cleared automatically when the Slurm job terminates).

---

## 3. Environment & Module Management (Lmod)

Vista uses TACC's **Lmod** module system to configure compilers, CUDA runtimes, and Python libraries.

```bash
# Standard compiler & CUDA stack
module load gcc/15.1.0 cuda/12.9
module load python3/3.11.8

# View loaded modules
module list

# Search for available packages
module spider nvpl
module spider torch
```

### 3.1 Python Virtual Environment Protocol on Compute Nodes
Always create and activate Python virtual environments on `$SCRATCH` or `$WORK` within an interactive `idev` session or batch script:

```bash
# 1. Request an interactive dev node
idev -p gh-dev -N 1 -n 1 -m 60

# 2. Load necessary modules
module load gcc cuda python3/3.11.8

# 3. Create virtual environment
python3 -m venv $SCRATCH/venvs/sengupta_cyber

# 4. Activate environment
source $SCRATCH/venvs/sengupta_cyber/bin/activate

# 5. Install PyTorch with CUDA support for aarch64
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install pyarrow pandas numpy scipy pytest
```

### 3.2 NVIDIA Performance Libraries (NVPL)
For optimized mathematical operations (BLAS, LAPACK, FFTW) on the Grace ARM Neoverse-V2 architecture:
```bash
module load nvpl
# Link with: -I${TACC_NVPL_DIR}/include -L${TACC_NVPL_DIR}/lib -lnvpl_blas_lp64_gomp
```

---

## 4. Production Queues & Slurm Job Management

### 4.1 Production Queue Limits

| Queue (`-p`) | Node Type | Max Nodes/Job | Max Duration | Max Nodes/User | Charge Rate |
|---|---|---|---|---|---|
| `gh` | Grace-Hopper (H200 GPU) | 64 nodes (64 GPUs) | 48:00:00 | 192 nodes | 1.00 SU / node-hr |
| `gh-dev` | Grace-Hopper Dev | 8 nodes (8 GPUs) | 02:00:00 | 8 nodes | 1.00 SU / node-hr |
| `gg` | Grace-Grace (144-core CPU) | 32 nodes (4608 cores) | 48:00:00 | 128 nodes | 0.33 SU / node-hr |

Check live queue limits at any time with `qlimits`.

### 4.2 TACC Accounting & Charging Rule
$$\text{SUs Billed} = (\text{Allocated Nodes}) \times \max(\text{Wall Clock Hours}, 0.25) \times (\text{Queue Rate})$$
- **15-Minute Minimum Charge**: Every job is charged for at least 15 minutes (0.25 hr).
- **Dedicated Allocation**: Vista does not use node-sharing. You receive the entire node (all 144 cores on GG, or 72 cores + 1 H200 on GH).
- **Release on Exit**: When a job terminates cleanly, unused wall clock time is not billed beyond the 15-minute floor.

---

## 5. Slurm Job Script Templates

### 5.1 Grace-Hopper (GH) Single-Node Deep Learning Training (`train_gh.slurm`)

```bash
#!/bin/bash
#SBATCH -J cyber_lstm_ae               # Job name
#SBATCH -o %j.out                      # Standard output file (%j expands to jobID)
#SBATCH -e %j.err                      # Standard error file
#SBATCH -p gh                          # Grace-Hopper GPU queue
#SBATCH -N 1                           # 1 compute node
#SBATCH -n 1                           # 1 task
#SBATCH -t 04:00:00                    # Wall clock time limit (4 hours)
#SBATCH -A YOUR_ALLOCATION_ID          # Project allocation

# Environment setup
module purge
module load gcc cuda python3/3.11.8
source $SCRATCH/venvs/sengupta_cyber/bin/activate

# Execute model training
python -u src/models/train_lstm_ae.py \
    --data-dir $WORK/data/processed/swat \
    --epochs 50 \
    --batch-size 256 \
    --device cuda
```

### 5.2 Grace-Hopper (GH) Multi-Node Distributed Training (`multinode_gh.slurm`)

```bash
#!/bin/bash
#SBATCH -J cyber_dist_trans            # Job name
#SBATCH -o %j.out                      # Output file
#SBATCH -e %j.err                      # Error file
#SBATCH -p gh                          # Grace-Hopper GPU queue
#SBATCH -N 4                           # 4 compute nodes (4 H200 GPUs)
#SBATCH -n 4                           # 4 tasks total (1 per node)
#SBATCH -t 08:00:00                    # 8 hours wall clock
#SBATCH -A YOUR_ALLOCATION_ID

module purge
module load gcc cuda python3/3.11.8
source $SCRATCH/venvs/sengupta_cyber/bin/activate

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Launch PyTorch DDP via torchrun across nodes using ibrun
ibrun -np $SLURM_NNODES torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/models/train_anomaly_transformer.py \
    --data-dir $WORK/data/processed/wadi
```

### 5.3 Grace-Grace (GG) Multi-Core Statistical Detection Benchmark (`eval_gg.slurm`)

```bash
#!/bin/bash
#SBATCH -J cusum_benchmark             # Job name
#SBATCH -o %j.out                      # Output file
#SBATCH -e %j.err                      # Error file
#SBATCH -p gg                          # Grace-Grace CPU queue (0.33 SU/hr)
#SBATCH -N 1                           # 1 compute node (144 ARM cores)
#SBATCH -n 1                           # 1 task
#SBATCH -t 02:00:00                    # 2 hours
#SBATCH -A YOUR_ALLOCATION_ID

module purge
module load gcc python3/3.11.8 nvpl
source $SCRATCH/venvs/sengupta_cyber/bin/activate

# Utilize OpenMP threading across all 144 Grace cores
export OMP_NUM_THREADS=144

python -u src/evaluation/cusum_experiment.py \
    --dataset all \
    --threads 144 \
    --output-dir $SCRATCH/results/cusum_eval
```

### 5.4 NVIDIA Multi-Process Service (MPS) for Concurrent GPU Tasks

```bash
#!/bin/bash
#SBATCH -J mps_parallel                # Job name
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 4                           # 4 concurrent processes on 1 GPU
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION_ID

# Configure MPS directories in node-local /tmp
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Start MPS control daemon
nvidia-cuda-mps-control -d
sleep 2

# Launch concurrent evaluations
python src/evaluation/eval_worker.py --split 0 &
python src/evaluation/eval_worker.py --split 1 &
python src/evaluation/eval_worker.py --split 2 &
python src/evaluation/eval_worker.py --split 3 &

wait

# Stop MPS daemon
echo quit | nvidia-cuda-mps-control
```

---

## 6. Interactive Debugging & Development (`idev`)

For testing code before submitting long batch jobs, use `idev`:

```bash
# Request 1 Grace-Hopper GPU node for 60 minutes
idev -p gh-dev -N 1 -n 1 -m 60 -A YOUR_ALLOCATION_ID

# Request 1 Grace-Grace CPU node for 120 minutes
idev -p gg -N 1 -n 144 -m 120 -A YOUR_ALLOCATION_ID
```

### Inspecting Running Node Resources
Once connected to a compute node:
```bash
# Check GPU memory & utilization on GH node
nvidia-smi

# Check CPU core activity across 144 cores on GG node
htop
```

---

## 7. Strict Autonomous Agent Operational Directives

When AI coding agents (or automated subagents) generate, run, or debug scripts on TACC Vista, they **MUST** strictly follow these rules:

1. **NO COMPUTATION ON LOGIN NODES**:
   - Compiling trivial single scripts is permitted on login nodes.
   - Any execution taking $>60\text{ seconds}$ or consuming $>2\text{ GB RAM}$ MUST be submitted to `gh-dev` or `gg` via `sbatch` or `idev`.
2. **NO HARDCODED HOSTNAMES OR NODE EXCLUSIONS**:
   - Slurm directives requesting specific physical nodes (e.g., `#SBATCH -w c642-011`) are strictly prohibited and deleted by TACC admins.
3. **DO NOT USE `--export` OR `--gres`**:
   - Do NOT use `--export` in Slurm scripts (interferes with Lmod environment).
   - Do NOT use `--gres=gpu:1` (Vista Slurm handles GPU allocation via partition `-p gh`).
4. **UNBUFFERED OUTPUT LOGGING**:
   - Always run Python commands with `-u` (e.g., `python -u script.py`) or set `export PYTHONUNBUFFERED=1` to ensure logs flush immediately to disk for inspection.
5. **PERSISTENT STORAGE MAPPING**:
   - Raw datasets and models $\to$ `$WORK/Agentic-Cybersecurity-Lab/`
   - Active experiment outputs & temporary tensors $\to$ `$SCRATCH/` or `/tmp/`
   - Code repository $\to$ `$HOME/` or `$WORK/`
6. **15-MINUTE MINIMUM SU BATCHING**:
   - Never launch dozens of individual 30-second scripts. Aggregate parameter sweeps or evaluations into single multi-threaded or multi-process jobs to avoid wasting allocation credits.
