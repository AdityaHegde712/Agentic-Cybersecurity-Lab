---
name: tacc-vista-hpc
description: Operational skill and safety guardrails for interacting with the TACC Vista supercomputer (Grace-Grace and Grace-Hopper architecture). Use whenever preparing Slurm batch scripts, requesting interactive compute nodes (idev), querying queue status, managing remote filesystems ($HOME, $WORK, $SCRATCH, /tmp), loading Lmod environment modules, configuring Python/PyTorch virtual environments on ARM/Hopper nodes, or executing remote jobs. Enforces mandatory user-confirmation protocols before issuing any command that executes remotely or consumes allocation Service Units (SUs).
---

# TACC Vista HPC Operational Skill & Safety Guardrails

This skill governs how AI agents interact with the **TACC Vista Supercomputer** (Project Charge Code: `TG-NAIRR260371`).

---

## 1. Mandatory Safety & Approval Guardrail (ALWAYS-ASK PROTOCOL)

> [!CAUTION]
> **MANDATORY USER CONFIRMATION**:
> Agents must **NEVER** execute remote SSH commands, submit Slurm jobs (`sbatch`), request interactive nodes (`idev`), or cancel jobs (`scancel`) autonomously without explicit user confirmation.
>
> **Protocol**:
> 1. Show the user the exact command line or `#SBATCH` script to be run.
> 2. Show the estimated SU cost: $\text{SUs} = \text{Nodes} \times \max(\text{Hours}, 0.25) \times \text{Rate}$.
> 3. Specify the target queue (`gh`, `gh-dev`, or `gg`).
> 4. Ask the user for explicit confirmation before proceeding.

---

## 2. System Hardware & Partition Reference

| Partition (`-p`) | Node Architecture | Cores/Node | Memory/Node | GPUs/Node | Max Nodes | Max Walltime | SU Rate |
|---|---|---|---|---|---|---|---|
| `gh` | NVIDIA Grace-Hopper (GH200) | 72 ARM cores | 116 GB LPDDR5X | 1x H200 (96 GB HBM3) | 64 | 48:00:00 | 1.00 SU/hr |
| `gh-dev` | NVIDIA Grace-Hopper Dev | 72 ARM cores | 116 GB LPDDR5X | 1x H200 (96 GB HBM3) | 8 | 02:00:00 | 1.00 SU/hr |
| `gg` | NVIDIA Grace-Grace (Dual CPU) | 144 ARM cores | 237 GB LPDDR5X | None (CPU only) | 32 | 48:00:00 | 0.33 SU/hr |

- **Project Charge Code**: `TG-NAIRR260371` (use with `-A TG-NAIRR260371`)
- **Login Node Policy**: Login nodes are Grace-Grace nodes. **NO computation >60s or >2GB RAM on login nodes**. Always dispatch workloads to compute nodes.

---

## 3. Storage Routing Rules & Purge Policies

```
┌───────────┬──────────────┬──────────────────┬────────────┬───────────────────────────────────────┐
│ Path      │ Type         │ Quota            │ Backup     │ Agent Storage Rule                    │
├───────────┼──────────────┼──────────────────┼────────────┼───────────────────────────────────────┤
│ $HOME     │ VAST Flash   │ 23 GB / 500k     │ Daily      │ Code repository, dotfiles, configs    │
│ $WORK     │ Global Lustre│ 1 TB / 3M        │ None       │ Persistent raw datasets, model weights│
│ $SCRATCH  │ VAST Flash   │ Unlimited (~10PB)│ None       │ Active training I/O (10-DAY PURGE!)   │
│ /tmp      │ Local NVMe   │ 286 GB / node    │ None       │ Temporary node scratch, MPS sockets   │
└───────────┴──────────────┴──────────────────┴────────────┴───────────────────────────────────────┘
```

> [!WARNING]
> **$SCRATCH Purge Policy**: Files on `$SCRATCH` not accessed within **10 days** are automatically deleted.
> - Only compute node reads/writes update access time.
> - **Agent Rule**: Always stage persistent data in `$WORK`. Only use `$SCRATCH` for active execution runs.

---

## 4. Environment Configuration & Modules (Lmod)

Execute all environment configuration inside an `idev` session or Slurm batch script:

### 4.1 Standard Module Stack
```bash
# Purge stale modules and load GCC, CUDA, and Python 3.11
module purge
module load gcc/15.1.0 cuda/12.9 python3/3.11.8

# For optimized ARM mathematical libraries (BLAS/LAPACK/FFTW)
module load nvpl
```

### 4.2 Python Virtual Environment on Compute Nodes
```bash
# Create venv in $SCRATCH or $WORK
python3 -m venv $SCRATCH/venvs/sengupta_cyber
source $SCRATCH/venvs/sengupta_cyber/bin/activate

# Upgrade pip and install PyTorch with aarch64 CUDA 12.9 wheel
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install pyarrow pandas numpy scipy pytest
```

---

## 5. Workflows & Command Recipes

### 5.1 Interactive Development (`idev`)
Use `idev` for debugging, test runs, and interactive tuning:

```bash
# 1 Grace-Hopper GPU node for 60 minutes (Cost: 1.0 SU)
idev -p gh-dev -N 1 -n 1 -m 60 -A TG-NAIRR260371

# 1 Grace-Grace CPU node for 120 minutes (Cost: 0.66 SUs)
idev -p gg -N 1 -n 144 -m 120 -A TG-NAIRR260371
```

### 5.2 Batch Submission (`sbatch`)
Always generate scripts with unbuffered logging (`python -u`):

```bash
#!/bin/bash
#SBATCH -J train_model
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A TG-NAIRR260371

module purge
module load gcc cuda python3/3.11.8
source $SCRATCH/venvs/sengupta_cyber/bin/activate

python -u src/models/train.py --data-dir $WORK/data/processed/swat
```

### 5.3 Monitoring & Diagnostics
```bash
# Check queue limits
qlimits

# View user's queued/running jobs
squeue -u $USER

# Alternative TACC queue monitor
showq -u

# Cancel a job (ALWAYS ASK FIRST)
scancel <JOB_ID>

# Account diagnostics
module load checklist
checklist
```

---

## 6. Agent Action Checklist

Before proposing or executing an HPC command:
- [ ] Has the Project Allocation ID (`TG-NAIRR260371`) been included in the directive?
- [ ] Is the partition correctly chosen (`gh-dev` for testing, `gh` for long GPU runs, `gg` for multi-core CPU)?
- [ ] Has the estimated SU cost been calculated and presented to the user?
- [ ] Is data reading mapped to `$WORK` or `$SCRATCH` rather than `$HOME`?
- [ ] Does the Python script include unbuffered output (`-u` or `PYTHONUNBUFFERED=1`)?
- [ ] Has explicit user confirmation been requested before firing the execution?
