# Codebase Overview

> Research codebase for agentic cyber-defense against sensor data attacks. Implements multi-dataset data pipeline (SWaT, WADI, BATADAL, HAI parquet stores + synthetic generator) and baseline detection architectures. Foundation for multi-agent detection and mitigation targeting cyber-physical sensor manipulation.

**Last updated:** 2026-10-10
**Primary language:** Python 3.11
**Architecture style:** Modular library + HPC compute workflows (TACC Vista Grace-Grace / Grace-Hopper)

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Runtime | Python 3.11 | Uses `typing` throughout; no async |
| Numeric | NumPy 2.4+ | Core array operations; `np.random.default_rng` for reproducibility |
| Data | pandas 3.0+, pyarrow 25.0+ | Fast Parquet data store loaders with schema & temporal validation |
| ML / DL | PyTorch 2.13.0+cu132 | Single-node / Multi-node DDP acceleration via `torchrun` and `ibrun` |
| Testing | pytest 9.1+ | `tests/spec/` locked contracts plus processed-data contracts; current verified gates are 96 spec tests + 40 processed-data tests |
| HPC | Slurm, Lmod, NVPL | TACC Vista (`gg` CPU and `gh` H200 GPU partitions) |

---

## Entry points

| Entry | Command | Purpose |
|---|---|---|
| Run all tests | `pytest tests/ -v` | Executes all 67 contract and dataset validation tests |
| Run locked spec tests | `pytest tests/spec -v` | Executes locked spec contract tests (27/27 must pass) |
| Validate processed store | `python scripts/validate_processed.py` | Validates 44 schema and integrity checks across SWaT/WADI/BATADAL/HAI |
| Smoke test synthetic | `python -c "from src.data.synthetic import SensorDataGenerator; ..."` | Validate generator instantiation |
| Smoke test multi-dataset | `python -c "from src.data.registry import load_dataset; ..."` | Validate registry loader returns train/val/test splits |
| Run B1 evaluation | `python src/evaluation/cusum_experiment.py` | Full CUSUM eval: synthetic grid + HAI; writes CSVs + report to `results/` |

---

## Key modules

| Path | Responsibility |
|---|---|
| `src/config.py` | Central configuration module resolving dataset paths, HPC parameters, and environment overrides. |
| `configs/default.yaml` | Base YAML configuration for local and TACC Vista execution defaults. |
| `src/data/synthetic.py` | `SensorDataGenerator` — deterministic sensor data with controllable attacks (step/ramp/periodic/coordinated) and correlated noise. |
| `src/data/hai_loader.py` | `HAILoader` — raw HAI-20.07 CSV loader with temporal splits and z-score normalization. |
| `src/data/registry.py` | Dataset registry dispatcher (`load_dataset`, `get_meta`) across SWaT, WADI, BATADAL, HAI. |
| `src/data/{swat,wadi,batadal,hai_store}_loader.py` | Standardized dataset loaders reading deterministic Parquet stores with metadata validation. |
| `src/detection/base.py` | `BaseDetector` ABC — two abstract methods: `update(y)` and `reset()`. |
| `src/detection/cusum.py` | `CUSUMDetector(mu0, k, h, reset_after_detection=True)` — one-sided upper CUSUM. |
| `src/evaluation/metrics.py` | `detection_delay`, `false_positive_rate`, `true_positive_rate`, `estimation_mse`, `estimation_bias`. |
| `src/attacks/additive.py` | Declared additive attack generators; returns observed values, exact delta, event metadata, and support mask. |
| `src/evaluation/attack_metrics.py` | Support-aware additive recovery metrics; explicit support masks prevent float-rounding artifacts from redefining ground truth. |
| `src/evaluation/injected_recovery.py` | Standardized BATADAL injection, original-unit delta reconstruction, declared-support projection, and temporal scoring. |
| `src/experiments/batadal_constraint_audit.py` | Aggregate-only target-range and companion-correlation audit for the narrow inverse benchmark. |
| `src/experiments/injection_ground_truth_audit.py` | Compares corrected analytical injection arithmetic with the legacy float32 round-trip path. |
| `scripts/audit_batadal_constraints.py` | Writes compact BATADAL constraint evidence under `results/narrow_inverse_research/data_audit/`. |
| `scripts/audit_injection_ground_truth.py` | Verifies configured injection scenarios without serializing raw data. |
| `docs/operations/hpc-usage-guide.md` | Comprehensive guide for running workloads on TACC Vista compute nodes (SLURM, queues, PyTorch, MPS, I/O). |
| `tests/spec/` | Locked contract tests (TDD immutable). **Do not modify.** |
| `tests/test_processed_data.py` | Dataset store contract tests verifying schema, timestamps, monotonicity, and labels. |

---

## Data layer

All data is file-based and staged for local and HPC storage:
- `data/processed/{swat,wadi,batadal,hai}/` — Deterministic Parquet stores (`train.parquet`, `val.parquet`, `test.parquet`, `meta.json`).
- `data/hai/raw/hai-20.07/` — Original raw HAI-20.07 CSV files.
- `docs/archive/` — Historical grant proposals, early survey notes, and exploratory literature.

---

## HPC & Development workflow

```bash
# 1. Activate environment
# Local:
.venv\Scripts\activate   # or conda activate sengupta_research

# Remote HPC (TACC Vista):
module load gcc cuda python3/3.11.8
source $SCRATCH/venvs/sengupta_cyber/bin/activate

# 2. Run test suite (67/67 must pass)
pytest tests/ -v

# 3. Validate processed stores (44 checks)
python scripts/validate_processed.py
```

---

## Current status

- **67/67 contract tests PASS** (27 locked spec tests + 40 processed data store tests).
- **44/44 processed store validation checks PASS**.
- **HPC migration guide complete**: [docs/operations/hpc-usage-guide.md](file:///c:/Users/hifia/Projects/Sengupta_Research/docs/operations/hpc-usage-guide.md).
- **Archiving complete**: Deprecated proposal drafts and early surveys moved to `docs/archive/`.
- **Documentation index**: [docs/README.md](docs/README.md) maps current architecture, research, operations, and archive authorities.
- **Current inverse benchmark**: BATADAL observation-only single-sensor bounded step attack; coupling and broader attack families remain deferred.
- **Ground-truth audit**: Corrected analytical original-unit delta arithmetic and explicit declared support are covered by locked spec tests and aggregate audit scripts.
