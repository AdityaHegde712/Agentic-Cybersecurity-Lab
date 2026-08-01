# Codebase Overview

> Research codebase for agentic cyber-defense against sensor data attacks. Implements B0 data pipeline (synthetic generation + HAI loader) and B1 CUSUM baseline detector. Foundation for a multi-agent detection/response system targeting cyber-physical sensor manipulation.

**Last updated:** 2026-07-31
**Primary language:** Python 3.11
**Architecture style:** Modular library (no network components, no database)

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Runtime | Python 3.11 | Uses `typing` throughout; no async |
| Numeric | NumPy 2.4+ | Core array operations; `np.random.default_rng` for reproducibility |
| Data | pandas 3.0+ | CSV loading with semicolon delimiter; used in HAI loader |
| Stats | SciPy 1.17+ | Available but not yet imported in B0/B1 modules |
| Testing | pytest 9.1+ | Contract tests locked by TDD policy; `pythonpath = src` in pytest.ini |
| Plotting | matplotlib | Used only in `results/` output scripts; not imported by core modules |

---

## Entry points

| Entry | Command | Purpose |
|---|---|---|
| Run all tests | `pytest tests/spec -v` | Executes locked contract tests (27/27 must pass) |
| Smoke test synthetic | `python -c "from src.data.synthetic import SensorDataGenerator; ..."` | Validate generator instantiation |
| Smoke test HAI | `python -c "from src.data.hai_loader import HAILoader; ..."` | Validate loader returns train/val/test splits |
| Run B1 evaluation | `python src/evaluation/cusum_experiment.py` | Full CUSUM eval: synthetic grid + HAI; writes CSVs + report to `results/` |

---

## Key modules

| Path | Responsibility |
|---|---|
| `src/data/synthetic.py` | `SensorDataGenerator` — deterministic sensor data with controllable attacks (step/ramp/periodic/coordinated) and correlated noise. Paired runs with different seeds isolate injected delta_x. |
| `src/data/hai_loader.py` | `HAILoader` — loads HAI-20.07 CSVs, splits train→train+val (temporal, last 20%), z-score normalizes on train only. Returns `Split` dataclass with X (T,59), y (T,), timestamps (T,). |
| `src/detection/base.py` | `BaseDetector` ABC — two abstract methods: `update(y)` and `reset()`. All detectors implement this interface. |
| `src/detection/cusum.py` | `CUSUMDetector(mu0, k, h, reset_after_detection=True)` — one-sided upper CUSUM. `S_t = max(0, S_{t-1} + y - mu0 - k)`, alarm when `S_t > h`. Auto-resets after detection if enabled. |
| `src/evaluation/metrics.py` | `detection_delay`, `false_positive_rate`, `true_positive_rate`, `estimation_mse`, `estimation_bias` — standalone metric functions. |
| `src/evaluation/cusum_experiment.py` | B1 evaluation pipeline: runs synthetic grid (4 magnitudes × 3 noise × 3 seeds × 5 sensors) on paired deviations; runs HAI per-sensor with system-level OR aggregation. Writes CSVs and markdown report. |
| `tests/spec/` | Locked contract tests (TDD immutable). **Do not modify.** |

---

## Non-obvious patterns

**Paired-deviation isolation convention**
B0 tests and B1 evaluation use `dev = data(attack, seed_a) - data(none, seed_b)` to isolate the injected delta_x from the unknown baseline signal. This is the only way to verify detection without knowing the true baseline. Both seeds must differ; both runs share identical generator parameters.

**HAI z-score normalization is train-only, with zero-variance handling**
`HAILoader` fits mean/std on `train_split.X` only and applies to all splits. Constant sensor columns (digital signals in raw HAI) are replaced with deterministic unit-variance jitter (seed=0) to satisfy the unit-variance contract without leaking information. This jitter is visible in `train.X.std()` being exactly 1.0 for those columns.

**Contract tests are immutable by design**
`tests/spec/test_*.py` files contain "LOCKED — do not modify" headers. These are TDD contracts written before implementation. Implementation agents may not modify them; test-writing agents may not refactor implementation. Violation breaks the B0/B1 acceptance pipeline.

**CUSUM parameters in B1 evaluation are NOT noise-calibrated**
The locked B1 experiment uses fixed absolute thresholds `k=0.5, h=5.0` for synthetic data and `k=0.5*sigma, h=5.0*sigma` for HAI. The synthetic evaluation fails at noise ≥ 1.0 because the fixed thresholds are below the deviation noise scale. See `results/cusum_report.md` for root cause.

**HAI system alarm is OR-aggregation across 59 sensors**
`cusum_experiment.py` fires a system alarm if ANY sensor triggers. With per-sensor `k=0.5*sigma, h=5.0*sigma`, the system false-positive rate is 90.4% (union of 59 independent false-alarm processes). The detector carries real attack information (TPR 0.958) but is drowned in alarm floor.

---

## Data layer

No database. All data is file-based:

- `data/hai/raw/hai-20.07/` — HAI-20.07 dataset: `train1.csv`, `train2.csv`, `test1.csv`, `test2.csv`. Semicolon-delimited, 64 columns (time + 59 sensors + attack + 3 attack_P columns). Parsed to epoch seconds for timestamps.
- `results/` — evaluation outputs: CSVs (`cusum_evaluation_synthetic.csv`, `cusum_evaluation_hai.csv`), plots (`synthetic_delta_x_plot.png`), and markdown reports (`cusum_report.md`, `hai_loader_report.md`).

Row counts (fixed, no `drop_duplicates`): train1=309,600; train2=241,200; test1=291,600; test2=153,000. Train+val=550,800; test=444,600.

---

## Development workflow

```bash
# 1. Activate environment
.venv\Scripts\activate   # Windows

# 2. Run locked contract tests (must all pass)
pytest tests/spec -v

# 3. Run B1 evaluation (generates results/)
python src/evaluation/cusum_experiment.py

# 4. Check results
cat results/cusum_report.md
```

---

## Before you change code

- **`tests/spec/` files are immutable.** Never edit. If a contract test fails, the implementation is wrong — fix the implementation.
- **`SensorDataGenerator` baseline is deterministic sine.** The sine frequencies are integer-period over T so half-window means are ~0. Changing the baseline breaks all paired-deviation tests.
- **HAI loader has no `drop_duplicates`.** Row-count integrity is a locked contract (len(train)+len(val)==550,800). Adding duplicate removal will fail `test_hai_loader.py::test_temporal_split_no_leakage`.
- **CUSUM `update()` mutates state.** The detector is not pure; calling it changes `S_t`. Re-running on the same data produces different results unless `reset()` is called first.
- **`src/evaluation/cusum_experiment.py` uses `sys.path.insert`.** The script manipulates `sys.path` to import from `src/`. Running via `python -m src.evaluation.cusum_experiment` from the root also works due to `pytest.ini`'s `pythonpath = src`.

---

## Current status

- **27/27 contract tests PASS** (test_synthetic: 10, test_hai_loader: 7, test_cusum: 7, test_imports: 3)
- **B1 acceptance FAILED** — documented in `results/cusum_report.md`
  - Synthetic: CUSUM thresholds uncalibrated for noise ≥ 1.0 (fixed `k=0.5, h=5.0` below deviation noise scale)
  - HAI: system FPR 0.904 due to OR-aggregation over 59 sensors (ARL collapses to ~10 samples)
  - Next iteration: relative CUSUM (`k = 0.5 * sigma_dev`), consensus ≥ 2 sensors, delay metric clips pre-attack alarms

---

## Key file paths (staleness risk)

| Path | Risk |
|---|---|
| `src/data/synthetic.py` | Stable — locked generator interface |
| `src/data/hai_loader.py` | Stable — locked loader interface |
| `src/detection/cusum.py` | Stable — locked CUSUM interface |
| `tests/spec/test_*.py` | Immutable by policy |
| `results/cusum_report.md` | Regenerated by `cusum_experiment.py` |
| `data/hai/raw/hai-20.07/` | External dataset; paths hardcoded in `hai_loader.py` |
