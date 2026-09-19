# Sprint 2 Decisions: Proposed Until Evidence Gate

## S2-D01: BATADAL Is the Neural De-Risking Dataset

**Status:** proposed.

**Context:** In the five-epoch GPU smoke, BATADAL training and validation loss both decreased. It achieved 2.88% FPR, 51.6% point TPR, 100% event recall, and one-sample median delay. HAI, SWaT, and WADI had much larger validation losses or weak detection.

**Proposed decision:** Use BATADAL first for controlled LSTM configuration ablations before changing architecture or training the other datasets longer.

**Failure condition:** If no bounded BATADAL configuration reduces validation loss, stop and diagnose split/preprocessing assumptions before architectural expansion.

## S2-D02: Configuration Before Architecture Forks

**Status:** proposed.

**Proposed decision:** Add per-dataset experiment configuration for normal-regime selection, sensor inclusion, context, hidden size, and training budget. Do not create separate neural architectures in Sprint 2.

**Rationale:** The current evidence indicates distribution mismatch for HAI/SWaT/WADI; an architecture fork would confound that diagnosis.

## S2-D03: Synthetic Recovery Is the Research Claim Gate

**Status:** accepted.

**Decision:** The project may claim additive attack modeling only after recovery is evaluated against injected known delta. Real dataset results remain detector evidence.

## S2-D04: Vista Environment Naming

**Status:** accepted.

**Decision:** Use `agentic_cyber` as the canonical Vista venv name. Load `gcc/15.1.0`, `cuda/12.9`, `python3/3.11.8`, and `nvpl` before activation; do not recreate the venv unless the module stack fails to restore `libpython3.11.so.1.0`.

## S2-D05: Versioned JSON Ablation Contract

**Status:** accepted.

**Decision:** Store the bounded BATADAL ablation grid in `configs/experiments/batadal_lstm_ablation.json` and load it through `src/experiments/lstm_ablation_config.py`. The contract fixes the dataset, seed, 10,000-row smoke limit, repository-relative results directory, and named context/hidden-size/epoch runs.

**Rationale:** A typed, standard-library JSON loader is portable to Vista, prevents machine-specific output paths, and gives the runner a deterministic manifest source.

**Evidence:** `tests/spec/test_lstm_ablation_config.py`, `tests/spec/test_lstm_ablation_runner.py`, and `tests/spec/test_lstm_forecaster.py` pass together. The runner writes its manifest and per-run summaries from the versioned contract.

## S2-D06: Aggregate-Only Ablation Selection

**Status:** accepted.

**Decision:** Select and inspect LSTM ablation runs from `summary.csv` and training histories only. `scripts/analyze_lstm_ablation.py` ranks best validation loss, point FPR, event recall, and median delay, then saves an `INSPECTION.md` report and `ranked_metrics.png`.

**Consequence:** Do not load or manually inspect high-volume `scores.csv` files when choosing a configuration. Raw scores remain source artifacts for later repository scripts, not conversational context.
