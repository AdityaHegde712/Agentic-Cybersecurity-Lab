# Sprint 2 Decisions: Proposed Until Evidence Gate

## S2-D01: BATADAL Is the Neural De-Risking Dataset

**Status:** accepted.

**Context:** In the five-epoch GPU smoke, BATADAL training and validation loss both decreased. It achieved 2.88% FPR, 51.6% point TPR, 100% event recall, and one-sample median delay. HAI, SWaT, and WADI had much larger validation losses or weak detection.

**Decision:** Use BATADAL first for controlled LSTM configuration ablations before changing architecture or training the other datasets longer.

**Failure condition:** If no bounded BATADAL configuration reduces validation loss, stop and diagnose split/preprocessing assumptions before architectural expansion.

**Evidence:** The 10,000-row `gh-dev` grid selected `context-32_hidden-128_epochs-10` with 0.132595 validation loss, 2.95% FPR, 54.3% point TPR, 100% event recall, and one-sample median delay. See `results/lstm_forecaster/batadal_ablation/analysis/INSPECTION.md`.

## S2-D02: Configuration Before Architecture Forks

**Status:** proposed.

**Proposed decision:** Add per-dataset experiment configuration for normal-regime selection, sensor inclusion, context, hidden size, and training budget. Do not create separate neural architectures in Sprint 2.

**Rationale:** The current evidence indicates distribution mismatch for HAI/SWaT/WADI; an architecture fork would confound that diagnosis.

**Evidence update:** The BATADAL configuration grid established a stable shared-architecture forecaster, so no architecture fork is justified before injected recovery evidence.

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

## S2-D07: Standardized Known-Delta Recovery Pilot

**Status:** accepted.

**Decision:** Inject step, ramp, periodic, and multi-sensor observation attacks into the 1,440-row normal BATADAL validation sequence in train-derived sensor standard-deviation units. Convert the resulting observation and known delta back to original sensor units for forecast-residual recovery metrics.

**Rationale:** Standardized magnitudes make scenario severity comparable across sensors while preserving an exact original-unit `delta_x` for magnitude, sign, support, onset, and duration scoring.

**Boundary:** Thresholds are calibrated on a disjoint 256-row clean validation prefix. The same validation split selected the checkpoint, so this run is a recovery-mechanics pilot rather than a held-out generalization claim. Real BATADAL labels are not used as delta ground truth.

**Outcome:** The pilot rejected the current thresholded residual as a delta estimator: it had support F1 of 0.061-0.092, sign accuracy of 0.021-0.042, and onset errors of 317-1,117 samples. The next change must diagnose calibration and temporal support, not add neural architecture complexity.

## S2-D08: BATADAL Is a Controlled Testbed, Not a Dataset Scope Reduction

**Status:** accepted.

**Decision:** Use BATADAL alone to isolate the immediate recovery-mechanics failure because it is the only current dataset with a shared-architecture forecaster that learned bounded normal dynamics. Retain HAI, SWaT, and WADI as an explicit subsequent shared-architecture conditioning audit before claiming cross-dataset robustness or creating per-dataset architectures.

**Rationale:** The failed recovery pilot is an estimator/calibration question independent of cross-dataset transfer. Training all datasets while the recovery score activates before synthetic attacks would multiply compute without distinguishing threshold drift from forecaster adaptation. A successful BATADAL correction becomes a fixed protocol to test fairly across all four datasets.

**Required next evidence for HAI/SWaT/WADI:** normal-regime/split audit, train-derived normalization check, context/epoch/batch configuration sweep, compact loss and clean-residual diagnostics, and a documented architecture-fork decision. Poor shared-model loss alone is not evidence that each dataset needs a distinct architecture.
