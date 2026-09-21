# Sprint 2 Tasks: Proposed and Atomic

- [x] 1. Add a repository-tracked BATADAL ablation configuration contract with fixed seed, dataset, context length, hidden size, epochs, and output directory.
  **Evidence:** `configs/experiments/batadal_lstm_ablation.json`, `src/experiments/lstm_ablation_config.py`, and `tests/spec/test_lstm_ablation_config.py`; focused configuration and LSTM contracts: 8 passed.

- [x] 2. Extend the LSTM runner to execute a named configuration grid and write one manifest plus separate result folders per run.
  **Evidence:** `src/experiments/lstm_ablation_runner.py`, `scripts/run_lstm_forecaster.py --ablation-config`, `results/lstm_forecaster/batadal_ablation/manifest.json`, and `summary.csv`.

- [x] 3. Add compact ablation analysis plots and a Markdown selection report.
  **Evidence:** `results/lstm_forecaster/batadal_ablation/analysis/INSPECTION.md` and `ranked_metrics.png`; the real artifact inspection matches the aggregate table and did not use score rows.

- [x] 4. Obtain owner approval and run the bounded BATADAL `gh-dev` smoke.
  **Evidence:** all four 10,000-row runs completed. The selected configuration is `context-32_hidden-128_epochs-10`; compact artifacts were copied locally and inspected.

- [x] 5. Add an injected additive-attack forecaster evaluation runner and locked recovery assertions.
  **Evidence:** `results/lstm_forecaster/batadal_injected_recovery/summary.csv` and `analysis/INSPECTION.md`; all four scenarios ran, but recovery failed (support F1 0.061-0.092, sign accuracy 0.021-0.042, onset error 317-1,117 samples).

- [x] 6. Run the locked BATADAL calibration and temporal-support diagnostic.
  **Evidence:** `results/lstm_forecaster/batadal_injected_recovery/diagnostic/INSPECTION.md`, `activation_drift_and_support.png`, and `attacked_sensor_traces.png`. Clean sensor activity remained 1.2-1.8% across all windows, while attack-phase activity on injected sensors was 71-100%. The failure mechanism is global pointwise support selection, not threshold drift or absent residual signal.

- [/] 7. Implement and evaluate a normal-calibrated temporal-support estimator.
  **Implementation evidence:** `src/evaluation/temporal_support.py`, `tests/spec/test_temporal_support.py`, `scripts/run_temporal_support_recovery.py`, and `scripts/analyze_temporal_support_recovery.py`; 9 focused recovery/support contracts pass locally. **Owner action remains:** run the comparison on Vista and copy back `results/lstm_forecaster/batadal_temporal_support_recovery/analysis/` plus `summary.csv`.

- [ ] 8. Design the shared-architecture HAI/SWaT/WADI conditioning audit after Task 7 fixes or bounds the BATADAL support-selection mechanism.
  **Acceptance criteria:** one versioned multi-dataset configuration contract, normal-regime/split audit, compact per-dataset forecast diagnostics, and an explicit decision whether dataset-specific configuration suffices before considering any architecture variation.

## Blockers and Owner Actions

- **Owner action before Task 4:** approve the exact `idev` command and SU estimate.
- **Owner action after Task 4:** copy the ablation analysis directory locally for visual inspection.

## Immediate Next Action

Build the Task 7 temporal-support estimator, then run its known-delta comparison on Vista. Task 8 remains mandatory before any cross-dataset robustness claim.
