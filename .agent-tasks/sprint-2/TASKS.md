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

- [/] 6. Run the locked BATADAL calibration and temporal-support diagnostic.
  **Implementation evidence:** `tests/spec/test_injected_recovery_diagnostic.py`, `tests/spec/test_injected_recovery_diagnostic_analysis.py`, `src/evaluation/injected_recovery_diagnostic.py`, `scripts/run_injected_recovery_diagnostic.py`, and `scripts/analyze_injected_recovery_diagnostic.py`; 10 focused contracts pass locally. **Owner action remains:** run the two Vista commands and copy back only `results/lstm_forecaster/batadal_injected_recovery/diagnostic/analysis/` plus the two diagnostic summary CSVs.

- [ ] 7. Design the shared-architecture HAI/SWaT/WADI conditioning audit after Task 6 identifies the BATADAL recovery failure mechanism.
  **Acceptance criteria:** one versioned multi-dataset configuration contract, normal-regime/split audit, compact per-dataset forecast diagnostics, and an explicit decision whether dataset-specific configuration suffices before considering any architecture variation.

## Blockers and Owner Actions

- **Owner action before Task 4:** approve the exact `idev` command and SU estimate.
- **Owner action after Task 4:** copy the ablation analysis directory locally for visual inspection.

## Immediate Next Action

Run Task 6 on Vista, inspect its compact artifacts, then select either a calibration/support remediation or a forecaster/attack-dynamics remediation. Task 7 remains mandatory before any cross-dataset robustness claim.
