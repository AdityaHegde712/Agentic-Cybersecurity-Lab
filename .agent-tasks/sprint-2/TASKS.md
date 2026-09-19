# Sprint 2 Tasks: Proposed and Atomic

- [x] 1. Add a repository-tracked BATADAL ablation configuration contract with fixed seed, dataset, context length, hidden size, epochs, and output directory.
  **Evidence:** `configs/experiments/batadal_lstm_ablation.json`, `src/experiments/lstm_ablation_config.py`, and `tests/spec/test_lstm_ablation_config.py`; focused configuration and LSTM contracts: 8 passed.

- [x] 2. Extend the LSTM runner to execute a named configuration grid and write one manifest plus separate result folders per run.
  **Evidence:** `src/experiments/lstm_ablation_runner.py`, `scripts/run_lstm_forecaster.py --ablation-config`, `results/lstm_forecaster/batadal_ablation/manifest.json`, and `summary.csv`.

- [x] 3. Add compact ablation analysis plots and a Markdown selection report.
  **Evidence:** `results/lstm_forecaster/batadal_ablation/analysis/INSPECTION.md` and `ranked_metrics.png`; the real artifact inspection matches the aggregate table and did not use score rows.

- [x] 4. Obtain owner approval and run the bounded BATADAL `gh-dev` smoke.
  **Evidence:** all four 10,000-row runs completed. The selected configuration is `context-32_hidden-128_epochs-10`; compact artifacts were copied locally and inspected.

- [/] 5. Add an injected additive-attack forecaster evaluation runner and locked recovery assertions.
  **Implementation evidence:** `configs/experiments/batadal_injected_recovery.json`, `src/evaluation/injected_recovery.py`, `scripts/run_injected_recovery.py`, `scripts/analyze_injected_recovery.py`, and the locked `tests/spec/test_injected_recovery*.py` contracts.
  **Remaining acceptance:** run the selected checkpoint on the 1,440-row selection-validation pilot, then inspect only its `summary.csv`, `manifest.json`, `analysis/INSPECTION.md`, and `analysis/recovery_metrics.png`.

## Blockers and Owner Actions

- **Owner action before Task 4:** approve the exact `idev` command and SU estimate.
- **Owner action after Task 4:** copy the ablation analysis directory locally for visual inspection.

## Immediate Next Action

Review the exact bounded `gh-dev` recovery smoke command and SU estimate, then request owner approval before execution.
