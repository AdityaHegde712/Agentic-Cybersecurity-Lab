# Sprint 2 Tasks: Proposed and Atomic

- [x] 1. Add a repository-tracked BATADAL ablation configuration contract with fixed seed, dataset, context length, hidden size, epochs, and output directory.
  **Evidence:** `configs/experiments/batadal_lstm_ablation.json`, `src/experiments/lstm_ablation_config.py`, and `tests/spec/test_lstm_ablation_config.py`; focused configuration and LSTM contracts: 8 passed.

- [/] 2. Extend the LSTM runner to execute a named configuration grid and write one manifest plus separate result folders per run.
  **Implementation evidence:** `src/experiments/lstm_ablation_runner.py`, `scripts/run_lstm_forecaster.py --ablation-config`, and `tests/spec/test_lstm_ablation_runner.py`; 10 focused contracts passed.
  **Remaining acceptance:** run the reduced GPU smoke after owner approval and inspect its generated artifacts.

- [/] 3. Add compact ablation analysis plots and a Markdown selection report.
  **Implementation evidence:** `src/experiments/lstm_ablation_analysis.py`, `scripts/analyze_lstm_ablation.py`, and `tests/spec/test_lstm_ablation_analysis.py`; the report ranks validation loss, FPR, event recall, and delay without loading score rows.
  **Remaining acceptance:** inspect the real generated `results/lstm_forecaster/batadal_ablation/analysis/` artifacts after the GPU smoke.

- [ ] 4. Obtain owner approval and run the bounded BATADAL `gh-dev` smoke.
  **Acceptance:** only compact analysis artifacts are copied back; selected configuration decision is recorded.

- [ ] 5. Add an injected additive-attack forecaster evaluation runner and locked recovery assertions.
  **Acceptance:** known delta produces support, magnitude, sign, onset, and duration metrics; real labels are not used as delta ground truth.

## Blockers and Owner Actions

- **Owner action before Task 4:** approve the exact `idev` command and SU estimate.
- **Owner action after Task 4:** copy the ablation analysis directory locally for visual inspection.

## Immediate Next Action

Review the exact bounded `gh-dev` smoke command and SU estimate, then request owner approval before execution.
