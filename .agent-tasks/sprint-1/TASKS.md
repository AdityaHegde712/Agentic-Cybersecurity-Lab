# Sprint 1 Tasks: Completed

- [x] Define the injected observation-only additive attack and recovery-metric contract.
- [x] Implement CUSUM, EWMA, and PCA/SPE statistical baselines across all canonical datasets.
- [x] Correct CUSUM accumulation and calibrate with contiguous normal-validation blocks.
- [x] Add persistence-residual baselines and compact plot/report inspection tooling.
- [x] Implement the normal-only LSTM forecaster with dataloaders, checkpoints, prediction, residual scoring, plots, and live epoch reporting.

## Final Evidence

| Deliverable | Evidence |
|---|---|
| Raw score failure diagnosed | `results/statistical_baselines/calibrated_smoke_analysis/INSPECTION.md` |
| Residual baseline improvement | `results/statistical_baselines/residual_smoke_analysis/INSPECTION.md` and `score_distribution_shift.png` |
| LSTM smoke outcome | `results/lstm_forecaster/gh_dev_smoke_analysis/INSPECTION.md` and `*_loss.png` |
| Meeting-ready interpretation | `RESEARCH_PROGRESS_REPORT.md` |

## Handoff

Sprint 1 is closed. Do not reopen it to tune raw-level score thresholds. Continue through `.agent-tasks/sprint-2/`.
