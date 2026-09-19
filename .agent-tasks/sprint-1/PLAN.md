# Sprint 1 Plan: Completed Baseline and Forecaster Foundation

## Outcome

Build the measurement layer needed before modeling temporal additive attacks. The accepted research boundary is observation-only sensor corruption: `observed(t) = clean(t) + delta(t)`. Real ICS labels support detection and localization only; injected attacks supply the only valid delta-recovery ground truth.

## Completed Execution Flow

1. Canonical HAI, SWaT, WADI, and BATADAL datasets feed a common feature-preparation boundary.
2. Raw statistical scores established a failure mode: normal operating-regime shifts can resemble attacks.
3. One-step persistence residual scores removed much of that level-shift behavior and provided event-localization evidence.
4. A normal-only LSTM next-step forecaster produced sustained forecast residuals, checkpoints, training histories, scores, and compact plots.
5. Vista smoke runs produced evidence artifacts without loading raw score files into agent context.

## Delivered Paths

| Area | Paths |
|---|---|
| Additive attack contract | `src/attacks/additive.py`, `src/evaluation/attack_metrics.py` |
| Statistical baselines | `src/evaluation/statistical_baselines.py`, `src/evaluation/baseline_runner.py`, `scripts/run_statistical_baselines.py` |
| Statistical inspection | `src/evaluation/result_analysis.py`, `scripts/analyze_baseline_results.py` |
| LSTM forecaster | `src/models/lstm_forecaster.py`, `scripts/run_lstm_forecaster.py` |
| LSTM inspection | `scripts/analyze_lstm_results.py` |
| Supervisor summary | `RESEARCH_PROGRESS_REPORT.md` |

## Evidence Gates Completed

- Local automated verification: 108 tests passed before the LSTM progress-reporting follow-up.
- CPU LSTM reduced end-to-end smoke created checkpoint, training metrics, scores, and plots.
- Vista `gg` 10,000-row residual baseline smoke completed and was inspected through aggregate reports and PNGs.
- Vista `gh-dev` 10,000-row, five-epoch LSTM smoke completed and was inspected through aggregate reports and PNGs.

## Sprint 1 Exit State

The code baseline is local `dev` commit `cca6d55`; remote synchronization is owner-verified, not assumed. Sprint 2 must preserve the attack-ground-truth boundary and must not treat real dataset labels as true delta values.
