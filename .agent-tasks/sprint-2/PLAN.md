# Sprint 2 Plan: Forecaster Generalization and Injected Attack Recovery

## Objective

Convert the Sprint 1 LSTM prototype into a defensible attack-modeling measurement layer. First establish a stable normal-state forecaster on BATADAL. Then evaluate whether its residual estimates recover known injected additive attacks and diagnose calibration failures before modifying the estimator. Preserve HAI, SWaT, and WADI as a planned shared-model conditioning audit rather than silently narrowing the research scope.

## Current State

Tasks 1 through 5 are complete. The reduced `gh-dev` ablation selected `context-32_hidden-128_epochs-10`: 0.132595 best validation loss, 2.95% FPR, 54.3% point TPR, 100% event recall, and one-sample median delay. The 1,440-row selection-validation injection pilot then failed the recovery gate: support F1 was 0.061-0.092, sign accuracy 0.021-0.042, and onset error 317-1,117 samples. The next work is a calibration and temporal-support diagnosis, not an architecture change or recovery claim.

## Execution Sequence

1. Run a bounded BATADAL LSTM ablation over context length, hidden size, and epoch budget using fixed normal-only splits.
2. Compare runs using validation loss, normal-test FPR, point TPR, event recall, delay, training curves, and score plots.
3. Freeze the best BATADAL configuration only if validation loss decreases and its operating point remains bounded.
4. Build an injected attack evaluation runner around the frozen forecaster: step, ramp, periodic, and multi-sensor attacks.
5. Score recovered residuals against known delta with support, magnitude, sign, onset, duration, and uncertainty diagnostics.
6. Diagnose clean residual-threshold drift and attacked-sensor temporal support before changing the forecaster or thresholding rule.
7. After a BATADAL recovery mechanism passes its redesign gate, audit HAI, SWaT, and WADI normal splits, sensor conditioning, and shared-architecture configuration under the same protocol.

## Scope Boundaries

**In scope:** BATADAL configuration ablation, reproducible experiment configs, injected observation-only attacks, calibration diagnostics, and statistical recovery analysis.

**Out of scope:** Per-dataset architecture forks, full multi-hour training, actor/process attacks, and agentic multi-attack reasoning. HAI, SWaT, and WADI receive a later shared-architecture conditioning audit before any architecture fork.

## Exit Criteria

- One BATADAL configuration completes a `gh-dev` reduced smoke with a checkpoint, loss history, score CSV, summary, and inspection plots.
- The selected configuration has decreasing validation loss and no worse than 5% normal-test FPR in its smoke result.
- The injected evaluation produces known-delta recovery metrics without modifying existing locked tests.
- A compact report distinguishes synthetic delta recovery from real-dataset detection evidence.

## Operational Constraints

- Use `gh-dev` for every neural smoke; user must approve the allocation command before execution.
- Estimate and state SU cost before any Vista allocation command.
- Use the actual Vista venv name `agentic_cyber`, after loading the Python/CUDA module stack.
- Write all outputs under repository-relative `results/`; inspect only generated reports and plots.
