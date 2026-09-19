# Sprint 2 Plan: Forecaster Generalization and Injected Attack Recovery

## Objective

Convert the Sprint 1 LSTM prototype into a defensible attack-modeling measurement layer. First establish a stable normal-state forecaster on BATADAL. Then evaluate whether its residual estimates recover known injected additive attacks.

## Current State

Tasks 1 through 4 are complete. The reduced `gh-dev` ablation selected `context-32_hidden-128_epochs-10`: 0.132595 best validation loss, 2.95% FPR, 54.3% point TPR, 100% event recall, and one-sample median delay. `scripts/run_injected_recovery.py` and `scripts/analyze_injected_recovery.py` now implement Task 5 against that checkpoint. The bounded `gh-dev` recovery smoke and its compact artifact inspection are next.

## Execution Sequence

1. Run a bounded BATADAL LSTM ablation over context length, hidden size, and epoch budget using fixed normal-only splits.
2. Compare runs using validation loss, normal-test FPR, point TPR, event recall, delay, training curves, and score plots.
3. Freeze the best BATADAL configuration only if validation loss decreases and its operating point remains bounded.
4. Build an injected attack evaluation runner around the frozen forecaster: step, ramp, periodic, and multi-sensor attacks.
5. Score recovered residuals against known delta with support, magnitude, sign, onset, duration, and uncertainty diagnostics.

## Scope Boundaries

**In scope:** BATADAL configuration ablation, reproducible experiment configs, injected observation-only attacks, statistical recovery analysis.

**Out of scope:** Per-dataset architecture forks, full multi-hour training, actor/process attacks, and agentic multi-attack reasoning. Each requires a later decision after Sprint 2 evidence.

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
