# Narrow Additive Inverse Research Decisions

## NIR-D01: Single-Sensor Step Attack Is the First Inverse Benchmark

**Status:** accepted on 2026-10-10.

**Decision:** Start with one bounded, finite-duration, signed step attack on
one sensor. Sensor coupling is deferred until this setting has traceable
recovery behavior.

**Rationale:** This reduces a free time-by-sensor reconstruction to four
parameters: sensor, onset, duration, and amplitude. It makes failure modes and
identifiability limits inspectable.

## NIR-D02: Benchmark Validity Precedes New Estimators

**Status:** accepted.

**Decision:** Complete the existing injection ground-truth audit before
optimizing, differentiating through, or assigning a posterior to the LSTM.

**Rationale:** Exact-nonzero support after float32 normalization reconstruction
may not represent the intended attack support.

## NIR-D03: Optimization Before RL or World Models

**Status:** proposed.

**Decision:** Use a constrained optimization/MAP baseline first. Treat LSTM
gradients as diagnostics. Defer RL and world-model approaches until there is a
valid passive-reconstruction baseline.

**Rationale:** The current task has no intervention/action loop. RL becomes
relevant later for active sensing or experimental design, not first-pass
observation-only reconstruction.

## NIR-D04: Conditional Claim Only

**Status:** accepted on 2026-10-10.

**Decision:** Do not claim general recovery of an unknown observation-only
attack from `y` alone. Report only conditional recovery under an explicit
normal-dynamics and attack-generator contract.

**Rationale:** A bounded step can be moved between the unobserved clean signal
and delta without changing `y` unless the benchmark introduces independent
constraints such as trusted companion sensors, known inputs, or a justified
restricted clean-dynamics class.
