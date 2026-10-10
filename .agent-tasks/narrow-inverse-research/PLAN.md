# Narrow Additive Inverse Research Plan

**Status:** proposed design; no inverse implementation or compute run has started.

## Objective

Establish whether a frozen normal-state predictor can reconstruct the narrowest
known additive attack from observations. The first benchmark is a BATADAL,
single-sensor, finite-duration, bounded signed step attack.

**Claim boundary:** this benchmark can establish conditional recovery under its
declared synthetic generator and normal-dynamics assumptions. It cannot show
that an arbitrary unknown observation-only attack is identifiable from `y`
alone.

## Frozen Benchmark Boundary

- Dataset and clean source: BATADAL normal rows, with a chronological split
  reserved for inference evaluation.
- Normal model: the existing frozen
  `context-32_hidden-128_epochs-10` checkpoint; it is not retrained per attack.
- Observation model: `y = x + delta`; actuator/process manipulation is out of
  scope.
- Attack family: `delta_t = a * 1[t0 <= t < t1] * e_j`.
- Unknown parameter vector: sensor `j`, onset `t0`, duration `t1 - t0`, and
  signed amplitude `a`.
- Bounds: attacks must remain inside declared sensor/data validity bounds;
  invalid proposals are rejected, never silently clipped.

## Sequence

1. Validate injected ground truth, no-attack behavior, normalization round
   trips, and tolerance-aware support scoring.
2. Record an explicit identifiability assumption: trusted unaffected sensors,
   known inputs, or a restricted normal-dynamics class must rule out equivalent
   clean-plus-attack decompositions.
3. Freeze an inference-only clean sequence disjoint from calibration and
   checkpoint selection.
4. Define parameter bounds from clean data and documented validity limits.
5. Implement a constrained optimization/MAP baseline and compare it with the
   existing residual control.
6. Add gradient and hidden-state diagnostics only to explain behavior; they do
   not constitute an algebraic LSTM inverse.
7. Add posterior uncertainty, SINDy/latent dynamics, coupling, RL, or world
   models only after the constrained baseline has a valid held-out result.

## Exit Criteria

- Ground truth has exact intended support under the documented numerical
  tolerance, and `observed == clean + delta` holds for no-attack and injected
  cases.
- An identifiability statement documents why the restricted clean trajectory
  and step attack cannot be exchanged without violating the benchmark's stated
  assumptions.
- The held-out benchmark records sensor, onset, duration, amplitude, support,
  sign, magnitude, and uncertainty metrics separately.
- Any reconstruction claim is limited to the declared attack family, bounds,
  dataset split, and predictor.
