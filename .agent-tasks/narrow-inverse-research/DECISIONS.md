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

**Status:** accepted on 2026-10-11.

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

## NIR-D05: Known Window and Sensor Define the First Conditional Inverse

**Status:** accepted on 2026-10-11.

**Decision:** Supply the attacked sensor and interval to the first benchmark.
Estimate the clean measurement, signed step amplitude/trajectory, and uncertainty.

**Consequence:** Support localization is not a primary success metric for the
first benchmark. Blind localization is a separately sealed later problem.

## NIR-D06: Offline Discovery and Causal Estimation Are Separate Gates

**Status:** accepted on 2026-10-11.

**Decision:** Permit future clean context in an offline smoother for method
discovery, but require a matched past-only filter that recursively uses
reconstructed clean history.

**Rationale:** The existing LSTM residual feeds attacked observations into later
contexts and can attenuate sustained offsets. Offline evidence cannot substitute
for causal recovery evidence.

## NIR-D07: Physics and Dataset Constraints Require Provenance

**Status:** accepted on 2026-10-11.

**Decision:** Use recorded actuator/status channels, public C-Town physics and
topology, and clean-data operating limits through a typed channel/constraint
manifest. Treat public physics as hard only after dataset alignment is validated;
otherwise use and ablate it as a soft prior.

**Rejected alternative:** Correlation-ranked companion sensors are not promoted
to trusted constraints without independent physical evidence.

## NIR-D08: Bayesian/MAP Is the Core; Method Families Compete as Dynamics Priors

**Status:** accepted on 2026-10-11.

**Decision:** Center the first estimator on constrained Bayesian/MAP inference.
Compare state-space, NARX/NARMAX, discrete sparse-equation, LSTM, and compact
probabilistic neural dynamics under the same inverse and evaluation contract.

**Consequence:** SINDy is one candidate rather than the default. LSTM gradients
remain diagnostic. World models are contingent; RL requires an action loop.

## NIR-D09: Local-First Preparation Before Vista Execution

**Status:** accepted on 2026-10-11.

**Decision:** Build and verify manifests, tests, estimators, runners, analysis,
and a reduced final-artifact smoke locally before pulling a reviewed commit on
Vista.

**Consequence:** No remote pull or allocation use is implied by local completion.
Every remote command, queue, and SU estimate remains an owner approval gate.

## NIR-D10: Sealed Chronological Evaluation

**Status:** accepted on 2026-10-11.

**Decision:** Use five chronological fit/select/calibration/development/final
blocks and hash all benchmark inputs and thresholds before the one-time final
evaluation. Measurement-defect corrections create a new seal rather than
overwriting prior evidence.

**Rationale:** Repeated inspection of the historical four-scenario validation
sequence cannot support a held-out reconstruction claim.

## NIR-D11: Public-Model Semantics Remain Candidate Until Aligned

**Status:** accepted on 2026-10-11.

**Decision:** Record C-Town element names and EPANET units as candidate public-
model semantics, while keeping BATADAL stored units unknown and topology
alignment false until the exact dataset-generator relationship is validated.
All current physics and empirical constraints remain soft.

**Rationale:** A public model's `LPS` setting establishes that model's unit
convention, but does not prove that released BATADAL columns were stored without
conversion. Prefix similarity is not sufficient provenance for a hard rule.

**Consequence:** The manifest must separate candidate semantics, stored-unit
verification, topology alignment, evidence scope, and benchmark threat-model
assumptions. `declared_unaffected` does not mean independently trusted.
