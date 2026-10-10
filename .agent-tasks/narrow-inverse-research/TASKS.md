# Narrow Additive Inverse Research Tasks

- [x] 1. Freeze the first benchmark scope: BATADAL, one sensor, bounded finite
  signed step, fixed normal predictor.
  **Evidence:** NIR-D01 and the user confirmation on 2026-10-10.

- [/] 2. Specify and test the injection measurement invariants.
  **Evidence:** `tests/spec/test_injection_ground_truth_contract.py` and
  `tests/spec/test_injection_ground_truth_audit.py` pass. On actual clean
  BATADAL values with the audit-only train-fitted normalizer, the corrected
  path has zero off-support delta cells and exact clean-plus-delta identity.
  The emulated old float32 round trip creates 11,403-11,416 off-support
  nonzero cells across the four historical scenarios, with maximum absolute
  residue `1.4787e-06`.
  **Remaining blocker:** the selected forecaster checkpoint is absent locally,
  so this does not reproduce or correct the historical checkpoint-based scores.

- [ ] 3. Specify the identifiability contract and its counterexamples.
  **Acceptance:** the artifact states the trusted measurements, known inputs,
  normal-dynamics restriction, and prohibited generalization claims.

- [ ] 4. Define a clean-disjoint calibration and held-out inference split.
  **Acceptance:** checkpoint selection, threshold calibration, and final
  parameter recovery use non-overlapping temporal data.

- [ ] 5. Define bounded parameter priors and sensor/data validity ranges.
  **Acceptance:** every candidate attack is accepted or rejected by explicit,
  versioned constraints.

- [x] 5a. Produce aggregate clean-training range and companion-candidate evidence.
  **Evidence:** `scripts/audit_batadal_constraints.py` reports 7,321 normal
  training rows and 43 sensors without raw-row output. For `L_T1`, observed
  range is `0.320112-4.871329`; correlation candidates are `S_V2`, `P_J14`,
  `P_J422`, `P_J289`, and `P_J280`. These are not trusted constraints.

- [ ] 6. Design a constrained optimization/MAP baseline and immutable tests.
  **Acceptance:** parameter and delta reconstruction metrics are specified
  before implementation or evaluation.

## Immediate Next Action

Specify the measurement-invariant test contract before implementing the inverse
estimator.
