# Narrow Additive Inverse Research Tasks

- [x] 1. Freeze the first benchmark scope: BATADAL, known `L_T1` sensor,
  known interval, bounded signed step, offline discovery plus causal estimator.
  **Evidence:** NIR-D01, NIR-D05, and NIR-D06.

- [/] 2. Complete injection measurement invariants with the selected forecaster.
  **Evidence:** Existing locked contracts pass and the train-normalizer audit has
  zero off-support cells plus exact clean-plus-delta identity.
  **Blocker:** the selected BATADAL checkpoint is absent locally.
  **Acceptance:** restore or regenerate it, record SHA-256/provenance, and rerun
  the unchanged historical estimator under the corrected support contract.

- [x] 3. Add the channel, trust, physics, and constraint manifest.
  **Acceptance:** every BATADAL variable has role, unit, attackability, trust,
  bounds/rates, topology source, and hard/soft status; correlations remain
  explicitly non-authoritative.
  **Evidence:** locked schema/validation tests now cover exact channel coverage,
  provenance, model-fit-only limits, trust/physics guards, and deterministic
  serialization; the red stage failed on the absent production module. The
  typed schema now passes all 5 focused tests and the full 101-test immutable
  specification suite.
  The real 43-channel builder contract now fails at the intentionally absent
  module and covers model-fit limits plus candidate/soft-only semantics.
  The builder passes that contract; an additional semantic gate now isolates
  stored-unit, canonical-reference, scoped-provenance, and enforcement-mode
  failures. Those corrections now pass all 13 focused tests and the full 109-
  test immutable specification suite.
  The deterministic artifact writer and versioned 43-channel JSON manifest now
  pass all 16 focused tests and the full 112-test immutable specification suite.
  The artifact SHA-256 is
  `e20b7728cbf4d22ea6fd6dbd4f5eca0b26f65a9165108fa8f8d8d890ad0bd347`.

- [x] 4. Freeze the identifiability and measurement-noise contract.
  **Acceptance:** document trusted inputs, normal-dynamics restrictions, synthetic
  versus real-data noise semantics, prohibited claims, and at least one competing
  feasible decomposition with ambiguity handling.
  **Evidence:** the proposed mathematical contract includes a BATADAL-scaled
  same-`y` witness. Locked observation/ambiguity tests reproduce the expected
  red-stage failure at the absent production module.
  Adversarial review found six bounded claim/operational gaps; the contract now
  separates `q`, `x`, and `y`, hard-set identifiability, posterior concentration,
  and finite-search status. Four additive tests reproduce the expected red stage
  for the missing exact-addition and certified-diameter API. The hardened API
  passes all 11 focused tests and all 123 immutable specification tests; the
  exact toy witness reports separation `0.5`, `ambiguous=true`, and
  `certified=false`. A second adversarial review rejected caller-controlled
  global certification and under-specified witness checks. The final additive
  Red gate fails at the intentionally absent evidence-backed hard-set
  certificate while preserving all prior assertions.
  A third adversarial review found that one newly frozen test falsely permits
  metadata-only structural certification. The immutable-test rule requires
  owner permission to replace that assertion with a hash-resolved evidence
  payload and certified global diameter bound.
  **Owner decision:** permission granted to correct the flawed certificate test.
  The corrected contract now passes 26 focused and 138 full specification tests:
  canonical evidence JSON and its SHA-256 resolve, certificate fields match the
  payload, uniqueness requires a global diameter bound at most `tau_delta`, and
  contradictory witness/certificate evidence is rejected.
  Final review found the certificate was not bound to the observation, benchmark
  case, norm, or full hard-set seal. The owner-authorized tests now reproduce the
  expected Red-stage missing-fingerprint failure for those exact bindings. The
  implementation now passes 32 focused and 144 full specification tests and
  rejects cross-observation, cross-norm, cross-case, and manifest reuse. A final
  review required the hard-set seal to include the actual support, tolerance,
  attack class, dynamics artifact, and numeric constraint parameters. That full
  schema now passes 36 focused and 148 full specification tests.
  Unverified candidate separation now remains diagnostic-only; 37 focused and
  149 full specification tests pass. Final adversarial review accepted the claim
  boundary.

- [/] 5. Implement and validate the chronological benchmark manifest.
  **Acceptance:** fit, selection, calibration, development, and final blocks are
  disjoint; configs, generator, seeds, metrics, thresholds, and artifact hashes
  can be sealed deterministically.

- [ ] 6. Prepare the local-first inverse research harness.
  **Acceptance:** immutable tests, common estimator/result interfaces, baseline
  runners, aggregate reports, static plots, and a reduced CPU run reach a
  verified non-empty final artifact without remote resources.

- [ ] 7. Implement the locked mathematical baselines and Bayesian/MAP pair.
  **Acceptance:** persistence/seasonal, robust regression/state-space,
  filter/smoother, and constrained sparse/TV estimators emit the same result
  schema; the causal path recursively uses reconstructed clean history.

- [ ] 8. Run the gated dynamics-model bake-off.
  **Acceptance:** state-space, NARX/NARMAX, discrete equation-discovery, LSTM, and
  one probabilistic neural candidate are evaluated only on authorized blocks;
  at most one statistical/system-ID and one neural model advance.

- [ ] 9. Seal and execute the final BATADAL benchmark on Vista.
  **Acceptance:** owner approves exact commands and SU estimates; reduced remote
  smoke passes first; sealed metrics, uncertainty, ambiguity, job ledger, and
  compact artifacts are complete for the November 18 proposal.

- [ ] 10. Decide contingent expansion from sealed evidence.
  **Acceptance:** record one of three outcomes: expand across BATADAL sensors and
  attack families, redesign a bounded failure, or stop. Each new dataset requires
  its own semantic/physics audit and benchmark seal.

## Owner Approval Gates

- Remote repository pull or synchronization.
- Every `ssh`, `idev`, `sbatch`, and `scancel` command.
- Production `gh` runs after a reduced final-artifact smoke.
- Any change to sealed final benchmark inputs or thresholds.

## Immediate Next Action

Write locked chronological-split and deterministic benchmark-seal tests before
implementing the Task 5 manifest.
