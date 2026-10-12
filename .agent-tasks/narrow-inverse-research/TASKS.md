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

- [/] 4. Freeze the identifiability and measurement-noise contract.
  **Acceptance:** document trusted inputs, normal-dynamics restrictions, synthetic
  versus real-data noise semantics, prohibited claims, and at least one competing
  feasible decomposition with ambiguity handling.
  **Current step:** write the mathematical contract and a falsifying same-`y`
  decomposition before implementing estimator interfaces.

- [ ] 5. Implement and validate the chronological benchmark manifest.
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

Write the identifiability and measurement-noise contract with an executable
same-observation ambiguity example.
