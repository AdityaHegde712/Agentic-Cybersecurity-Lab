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

- [x] 5. Implement and validate the chronological benchmark manifest.
  **Acceptance:** fit, selection, calibration, development, and final blocks are
  disjoint; configs, generator, seeds, metrics, thresholds, and artifact hashes
  can be sealed deterministically.
  **Evidence:** local inspection confirmed 7,321 train rows plus 1,440 validation
  rows. The planned blocks are 5,840 fit, 1,481 selection, and three consecutive
  480-row validation blocks with hourly-contiguous boundary timestamps. Train,
  validation, metadata, and channel-manifest SHA-256 values were recorded.
  Locked tests reproduce the expected Red-stage missing-module failure for the
  typed benchmark definition, injection grids, seeds, thresholds, artifact
  hashes, canonical serialization, and fail-closed readiness. All 24 focused
  benchmark manifest tests and all 173 specification tests passed after owner-
  approved reconciliation of the metric-spec assertion. The benchmark manifest
  was persisted to `configs/manifests/batadal-benchmark-manifest-v1.json` with
  SHA-256 `83a5124a9189572e5532471f12b7f866e33f07b7411c645ea8aaff5c843a5d05`.

- [x] 6. Prepare the local-first inverse research harness.
  **Acceptance:** immutable tests, common estimator/result interfaces, baseline
  runners, aggregate reports, static plots, and a reduced CPU run reach a
  verified non-empty final artifact without remote resources.
  **Evidence:** locked specification suite `tests/spec/test_inverse_harness.py`
  (10 tests) verifies deterministic SHA-256 hash ranking, regime cutoff extraction,
  grid scenario generation, exact metric computation, and suite execution.
  All 199 specification tests pass. The execution script `scripts/run_inverse_benchmark.py`
  ran across the 120-scenario development grid (480 total evaluations), generating
  `results/narrow_inverse_research/development_benchmark/{summary.csv,summary.json,BENCHMARK_REPORT.md}`
  and 9 static comparison plots in `plots/`. `StepMAPEstimator` achieved 76.42% NRMSE
  improvement over the best causal baseline, meeting all precommitted acceptance gates.

- [x] 7. Implement the locked mathematical baselines and Bayesian/MAP pair.
  **Acceptance:** persistence/seasonal, robust regression/state-space,
  filter/smoother, and constrained sparse/TV estimators emit the same result
  schema; the causal path recursively uses reconstructed clean history.
  **Evidence:** locked specification suite `tests/spec/test_inverse_estimators.py`
  (16 tests) verifies `ReconstructionResult`, `BaseDeltaEstimator`,
  `PersistenceEstimator`, `SeasonalNaiveEstimator` (diurnal 24h),
  `CompanionRegressionEstimator` (Ridge on 42 declared-unaffected sensors),
  and `StepMAPEstimator` (fused lasso ADMM and Scipy L-BFGS-B under `L_T1`
  empirical bounds). All 16 estimator tests and all 199 full specification tests pass.

- [x] 7b. Execute development benchmark and author October 18 Progress Report.
  **Acceptance:** 480 scenario runs evaluated; 2x2 grid decomposition plots generated;
  uninformative F1 chart replaced with `amplitude_error_by_regime.png`;
  all precommitted acceptance criteria met (Step-MAP: 0.2031 median NRMSE, 76.42%
  improvement, 0.1441 std amp error, 100% sign accuracy); full report authored at
  `docs/reports/OCTOBER_18_PROGRESS_REPORT.md` with rendered LaTeX math and image links.

- [x] 8a. Implement surrogate model suite and PyTorch companion dataloaders.
  **Acceptance:** `BasePlantSurrogate`, `SurrogatePrediction`, `CompanionDataset`
  windowing, `RidgeCompanionSurrogate`, `PhysicsGuidedMLPSurrogate` (with hydraulic mass-balance
  loss), and `DilatedTCNSurrogate` (causal dilated 1D-CNN) pass locked contracts.
  **Evidence:** 9 locked spec tests in `tests/spec/test_surrogates.py` pass; all 248 test suite tests pass.

- [x] 8b. Author TACC Vista GH200 execution suite and verify local CPU smoke path.
  **Acceptance:** `scripts/train_surrogates_vista.py` and `scripts/slurm/train_surrogates_vista.slurm`
  (targeting `gh` partition) authored; `--smoke --model all` produces checkpoints and metrics.
  **Evidence:** local smoke test produced `results/surrogates/vista_bakeoff/{ridge,pinn,tcn}.pt`,
  `metrics.json`, and `BAKEOFF_SUMMARY.md`.

- [ ] 8c. Execute gated dynamics-model bake-off on TACC Vista GH200.
  **Acceptance:** owner approves exact commands and SU estimates; job runs on Vista GH200;
  lowest validation residual variance surrogate advances to Step-MAP integration.

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

Request owner approval for remote execution of `train_surrogates_vista.slurm` on TACC Vista GH200
with exact command line and Service Unit (SU) estimate (Step 4).
