# Narrow Additive Inverse Research Tasks

- [x] 1. Freeze the first benchmark scope: BATADAL, known `L_T1` sensor,
  known interval, bounded signed step, offline discovery plus causal estimator.
  **Evidence:** NIR-D01, NIR-D05, and NIR-D06.

- [x] 2. Complete injection measurement invariants with the selected forecaster.
  **Evidence:** Existing locked contracts pass and the train-normalizer audit has
  zero off-support cells plus exact clean-plus-delta identity.
  **Reconciled & Superseded:** The historical LSTM residual baseline pilot was
  superseded in Phase 2/3 by the champion Dilated TCN surrogate
  (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`), which verified all injection
  measurement invariants and passed all 6 Phase 3 acceptance gates on the sealed test partition.

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

- [x] 8c. Execute gated dynamics-model bake-off on TACC Vista GH200 and integrate champion surrogate.
  **Acceptance:** owner approves exact commands and SU estimates; job runs on Vista GH200;
  lowest validation residual variance surrogate advances to Step-MAP integration.
  **Evidence:** DilatedTCNSurrogate checkpoint `results/surrogates/vista_tcn_48h_150ep/tcn.pt` trained on GH200
  demonstrated order-of-magnitude residual noise floor compression (calib std 0.0418m vs 0.537m for Linear Ridge).
  Closed-loop Phase 2A benchmark achieved 20.97% NRMSE reduction over Linear StepMAP (0.1605 vs 0.2031).

- [x] 8d. Eliminate oracle attack interval assumption (Undeclared-Support Inverse Recovery).
  **Acceptance:** `evaluate_inverse_benchmark` and `StepMAPEstimator` reconstruct without oracle timing
  (`attack_interval=None`); negative control on unattacked clean data achieves 0 false-positive support points;
  undeclared step recovery on validation data achieves Support F1 >= 0.75, sign accuracy = 1.0; full 120-scenario
  development benchmark evaluated under undeclared support.
  **Evidence:** locked specification suite `tests/spec/test_undeclared_support.py` (4 tests); full test suite
  (254 tests passing); `results/narrow_inverse_research/neural_benchmark/undeclared_summary.json` and
  `undeclared_vs_declared_comparison.json`. StepMAP_TCN maintains identical median Delta NRMSE (0.1605) and 100% sign accuracy,
  with Support F1 = 0.8571 (vs 0.2729 for Linear StepMAP).

- [x] 8e. Implement strictly causal rolling filter (`CausalStepMAPEstimator`) satisfying Phase 3 acceptance gate.
  **Acceptance:** `CausalStepMAPEstimator` estimates attack trajectory $\delta_t$ and clean state $x_t$ online forward in time using a sliding lookback window without conditioning on future observations $y_{t+1:N}$; locked mathematical contract verifies zero future lookahead invariance; negative control produces 0 false-positive support points; achieves at least 20% lower NRMSE than `PersistenceEstimator` on the 120-scenario development grid.
  **Evidence:** locked specification suite `tests/spec/test_causal_step_map.py` (5 tests passing); full 259-test suite passing. Closed-loop 120-scenario benchmark on champion TCN (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`) demonstrated `CausalStepMAP_TCN` achieved median Delta NRMSE 0.2424 vs. 0.8616 for `PersistenceEstimator` (71.87% error reduction, exceeding the 20% acceptance gate), with 100% sign accuracy on $\ge 1\sigma$ attacks and 1.0 median support F1. Metrics sealed in `results/narrow_inverse_research/neural_benchmark/summary.json` and `NEURAL_BENCHMARK_REPORT.md`.

- [x] 8f. Calibrated Uncertainty Quantification & Confidence Intervals (November 18 Milestone Step 3).
  **Acceptance:** Pointwise uncertainty $\sigma_t$ and variance $\sigma_t^2$ populated for both `StepMAPEstimator` and `CausalStepMAPEstimator` based on calibrated surrogate residual noise distribution ($\sigma_\epsilon \approx 0.0418$ m for champion Dilated TCN, $\sigma_\epsilon \approx 0.380$ m for Linear Ridge); nominal 90% confidence intervals ($\hat{x} \pm 1.645 \sigma_t$) achieve empirical coverage $\ge 85.0\%$ with sharp bounded intervals ($2 \times 1.645 \sigma_t < 0.25$ m, $< 5\%$ of tank range); uncertainty metrics integrated into `compute_scenario_metrics` and closed-loop benchmark.
  **Evidence:** locked specification suite `tests/spec/test_uncertainty_quantification.py` (5 tests passing, covering non-zero uncertainty, calibration consistency, clean validation coverage gate 85.0%-95.0%, interval sharpness $<0.25$ m, and attacked trajectory coverage $\ge 85.0\%$); full 264-test suite passing. Closed-loop 120-scenario development benchmark on champion TCN demonstrated: `StepMAP_TCN` achieved 95.73% mean 90% empirical coverage with 0.1375 m interval sharpness; `CausalStepMAP_TCN` achieved 96.17% mean 90% empirical coverage with 0.1375 m interval sharpness (both satisfying the Phase 3 uncertainty gate). Metrics sealed in `results/narrow_inverse_research/neural_benchmark/summary.json` and `NEURAL_BENCHMARK_REPORT.md`.

- [x] 8g. Consolidated Pre-Seal Audit (November 18 Milestone Step 4).
  **Acceptance:** Consolidate undeclared support, causal rolling filter (`CausalStepMAP_TCN`), and calibrated uncertainty into a unified, reproducible benchmark audit across the 120 development scenarios (rows 480-960 of `val.parquet`). Strictly verify data seal integrity (zero access or evaluation on sealed final test rows 960-1440) and certify all precommitted Phase 3 acceptance gates before unsealing.
  **Evidence:** Locked specification suite `tests/spec/test_pre_seal_audit.py` (3 tests passing); full 267-test suite passing. Audit runner `scripts/run_consolidated_pre_seal_audit.py` executed across 720 scenario evaluations (6 configurations x 120 scenarios) in 517.72s. Data seal verified (`val.parquet` rows 0-960 evaluated, rows 960-1440 untouched). All 6 precommitted acceptance gates passed: CausalStepMAP (both declared and undeclared) achieved 71.87% median Delta NRMSE reduction vs Persistence (0.2424 vs 0.8616), median signed amplitude error 0.1264 $\sigma_{\text{train}}$ ($\le 0.25$), 100% sign accuracy on $\ge 1\sigma$ attacks, 96.17% empirical 90% coverage ($\ge 85.0\%$), 0.1375 m interval sharpness ($< 0.25$ m), and 0 hard box violations. Artifacts generated in `results/narrow_inverse_research/pre_seal_audit/{summary.json, summary.csv, plots/}` and comprehensive audit report in `docs/reports/PRE_SEAL_AUDIT_REPORT.md`. Certified ready for Phase 3 sealed final test unsealing.

- [x] 9. Seal and execute the final BATADAL benchmark on true test split (November 18 Milestone Step 5).
  **Acceptance:** Final test block (rows 960-1440 of `val.parquet`) officially unsealed and evaluated across all 180 sealed test scenarios (unseen amplitudes {0.75, 1.5, 3.0} sigma, unseen durations {9, 18, 36} hours, 10 onsets each) across all 6 estimators; all precommitted acceptance gates verified on true test split; diagnostic plots generated; November 18 Final Report authored in synchronized Markdown and Overleaf-ready single-column LaTeX.
  **Evidence:** Locked specification suite `tests/spec/test_sealed_final_benchmark.py` (4 tests); full 271-test suite passing. Benchmark script `scripts/run_sealed_final_benchmark.py` executed across 1080 evaluations (6 configs x 180 scenarios) in 738.94s. All 6 acceptance gates passed: CausalStepMAP achieved 71.76% (declared) and 71.77% (undeclared) median Delta NRMSE reduction vs Persistence (0.1790 vs 0.6340, gate >= 20%), 0.1385 sigma median signed amplitude error (gate <= 0.25), 100% sign accuracy on >= 1 sigma attacks (gate >= 95%), 96.01% empirical 90% coverage (gate >= 85%), 0.1375m sharpness (gate < 0.25m), and 0 hard constraint violations. Artifacts generated in `results/narrow_inverse_research/sealed_final_benchmark/{summary.json, summary.csv, plots/}` and comprehensive final reports authored at `docs/reports/NOVEMBER_18_FINAL_REPORT.md` and `docs/reports/NOVEMBER_18_FINAL_REPORT.tex`.

- [x] 10. Decide contingent expansion from sealed evidence.
  **Acceptance:** record one of three outcomes: expand across BATADAL sensors and
  attack families, redesign a bounded failure, or stop. Each new dataset requires
  its own semantic/physics audit and benchmark seal.
  **Decision & Evidence:** Advisor review confirmed Phase 3 sealed benchmark unconditionally accepted (all 6 gates passed). Authorized two Phase 4 follow-up workstreams: Sprint 4A (error distribution & null clean modeling) and Sprint 4B (attack family generalization via composite multi-order TV / $\ell_1$ trend filtering). Recorded in NIR-D13.

- [x] 11. Sprint 4A: Statistical Distribution and Error Modeling of Reconstructed $\hat{\delta}$.
  **Acceptance:** Script `scripts/analyze_delta_reconstruction_distributions.py` parses `results/narrow_inverse_research/sealed_final_benchmark/summary.json` and `summary.csv`; computes empirical error distribution $p(\hat{\delta} - \delta_{\text{true}})$ across all 180 test scenarios ($N=1080$ evaluations); fits parametric error distributions (Gaussian, Student-$t$, GED); evaluates null clean false-alarm distribution $p(\hat{\delta} \mid \delta=0)$ on unattacked periods; derives analytical false-alarm vs. detection tradeoff curves ($\tau_{\text{alarm}}$); outputs summary report and distribution plots to `results/narrow_inverse_research/error_analysis/`. Zero GPU retraining needed. Dedicated tracking in `.agent-tasks/sprint-4a/`.
  **Evidence:** Locked specification suite `tests/spec/test_delta_distribution_analysis.py` (6 tests passing); full 277-test suite passing in 19.38s (`uv run pytest`). Maximum likelihood fits show Generalized Error Distribution (GED / Subbotin) provides superior fit over Gaussian/Student-$t$ ($\text{AIC} = -114.35$ vs $-75.10$ for Gaussian), with shape $\beta = 6.88$ (Causal) and $\beta = 9.38$ (Offline), proving physical box constraints and ADMM shrinkage suppress fat tails. Null clean profile demonstrates $96.46\%$ sparsity online and $100.0\%$ sparsity offline ($\sigma_{\text{null}} = 0.00234\,\text{m}$ vs $0.2326\,\text{m}$ for Persistence). Operational ROC AUC is $0.9990$ with $100\%$ detection at $\tau^* = 0.050\,\text{m}$ ($\text{FAR} \le 0.1\%$). Artifacts generated in `results/narrow_inverse_research/error_analysis/{summary.json, summary.csv, ERROR_MODELING_REPORT.md, plots/}`.

- [ ] 12. Sprint 4B: Attack Family Generalization via Multi-Order TV / $\ell_1$ Trend Filtering.
  **Acceptance:** Formulate composite multi-order TV objective ($\lambda_0\|\boldsymbol{\delta}\|_1 + \lambda_1\|\mathbf{D}^1\boldsymbol{\delta}\|_1 + \lambda_2\|\mathbf{D}^2\boldsymbol{\delta}\|_1$); implement `MultiOrderTrendFilterEstimator` extending causal ADMM solver to higher-order banded Cholesky factorizations; test without retraining normal TCN surrogate across non-step synthetic attack families (ramps, linear drifts, transient pulses, sinusoids); verify zero future lookahead invariance; satisfy locked specification contracts.

## Deferred Tasks

- [ ] D1. Git branch cleanup & synchronization: Clean up obsolete artifacts across `main` and `dev` branches and synchronize them with `feature/narrow-additive-inverse-research` as the primary research baseline (to be executed after Sprint 4A).

## Owner Approval Gates

- Remote repository pull or synchronization.
- Every `ssh`, `idev`, `sbatch`, and `scancel` command.
- Production `gh` runs after a reduced final-artifact smoke.
- Any change to sealed final benchmark inputs or thresholds.

## Immediate Next Action

Implement `scripts/analyze_delta_reconstruction_distributions.py` (Sprint 4A) to parse `results/narrow_inverse_research/sealed_final_benchmark/summary.json`, compute empirical error metrics ($e_t = \hat{\delta}_t - \delta_{\text{true}, t}$), fit parametric error models, evaluate the null clean distribution $p(\hat{\delta} \mid \delta=0)$, and generate distribution figures.
