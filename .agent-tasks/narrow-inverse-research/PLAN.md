# Narrow Additive Inverse Research Plan

**Status:** Phase 3 Sealed Final Test Benchmark Complete on 2026-10-26 (all 6 precommitted gates passed on rows 960–1440 of `val.parquet`). Active Phase: Phase 4 Contingent Expansion (Sprint 4A: Error Distribution Modeling; Sprint 4B: Multi-Order TV Attack Generalization).

## Objective and Claim Boundary

Develop a mathematically defensible estimator for a sensor-observation attack

`y_t = x_t + delta_t`

under a deliberately restricted first benchmark: BATADAL, one known attacked
sensor, one known attack interval, and a bounded finite-duration signed step.
The estimator must reconstruct the clean measurement `x`, attack trajectory
`delta`, and uncertainty. Offline smoothing may use samples after the attack for
method discovery, but the final research output must include a past-only causal
form.

This project may establish conditional reconstruction under explicit dynamics,
trust, physics, noise, split, and attack assumptions. It may not claim that an
arbitrary decomposition of `y` into `x + delta` is uniquely identifiable.

## Current Evidence and Blocker

- The declared-support injection contract is corrected. Aggregate BATADAL audits
  show zero off-support delta cells and exact `observed == clean + delta` for the
  synthetic benchmark.
- The existing LSTM residual is a baseline, not an inverse estimator. It consumes
  attacked observations in later context windows, so sustained attacks can be
  absorbed into its predicted clean history.
- The selected `context-32_hidden-128_epochs-10` BATADAL checkpoint is absent
  locally. Checkpoint-dependent results cannot be reproduced until the exact
  artifact is restored or the configuration is regenerated and separately
  identified.

## Measurement and Identifiability Contract

For synthetic evaluation, `x` is the recorded clean BATADAL measurement. Normal
process and sensor variation therefore belongs to `x`, while the injected attack
is exactly `delta`. For real attack intervals, use a calibrated observation and
dynamics likelihood; real labels do not provide counterfactual `delta` ground
truth.

The first benchmark assumes that the target sensor and attack interval are
supplied and that declared companion inputs are unaffected. A versioned channel
manifest must record each variable's role, unit, continuous/status type,
attackability, trust assumption, topology mapping, empirical bounds/rates, and
source provenance. Correlation-ranked channels are candidates only and never
become trusted constraints without independent physical justification.

Public C-Town topology, pump/valve logic, units, and conservation relationships
may constrain the solution. A physical rule is hard only after its alignment
with the dataset is validated. Otherwise it is a documented soft prior and must
be ablated. Empirical bounds are derived only from the model-fit split.

The research must include a falsification case that constructs two feasible
`x/delta` decompositions for the same `y`. When constraints do not eliminate the
alternative, the estimator reports ambiguity or a multimodal posterior rather
than a falsely unique reconstruction.

## Stable Interfaces

1. `ChannelConstraintManifest` records channel semantics, trust/threat roles,
   valid ranges/rates, topology references, and hard/soft constraint provenance.
2. `InverseBenchmarkManifest` records chronological partitions, data and code
   hashes, checkpoint provenance, generator parameters, configurations, seeds,
   metrics, and pre-sealed acceptance thresholds.
3. `DeltaEstimator.fit_normal(...)` learns normal dynamics from authorized clean
   data and recorded exogenous inputs.
4. `DeltaEstimator.reconstruct(...)` consumes the observation, known sensor and
   interval, constraints, and causal/offline mode.
5. `ReconstructionResult` returns `x_hat`, `delta_hat`, uncertainty intervals,
   constraint violations, provenance, and an ambiguity flag.

## Research Method Ladder

| Tier | Methods | Role |
| --- | --- | --- |
| Diagnostic bounds | Zero attack, clean-history oracle | Lower/upper information controls; never eligible for model ranking |
| Statistical controls | Persistence, seasonal naive, robust ARX/VARX, dynamic-factor models | Locked causal baselines |
| State estimation | Kalman/state-space filter and offline smoother | Interpretable causal/offline comparison with uncertainty |
| Core inverse | Constrained sparse/group-sparse and total-variation Bayesian/MAP filter-smoother | Primary mathematical estimator |
| Learned dynamics | Existing LSTM plus one compact probabilistic sequence model | Compare learned dynamics under the same inverse objective |
| Equation discovery | Discrete sparse equations, NARX/NARMAX, symbolic candidates | Candidate dynamics priors after regime/control and stability validation |

LSTM gradients and hidden-state probes diagnose sensitivity and corrupted-history
attenuation; they are not algebraic inverses. A world model becomes eligible only
after the state/constraint contract and core baselines exist. Reinforcement
learning is deferred until there is an action loop such as active sensing,
safe simulated intervention, or experiment selection.

## Chronological Benchmark Seal

Use the 8,761 normal BATADAL rows without temporal shuffling:

| Block | Rows | Authorized use |
| --- | ---: | --- |
| Model fit | `0:5840` | Fit dynamics and empirical constraints |
| Model selection | `5840:7321` | Choose hyperparameters/model structure |
| Clean calibration | First 480 validation rows | Noise, posterior, and decision calibration |
| Development injection | Next 480 validation rows | Choose estimator family and attack prior |
| Sealed final injection | Last 480 validation rows | One-time final evaluation |

Development injections use `L_T1`, both signs, amplitudes
`{0.5, 1.0, 2.0}` train standard deviations, durations
`{6, 12, 24, 48}` hours, and five deterministic regime-stratified onsets per
combination. The sealed suite uses amplitudes `{0.75, 1.5, 3.0}`, durations
`{9, 18, 36}` hours, and ten unseen regime-stratified onsets per combination.

Before final execution, hash the data, channel/constraint manifest, generator,
checkpoint, estimator and baseline configurations, seeds, and acceptance
thresholds. A post-seal change is permitted only for a documented measurement
defect and creates a new seal; results from the old seal remain preserved.

## Phased Execution

### Phase 0: Local-First Research Harness - October 11-18, 2026

Prepare all code locally before pulling a reviewed commit on Vista: typed
manifests, split validation, constraint audit, estimator interfaces, frozen
tests, experiment runners, compact aggregate reports, plots, and a reduced CPU
end-to-end path. Restore or separately regenerate the missing checkpoint and
record its SHA-256 and provenance. Reproduce the historical estimator with the
correct support contract.

**Exit:** local tests pass; a reduced run reaches the final reconstruction
artifact; identifiability/noise contracts and sealed metric definitions exist;
the October 18 report clearly distinguishes validated evidence from planned work.

### Phase 1: Locked Mathematical Baselines - October 19-27

Implement the statistical and state-space controls, then the matched constrained
Bayesian/MAP pair: an offline smoother and a causal rolling filter that feeds
reconstructed `x_hat`, never attacked `y`, into subsequent clean history. Use a
step prior first and permit a total-variation form only as a separately identified
ablation.

**Exit:** every eligible method emits the same result schema; the reduced causal
and offline pipelines pass; ambiguity and no-attack behavior are measured.

### Phase 2: Dynamics Bake-Off - October 28-November 8

Compare state-space, NARX/NARMAX, discrete sparse-equation, LSTM, and one compact
probabilistic sequence model behind the same inverse objective. Equation-discovery
models must pass regime segmentation, actuator-input, units, stability,
blocked-time prediction, and rollout-error checks.

**Exit:** select at most one statistical/system-identification dynamics model and
one neural model using only model-selection, calibration, and development blocks.

### Phase 3: Sealed BATADAL Evaluation - November 9-17

Run the sealed step suite once, quantify uncertainty and ambiguity, and produce a
job/SU/artifact ledger for the November 18 proposal. Success requires:

- At least 20% median delta-trajectory NRMSE improvement over the best eligible
  baseline, with a paired 95% bootstrap interval excluding zero improvement.
- Median signed-amplitude error no greater than `0.25` train standard deviations
  and at least 95% sign accuracy for attacks with magnitude at least `1 sigma`.
- Nominal 90% posterior intervals with 85-95% empirical coverage; sharpness is
  reported alongside coverage.
- Zero hard-constraint violations; soft-constraint violations are reported by
  source and severity.
- The causal estimator independently beats the best causal baseline; offline
  smoothing does not substitute for this gate.

**Exit:** defensible positive result or a publishable bounded failure explaining
non-identifiability, model contamination, posterior ambiguity, or physics mismatch.

### Phase 4: Contingent Expansion - Authorized Sprints 4A & 4B

Following completion of Phase 3 and all 6 precommitted acceptance gates on the sealed test block, two follow-up workstreams are authorized:

#### Sprint 4A: Statistical Distribution & Error Modeling of Reconstructed $\hat{\delta}$
- Characterize the empirical distribution of reconstruction error $e_t = \hat{\delta}_t - \delta_{\text{true}, t}$ across all 180 test scenarios ($N=1080$ evaluations).
- Quantify normality, skewness, kurtosis, and tail risk; fit parametric models (Gaussian, Student-$t$, Generalized Error Distribution).
- Characterize the null clean false-alarm distribution $p(\hat{\delta}_t \mid \delta=0)$ across unattacked intervals to establish operational detection thresholds ($\tau_{\text{alarm}}$).
- Zero GPU retraining needed: consumes existing sealed benchmark artifacts (`results/narrow_inverse_research/sealed_final_benchmark/summary.json`).

#### Sprint 4B: Attack Family Generalization via Multi-Order Total Variation / $\ell_1$ Trend Filtering
- Generalize Stage 2 optimization beyond piecewise-constant steps to continuous ramps, linear sensor drifts, transient pulses, and sinusoidal attacks.
- Reformulate the objective with Composite Multi-Order Total Variation / $\ell_1$ Trend Filtering:
  $$\min_{\boldsymbol{\delta}} \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{x}}_{\text{surr}} - \boldsymbol{\delta}\|_2^2 + \lambda_0 \|\boldsymbol{\delta}\|_1 + \lambda_1 \|\mathbf{D}^1 \boldsymbol{\delta}\|_1 + \lambda_2 \|\mathbf{D}^2 \boldsymbol{\delta}\|_1$$
  subject to hydraulic box constraints $y_t - x_{\max} \le \delta_t \le y_t - x_{\min}$.
- First difference $\mathbf{D}^1$ enforces piecewise constancy (steps, square pulses); second difference $\mathbf{D}^2$ enforces piecewise linearity (ramps, linear drifts).
- Preserves the trained normal TCN surrogate (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`) with zero retraining because normal plant physics from uncorrupted companions is invariant to the attack profile.
- Extends the banded Cholesky ADMM solver to higher-order banded systems $[(1 + 2\rho_1 + 2\rho_2)\mathbf{I} + \rho_1 \mathbf{D}^{1T} \mathbf{D}^1 + \rho_2 \mathbf{D}^{2T} \mathbf{D}^2]$.

**Exit:** empirical error distribution parameterized; multi-order ADMM solver implemented, unit tested, and benchmarked across non-step attack families.

## Vista Compute Envelope

| Work package | SU ceiling | Queue preference |
| --- | ---: | --- |
| Checkpoint/reproduction and semantics | 50 | `gh-dev`, limited `gg` |
| Mathematical/statistical/system-ID sweeps | 150 | `gg` |
| Neural/probabilistic candidate screening | 250 | `gh-dev`, then single-node `gh` |
| Sealed replication and uncertainty | 300 | `gh` with `gg` aggregation |
| Contingent sensor/dataset transfer | 150 | Gate-dependent |
| Failure reserve | 100 | Unassigned |
| **Total ceiling** | **1,000** | |

The allocation is a ceiling, not a spending target. No production `gh` job starts
until the same code completes a reduced final-artifact smoke. Short configurations
are batched to avoid the 15-minute minimum charge. Persistent datasets,
checkpoints, and final artifacts use `$WORK`; active run I/O uses `$SCRATCH`.

Every remote action remains owner-controlled. Before `ssh`, `idev`, `sbatch`, or
`scancel`, present the exact command or script, queue, node count, walltime, and
estimated SU cost. Each run records job ID, Git commit, manifest hashes, seed,
hardware, walltime, utilization, SU charge, and output locations.

## Evidence and State Discipline

Existing test assertions remain immutable. New frozen contracts cover injection
identity, noise semantics, channel/constraint provenance, chronological split
disjointness, recursive decontamination, result shapes, posterior intervals,
ambiguity reporting, deterministic seeds, and benchmark-seal hashes.

Automated tests, local reduced E2E, Vista smoke, sealed scientific evaluation,
documentation, and publication are separate gates. After each accepted task,
update this plan, `DECISIONS.md`, and `TASKS.md`; update root `HANDOFF.md` at the
next pause or phase boundary.

## Immediate Next Action

Implement `scripts/analyze_delta_reconstruction_distributions.py` (Sprint 4A) to parse `results/narrow_inverse_research/sealed_final_benchmark/summary.json`, compute empirical error metrics ($e_t = \hat{\delta}_t - \delta_{\text{true}, t}$), fit parametric error models, evaluate the null clean distribution $p(\hat{\delta} \mid \delta=0)$, and generate distribution figures in `results/narrow_inverse_research/error_analysis/`.
