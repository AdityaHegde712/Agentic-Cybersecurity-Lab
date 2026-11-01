# Generalized Empirical Inverse Benchmark Report Template

## 1. Executive Research Finding
- **Core Thesis**: Concisely state the physical or mathematical inversion mechanism validated or refuted (e.g. failure of autoregressive state propagation under sustained spoofing vs. decoupling via companion subspace).
- **Headline Result**: Key quantitative metric of the proposed inverse estimator versus the strongest causal baseline (percentage error reduction, bootstrap confidence).
- **Identifiability Boundary**: Exact mathematical and physical conditions under which $\delta_t$ is identifiable (e.g. uncorrupted companion subspace, regularized attack family, physical plant bounds).

## 2. Inverse Problem Formulation & Mathematical Inversion Mechanism
- **Observation Model**: Formulate $y_t = x_t + \delta_t$ with explicit target channel(s), attack family, and declared uncorrupted subspace.
- **Two-Stage Inversion Architecture**:
  - **Stage 1 (Plant Surrogate)**: Estimation of clean plant state $\hat{x}_{\text{surr}}$ using uncorrupted companion sensors or external actuators.
  - **Stage 2 (Programmatic Optimization/MAP)**: Exact objective function:
    $$\min_{\delta} \mathcal{L}_{\text{fidelity}}(y - \hat{x}_{\text{surr}} - \delta) + \lambda_{\text{struct}} \mathcal{R}_{\text{struct}}(\delta) + \lambda_{\text{sparse}} \mathcal{R}_{\text{sparse}}(\delta)$$
    Subject to verified physical bounds (e.g. $y_t - x_{\max} \le \delta_t \le y_t - x_{\min}$).

## 3. Empirical Benchmark & Acceptance Ledger
- **Estimator Comparison Table**:
  - Median and mean pointwise Delta NRMSE on declared attack interval.
  - Support $F_1$ score (temporal onset and offset recovery).
  - Signed amplitude error (in training standard deviations).
  - Clean $x$ reconstruction NRMSE.
  - Sign accuracy on attacks $\ge 1.0\sigma$.
  - Percentage improvement over best causal baseline.
- **Precommitted Acceptance Gates**: Explicit table comparing empirical results against pre-sealed pass/fail thresholds.

## 4. Failure Modes & Regime Degradation Analysis
- **Temporal Duration Breakdown**: Analysis of why and where autoregressive self-history breaks down as attack duration scales (e.g. error curves across duration horizons).
- **Amplitude & Noise Floor Breakdown**: Performance across weak ($<1\sigma$) vs. strong ($>2\sigma$) attacks.
- **Dynamic Operating Regime Sensitivity**: Performance stability across low, medium, and high process dynamic regimes.

## 5. Contribution to Inverse Recovery & Next Steps
- **Generalization Steps**: How this specific empirical result advances the broader unconstrained inverse problem.
- **Current Model Limitations**: Remaining edge cases, residual noise floors, or unmodeled nonlinearities.
- **Concrete Next Milestone**: Immediate next computational experiment or model scaling step.
