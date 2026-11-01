# Sprint 4B Decisions

## S4B-D01: Pentadiagonal Banded Cholesky Solver Architecture
- **Context:** Incorporating a second-order difference penalty $\|\mathbf{D}^2\boldsymbol{\delta}\|_1$ increases the bandwidth of $\mathbf{A} = \mathbf{I} + \rho_0 \mathbf{I} + \rho_1 (\mathbf{D}^1)^\top \mathbf{D}^1 + \rho_2 (\mathbf{D}^2)^\top \mathbf{D}^2$ from tridiagonal (bandwidth 1) to pentadiagonal (bandwidth 2).
- **Decision:** Implement specialized banded Cholesky factorization (`scipy.linalg.cholesky_banded` and `scipy.linalg.cho_solve_banded`) with precomputed factorization for constant lookback windows $W$.
- **Rationale:** Preserves strictly linear $O(N)$ computational complexity and sub-millisecond per-step causal inference runtime without resorting to slow general dense solvers ($O(N^3)$).

## S4B-D02: Zero Surrogate Retraining Invariance
- **Context:** Testing new attack geometries could tempt retraining plant dynamics for each attack shape.
- **Decision:** The Stage 1 Dilated TCN plant surrogate (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`) remains completely frozen and unchanged.
- **Rationale:** Normal physical dynamics $x_{t+1} = f(x_{\le t}, u_{\le t})$ depend solely on hydraulic mechanics and uncorrupted companion telemetry, completely independent of whatever corruptions an adversary injects into sensor $y$.

## S4B-D03: Multi-Order Hyperparameter Balancing
- **Context:** Setting $\lambda_0, \lambda_1, \lambda_2$ inappropriately could cause over-smoothing or staircasing.
- **Decision:** Standardize default parameterization to:
  - $\lambda_0 = 0.05$ (temporal sparsity)
  - $\lambda_1 = 0.50$ (step discontinuity penalization)
  - $\lambda_2 = 1.00$ (curvature / slope variation penalization)
  Allow user/scenario overrides and evaluate ablation $\lambda_2 = 0$ (pure TV) vs $\lambda_2 > 0$ (trend filtering).
- **Rationale:** Allows the solver to concurrently recover both sharp step transitions and continuous slope drifts.

## S4B-D04: Synthetic Attack Family Benchmark Protocol
- **Context:** To prove generalization, estimators must be evaluated across diverse attack morphologies beyond the Phase 3 step attack.
- **Decision:** Construct a versioned 5-family attack generator spanning steps, ramps, linear drifts, transient pulses, and sinusoids across validation split rows 480–960, benchmarking `PersistenceEstimator`, `StepMAP_TCN` (first-order TV), and `MultiOrderTrendFilter_TCN` (multi-order TV).
- **Rationale:** Rigorously isolates the marginal benefit of second-order filtering on non-step morphologies.
