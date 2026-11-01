# Sprint 4A Decisions

## S4A-D01: Decoupling Statistical Modeling from Model Retraining
- **Context:** Sprint 4A requires characterizing empirical reconstruction errors $e_t = \hat{\delta}_t - \delta_{\text{true}, t}$ and the null clean false-alarm distribution $p(\hat{\delta} \mid \delta=0)$.
- **Decision:** Do NOT retrain or fine-tune the plant surrogate or inverse solvers. Directly utilize the sealed Phase 3 evaluation records (`summary.csv`) and evaluate clean unattacked telemetry on the existing frozen champion TCN checkpoint (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`).
- **Rationale:** Preserves the statistical independence of the benchmark seal and avoids wasting GPU allocation or modifying the proven Phase 3 baseline.

## S4A-D02: Candidate Parametric Distribution Triad
- **Context:** Empirical reconstruction errors in inverse recovery problems often exhibit peaked modes with heavier tails than Gaussian assumptions predict due to $\ell_1$/TV soft-threshold shrinkage.
- **Decision:** Model the error distribution using three explicit parametric families:
  1. Gaussian $\mathcal{N}(\mu, \sigma^2)$ (null hypothesis baseline)
  2. Student-$t$ $t(\nu, \mu, \sigma)$ (robust heavy-tailed model)
  3. Generalized Error Distribution $\text{GED}(\mu, \sigma, \beta)$ (flexible kurtosis/shape family via `scipy.stats.gennorm`)
- **Rationale:** Comparing AIC, BIC, and KS statistics across this triad rigorously tests whether residual reconstruction error requires robust (leptokurtic) modeling or standard Gaussian approximations.

## S4A-D03: Null Clean Distribution Definition
- **Context:** The null distribution $p(\hat{\delta} \mid \delta=0)$ characterizes the false-positive response of the estimators under normal operation.
- **Decision:** Evaluate `StepMAPEstimator` and `CausalStepMAPEstimator` on the authentic unattacked validation block (rows 960–1440 of `data/processed/batadal/val.parquet`) with undeclared support (`attack_interval=None`).
- **Rationale:** Measures the exact empirical false-positive rate and null noise envelope directly on the sealed validation split without synthetic contamination.

## S4A-D04: Operational Detection Threshold Optimization
- **Context:** Downstream ICS intrusion detection systems require a deterministic threshold $\tau_{\text{alarm}}$ to trigger alarms based on $|\hat{\delta}_t|$.
- **Decision:** Construct empirical ROC curves and compute exact threshold cutoffs $\tau^*$ for False Alarm Rates $\alpha \in \{0.001, 0.01, 0.05\}$, validating against nominal conformal bounds ($3\sigma_\epsilon \approx 0.125\,\text{m}$).
- **Rationale:** Bridges inverse state estimation with practical cyber-physical operational alerting.
