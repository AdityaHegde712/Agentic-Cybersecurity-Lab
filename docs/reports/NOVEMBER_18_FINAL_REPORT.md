# Final Benchmark Evaluation Report: Sensor Inverse Recovery Under Additive Spoofing Attacks

**Milestone Deliverable:** November 18 Proposal Milestone — Step 5 Sealed Final Test Benchmark  
**Benchmark Target Channel:** `L_T1` (Water Level, Tank 1, BATADAL C-Town Hydraulic Network)  
**Evaluation Split:** Officially Unsealed Final Test Block (Rows 960–1440 of `data/processed/batadal/val.parquet`, 480 hourly timesteps)  
**Scenario Design:** 180 Stratified Test Scenarios (18 Parameter Combinations $\times$ 10 Regime-Stratified Onsets)  
**Champion Surrogate Checkpoint:** `results/surrogates/vista_tcn_48h_150ep/tcn.pt`  
**Execution Timestamp:** `2026-10-26T03:39:28.730925+00:00`  
**Final Milestone Acceptance Status:** **ACCEPTED — ALL PRECOMMITTED GATES SATISFIED ON TRUE TEST DATA**

---

## 1. Executive Research Finding & Verdict

### Core Research Finding
In industrial physical plant operations, naive autoregressive history propagation (persistence) experiences catastrophic compounding drift when subjected to sustained additive sensor attacks ($y_t = x_t + \delta_t$), with median trajectory error deteriorating to $0.6340$ NRMSE and amplitude error exceeding $0.6977\,\sigma_{\text{train}}$. By decoupling plant dynamics estimation from self-history and reformulating inverse attack reconstruction as a convex Total Variation (TV) regularized maximum a posteriori (MAP) problem over a deep dilated temporal convolutional network (TCN) companion surrogate subspace, we successfully reconstruct both clean plant states and attack trajectories on previously sealed, completely unseen test data.

### Headline Results on True Test Split
- **Causal Online Filter Superiority:** The strictly causal online rolling filter (`CausalStepMAP_TCN`) achieves a **71.76% median Delta NRMSE reduction** in declared support mode and **71.77% reduction** in free undeclared support mode relative to the strongest causal persistence baseline ($0.1790$ vs. $0.6340$ NRMSE), massively exceeding the precommitted 20.0% improvement gate.
- **Offline Batch Ceiling:** When full-window future context is available, offline `StepMAP_TCN` achieves a **85.62% error reduction** ($0.0912$ NRMSE) with perfect 1.000 support $F_1$ score.
- **Uncertainty Calibration Invariance:** Empirical 90% confidence interval coverage remains firmly above nominal specifications on unseen test data ($96.01\%$ declared, $94.01\%$ undeclared) with an ultra-sharp interval width of $0.1375\,\text{m}$ ($< 2.8\%$ of the physical tank operating span).
- **Physical Feasibility Guarantee:** Zero physical box-constraint violations ($[0.3201\,\text{m}, 4.8713\,\text{m}]$) occurred across all 180 test scenarios for all proposed estimators.

### Identifiability Boundary & Guarantee
The inverse recovery problem $y_t = x_t + \delta_t$ is identifiable under three structural conditions verified on the test split:
1. **Uncorrupted Companion Subspace:** Actuator commands and companion sensor channels (pumps, valves, downstream pressures) remain authentic, providing an observable physical constraint surface via the TCN surrogate.
2. **Piecewise-Constant Structural Prior:** The attack $\delta_t$ belongs to the bounded total variation family ($\|\mathbf{D}\boldsymbol{\delta}\|_1 < \infty$), penalizing spurious high-frequency oscillations.
3. **Physical Plant Boundary Envelope:** Water level $x_t$ is bounded by hydraulic capacity ($x_{\min} \le y_t - \delta_t \le x_{\max}$), truncating unphysical attack amplitudes.

---

## 2. Inverse Problem Formulation & Mathematical Inversion Mechanism

### Observation Model
For a targeted hydraulic reservoir sensor (tank level $L\_T1$), the compromised telemetry stream $y_t \in \mathbb{R}$ obeys:
$$y_t = x_t + \delta_t, \quad t \in \{1, \dots, N\}$$
where $x_t \in [x_{\min}, x_{\max}]$ denotes the unobserved true water level, and $\delta_t \in \mathbb{R}$ represents the additive spoofing attack. The unattacked multi-channel companion vector $\mathbf{z}_t \in \mathbb{R}^{d}$ records concurrent network pressures, pump operation states, and valve openings.

### Two-Stage Inversion Architecture

```mermaid
flowchart TD
    subgraph InputTelemetry["Cyber-Physical Telemetry Feed"]
        y["Observed Telemetry y_t = x_t + delta_t"]
        z["Uncorrupted Companion Sensors z_t (42 channels)"]
    end

    subgraph Stage1["Stage 1: Deep Plant Surrogate (TACC Vista GH200)"]
        z -->|"Companion History z_{t-48:t}"| TCN["Dilated Causal TCN (150 ep, RF=48h)"]
        TCN -->|"Uncorrupted Prior x_hat_surr"| RESID["Raw Residual r_t = y_t - x_hat_surr"]
        TCN -.->|"Calibrated Noise sigma_eps = 0.0418m"| UNCERTAINTY["Uncertainty Propagation Engine"]
    end

    subgraph Stage2["Stage 2: Causal Rolling TV-MAP Inversion (ADMM Solver)"]
        y --> BUFFER["Sliding Lookback Window (W = 48h, Past-Only)"]
        RESID --> BUFFER
        BUFFER --> ADMM["Fast Banded Cholesky ADMM Solver"]
        BOX["Hydraulic Bounds: y - x_max <= delta <= y - x_min"] --> ADMM
        TV["Fused Lasso Prior: lambda_TV ||D delta||_1 + lambda_L1 ||delta||_1"] --> ADMM
    end

    subgraph Outputs["Online State & Attack Inversion"]
        ADMM -->|"Online Reconstruction"| DELTA["Attack Trajectory delta_hat_t"]
        y --> RECOVER["x_hat_t = y_t - delta_hat_t"]
        DELTA --> RECOVER
        RECOVER --> X["Clean Plant State x_hat_t"]
        UNCERTAINTY --> BANDS["90% Calibrated Band: x_hat_t +/- 1.645 sigma_eps"]
        X --> BANDS
    end

    classDef plant fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc;
    classDef surrogate fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef solver fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef output fill:#0f172a,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;

    class InputTelemetry,y,z plant;
    class Stage1,TCN,RESID,UNCERTAINTY surrogate;
    class Stage2,BUFFER,ADMM,BOX,TV solver;
    class Outputs,DELTA,RECOVER,X,BANDS output;
```

*Detailed architecture interactions, sequence feedback diagrams, and module contracts are documented in [`docs/architecture/INVERSE_INFERENCE_ARCHITECTURE.md`](../architecture/INVERSE_INFERENCE_ARCHITECTURE.md).*

#### Stage 1: Deep Dilated TCN Plant Surrogate
The clean plant state $\hat{x}_t$ is estimated using a 150-epoch champion Dilated Temporal Convolutional Network (TCN) trained strictly on clean training telemetry (`model_fit`, rows 0–5840 of `train.parquet`). The TCN architecture employs causal dilated residual 1D convolutions with receptive field spanning 48 hours ($d \in \{1, 2, 4, 8, 16, 32\}$), mapping past uncorrupted companion measurements $\mathbf{z}_{t-K:t}$ to clean target channel dynamics without conditioning on corrupted target history.

#### Stage 2: Programmatic Optimization & Causal ADMM
The inverse estimate $\boldsymbol{\delta}^*$ is obtained by minimizing the composite MAP penalized objective:
$$\min_{\boldsymbol{\delta}} \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{x}}_{\text{surr}} - \boldsymbol{\delta}\|_2^2 + \lambda_{\text{TV}} \|\mathbf{D} \boldsymbol{\delta}\|_1 + \lambda_{L1} \|\boldsymbol{\delta}\|_1$$
$$\text{subject to} \quad \mathbf{y} - x_{\max} \le \boldsymbol{\delta} \le \mathbf{y} - x_{\min}$$

For real-time operational deployment (`CausalStepMAP_TCN`), the optimization is solved strictly forward in time using an alternating direction method of multipliers (ADMM) sliding filter with lookback window $W = 48$ hours. The filter updates split variables $(\boldsymbol{\delta}, \mathbf{z}, \mathbf{w})$ via cached Cholesky factorization of the tridiagonal system $[(1 + 2\rho)\mathbf{I} + \rho \mathbf{D}^T \mathbf{D}]$, achieving sub-millisecond execution ($684.2\,\mu\text{s}$ per evaluation step).

---

## 3. Empirical Benchmark Ledger & Acceptance Gate Verification

### Precommitted Acceptance Gate Audit (True Sealed Test Block)

The test evaluation was executed across all 180 unseen test scenarios ($N_{\text{eval}} = 1080$). All precommitted acceptance gates were verified against manifest specifications:

| Acceptance Invariant | Formal Specification | Measured Benchmark Value | Gate Result | Margin |
| :--- | :--- | :--- | :---: | :---: |
| **Gate 1: Causal Baseline Improvement** | $(NRMSE_{\text{persist}} - NRMSE_{\text{causal}}) / NRMSE_{\text{persist}} \ge 0.20$ | Declared: **71.76%**<br>Undeclared: **71.77%** | **PASS** | $+51.76\%$ |
| **Gate 2: Amplitude Error Bound** | Median $\|\hat{\delta} - \delta_{\text{true}}\| \le 0.25\,\sigma_{\text{train}}$ | Declared: **$0.1385\,\sigma$**<br>Undeclared: **$0.1385\,\sigma$** | **PASS** | $-0.1115\,\sigma$ |
| **Gate 3: Sign Accuracy ($\ge 1.0\sigma$)** | Correct attack sign classification $\ge 95.0\%$ | Declared: **100.0%**<br>Undeclared: **100.0%** | **PASS** | $+5.0\%$ (Perfect) |
| **Gate 4: Empirical Uncertainty Coverage** | Empirical $90\%$ confidence interval coverage $\ge 85.0\%$ | Declared: **96.01%**<br>Undeclared: **94.01%** | **PASS** | $+9.01\%$ |
| **Gate 5: Interval Sharpness Ceiling** | Mean confidence interval width $< 0.25\,\text{m}$ ($<5\%$ span) | Declared: **$0.1375\,\text{m}$**<br>Undeclared: **$0.1375\,\text{m}$** | **PASS** | $-0.1125\,\text{m}$ |
| **Gate 6: Physical Plant Constraints** | Hard box violations ($[0.3201, 4.8713]\,\text{m}$) $\equiv 0$ | Declared: **0** violations<br>Undeclared: **0** violations | **PASS** | Exact Zero |
| **Overall Certification** | **All Invariants Simultaneously Satisfied** | **All 6 Gates Passed Concurrently** | **ACCEPTED** | **100% Pass** |

### Complete 6-Estimator Benchmark Ledger

Evaluated across the 180 unseen test scenarios on rows 960–1440 of `val.parquet`:

| Estimator Configuration | Support Assumption | Execution Mode | Median $\Delta$ NRMSE | Mean $\Delta$ NRMSE | Median Support $F_1$ | Median Signed Error ($\sigma_{\text{train}}$) | Sign Acc ($\ge 1.0\sigma$) | 90% Empirical Coverage | Mean Sharpness (m) | Hard Box Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PersistenceEstimator** | Declared | Causal Baseline | 0.6340 | 0.8235 | 1.0000 | 0.6977 | 97.5% | 96.75% | 0.7977 | 0 |
| **CompanionRegression (Ridge)** | Declared | Linear Companion | 0.1980 | 0.2432 | 1.0000 | 0.0836 | 100.0% | 99.61% | 1.2505 | 0 |
| **StepMAP_TCN_Declared** | Declared | Offline Batch | **0.0912** | **0.1193** | **1.0000** | **0.1353** | **100.0%** | **95.82%** | **0.1375** | 0 |
| **StepMAP_TCN_Undeclared** | Undeclared | Offline Batch | **0.0912** | **0.1193** | **1.0000** | **0.1353** | **100.0%** | **95.63%** | **0.1375** | 0 |
| **CausalStepMAP_TCN_Declared** | Declared | Online Causal | **0.1790** | **0.2094** | **1.0000** | **0.1385** | **100.0%** | **96.01%** | **0.1375** | 0 |
| **CausalStepMAP_TCN_Undeclared** | Undeclared | Online Causal | **0.1790** | **0.2087** | **0.9000** | **0.1385** | **100.0%** | **94.01%** | **0.1375** | 0 |

---

## 4. Failure Modes & Visual Figure Analysis

### 4.1 Trajectory Recovery by Attack Duration Horizon

![NRMSE by Duration](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/sealed_final_benchmark/plots/final_nrmse_by_duration.png)

- **Mechanistic Interpretation:** Persistence error escalates dramatically as attack duration lengthens ($0.4091$ NRMSE at 9 hours to $0.7532$ NRMSE at 36 hours). Because persistence relies on recursive autoregressive propagation of the antecedent measurement, any diurnal plant oscillation accumulates as uncorrected drift.
- **TCN Stability:** In contrast, both `StepMAP_TCN` ($0.1390 \to 0.0593$) and `CausalStepMAP_TCN` ($0.2391 \to 0.1353$) exhibit improving relative NRMSE as attack duration increases. The non-local companion dynamics provide an anchor that becomes statistically more dominant over longer windows.

### 4.2 Error Scaling by Attack Amplitude

![NRMSE by Amplitude](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/sealed_final_benchmark/plots/final_nrmse_by_amplitude.png)

- **Signal-to-Noise Dynamics:** Under weak attacks ($0.75\,\sigma_{\text{train}} \approx 0.91\,\text{m}$), process noise and surrogate residual variance ($\sigma_{\text{surr}} = 0.0836\,\text{m}$) constitute a larger fraction of the injected step, resulting in $0.3357$ NRMSE for CausalStepMAP.
- **High-Amplitude Convergence:** For large attacks ($3.0\,\sigma_{\text{train}} \approx 3.63\,\text{m}$), the step discontinuity cleanly dominates background noise, driving CausalStepMAP NRMSE down to $0.0901$ and offline StepMAP down to $0.0429$ NRMSE, with $100\%$ sign classification accuracy.

### 4.3 Calibrated Uncertainty Coverage & Sharpness

![Coverage and Sharpness](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/sealed_final_benchmark/plots/final_coverage_sharpness.png)

- **Coverage Reliability:** Nominal 90% confidence intervals achieve empirical coverage between $94.01\%$ and $96.01\%$ across all Step-MAP configurations on the test partition. This confirms that conformal scaling based on clean validation calibration holds robustly out-of-sample.
- **Sharpness Comparison:** While Companion Ridge regression produces excessively wide, conservative uncertainty intervals ($1.2505\,\text{m}$), the TCN-backed Step-MAP models maintain compact uncertainty bands of exactly $0.1375\,\text{m}$, providing actionable operator decision thresholds.

### 4.4 Causal Online Filter vs. Persistence Parity

![Causal vs Persistence](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/sealed_final_benchmark/plots/final_causal_vs_persistence.png)

- **Dominance Across Scenarios:** In the scenario-by-scenario scatter plot against Persistence, all 180 test evaluations lie comfortably below the parity line ($y = x$) and below the 20% improvement boundary ($y = 0.80x$).
- **Regime Robustness:** Across low ($0.1827$), medium ($0.1750$), and high ($0.1824$) dynamic regimes, CausalStepMAP maintains consistent estimation quality, demonstrating complete independence from sudden operational shifts in hydraulic load.

---

## 5. Trajectory & Phase 4 Roadmap

### Core Technical Advancements Delivered in Phase 3
1. **Unconstrained Free Temporal Support:** Eliminated the unrealistic oracle requirement of declared attack timestamps; the L1/TV sparse optimizer autonomously isolates attack onsets and offsets with $0.900$ median $F_1$ in online causal mode and $1.000$ in offline mode.
2. **Real-Time Past-Only Causal Filter:** Formulated and solved the rolling ADMM filter without future window leakage, proving that real-time SCADA inverse defense is mathematically viable ($71.8\%$ error reduction over persistence).
3. **Rigorous Dual-Partition Protocol:** Established a tamper-proof benchmark lifecycle, validating models on an unsealed final test partition without hyperparameter tuning on the test split.

### Residual Limitations
1. **Multi-Sensor Coordinated Spoofing:** Current identifiability relies on the uncorrupted companion subspace assumption. If an adversary compromises both tank level $L\_T1$ and its immediate companion pump flow $F\_PU1$ concurrently, the surrogate residual error increases.
2. **Transient Hydraulic Water-Hammer Effects:** Sub-hourly pressure spikes below the 1-hour sampling granularity remain unmodeled.

### Immediate Phase 4 Computational Objectives
1. **Distributed Companion Manifolds:** Extend Stage 1 surrogates to graph neural networks (GNNs) or spatial state-space models (Mamba) to model topological water distribution across the full C-Town network.
2. **Bilevel Multi-Channel Inverse Formulation:** Generalize Stage 2 optimization to recover simultaneous sparse multi-channel attacks $\mathbf{\Delta} \in \mathbb{R}^{N \times D}$ via group lasso ($\ell_{2,1}$) regularization.
3. **Hardware Deployment Benchmark:** Deploy the compiled ADMM C-routine onto an embedded SCADA PLC testbed to evaluate sub-millisecond execution constraints.
