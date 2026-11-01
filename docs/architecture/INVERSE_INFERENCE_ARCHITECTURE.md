# Architecture Specification: Closed-Loop Temporal Additive Attack Inversion

This document specifies the end-to-end system architecture for real-time additive sensor attack recovery ($y_t = x_t + \delta_t$) validated on the BATADAL benchmark.

---

## 1. Interaction-First System Flowchart

```mermaid
flowchart TD
    subgraph PhysicalPlant["Cyber-Physical System & Attack Model"]
        P["Physical Plant Dynamics (Mass Conservation)"] -->|"Clean State x(t)"| CH_TGT["Target Channel (L_T1)"]
        P -->|"Uncorrupted Subspace z(t)"| COMP["Companion Telemetry (42 Sensors: Flows, Pumps, Pressures)"]
        ADV["Adversary / Threat Model"] -->|"Additive Perturbation delta(t)"| CH_TGT
        CH_TGT -->|"Corrupted Telemetry y(t) = x(t) + delta(t)"| SCADA["SCADA Observation Feed y(t)"]
    end

    subgraph Stage1["Stage 1: Deep Surrogate Dynamics (TACC Vista GH200)"]
        COMP -->|"Companion History z(t-48:t)"| TCN["Dilated Causal TCN (150 ep, receptive field 48h)"]
        TCN -->|"Clean Forward Estimate x_surr(t)"| RESID["Raw Residual r(t) = y(t) - x_surr(t)"]
        TCN -.->|"Calibrated Noise Scale sigma_eps = 0.0418m"| UNCERTAINTY["Uncertainty Engine"]
    end

    subgraph Stage2["Stage 2: Causal Rolling TV-MAP Inversion (ADMM Solver)"]
        SCADA -->|"Observed y(t)"| ROLLING["Sliding Lookback Buffer (W = 48h, past-only)"]
        RESID -->|"Residual r(t)"| ROLLING
        ROLLING --> ADMM["Fast Banded Cholesky ADMM Solver"]
        BOUNDS["Physical Hydraulic Bounds (0.32m <= x <= 4.87m)"] -->|"Box Constraints: y - x_max <= delta <= y - x_min"| ADMM
        PENALTIES["Fused Lasso Regularization: lambda_TV ||D delta||_1 + lambda_L1 ||delta||_1"] -->|"Enforce Step Sparsity"| ADMM
    end

    subgraph OutputEstimation["Reconstruction & Uncertainty Outputs"]
        ADMM -->|"Online Inversion at step t"| DELTA_HAT["Estimated Attack delta_hat(t)"]
        SCADA --> SUBTRACT["Subtraction: x_hat(t) = y(t) - delta_hat(t)"]
        DELTA_HAT --> SUBTRACT
        SUBTRACT --> X_HAT["Recovered Clean State x_hat(t)"]
        UNCERTAINTY --> BANDS["90% Calibrated Interval: x_hat(t) +/- 1.645 sigma_eps"]
        X_HAT --> BANDS
    end

    subgraph FeedbackLoop["Online Operational Feedback Loop"]
        X_HAT -->|"Decontaminated Past History"| REC_BUFFER["Causal Clean State Buffer"]
        REC_BUFFER -.->|"Past History (Zero Future Lookahead)"| SCADA
    end

    classDef plant fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc;
    classDef surrogate fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef solver fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef output fill:#0f172a,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;
    classDef feedback fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;

    class PhysicalPlant,P,CH_TGT,COMP,ADV,SCADA plant;
    class Stage1,TCN,RESID,UNCERTAINTY surrogate;
    class Stage2,ROLLING,ADMM,BOUNDS,PENALTIES solver;
    class OutputEstimation,DELTA_HAT,SUBTRACT,X_HAT,BANDS output;
    class FeedbackLoop,REC_BUFFER feedback;
```

---

## 2. Temporal Execution & Feedback Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Plant as Physical Plant
    participant Adv as Adversary
    participant SCADA as SCADA Stream y(t)
    participant TCN as Stage 1: Dilated TCN
    participant ADMM as Stage 2: Causal Rolling Solver
    participant System as Process Controller / Historian

    Note over Plant,Adv: Hourly Time Step t
    Plant->>SCADA: Transmits compromised reading y(t) = x(t) + delta(t)
    Plant->>TCN: Transmits uncorrupted companion sensors z(t) (flows, pressures)
    
    rect rgb(20, 30, 45)
        Note over TCN: Forward State Estimation (Stage 1)
        TCN->>TCN: Computes causal convolution over companion buffer z(t-48:t)
        TCN->>ADMM: Emits clean prior estimate x_surr(t) & noise floor sigma_eps
    end

    rect rgb(20, 45, 30)
        Note over ADMM: Causal Inversion Optimization (Stage 2)
        SCADA->>ADMM: Supplies observed y(t) into sliding window [t-47 : t]
        ADMM->>ADMM: Solves TV-MAP via cached banded Cholesky factor (rho=1.0)
        ADMM->>ADMM: Applies soft thresholding & hydraulic box bounds
        ADMM->>System: Emits recovered delta_hat(t) & clean x_hat(t)
        ADMM->>System: Emits 90% confidence bounds [x_hat +/- 1.645 sigma_eps]
    end

    rect rgb(45, 35, 20)
        Note over System: Safe Causal Feedback
        System->>Plant: Dispatches control actions based on clean x_hat(t), not corrupted y(t)
    end
```

---

## 3. Structural Module Specifications

| Module | Operational Role | Key Invariant / Guarantee | Verification Metric |
| :--- | :--- | :--- | :--- |
| **Observation Feed (`SCADA`)** | Ingests continuous telemetry $y_t$ | $y_t = x_t + \delta_t$ under target channel $L_{T1}$ | Manifest hash binding |
| **Stage 1: Plant Surrogate (`DilatedTCNSurrogate`)** | Predicts physical plant state from companion subspace $S_{\text{comp}}$ | Causal 48h receptive field; zero target feedback | Validation RMSE $0.0678\,\text{m}$ ($R^2 = 0.9968$), $\sigma_\epsilon = 0.0418\,\text{m}$ |
| **Stage 2: Causal Solver (`CausalStepMAPEstimator`)** | Inverts residual $r_t = y_t - \hat{x}_t$ for $\hat{\delta}_t$ forward in time | Strict past-only causality; sliding window $W=48\,\text{h}$; ADMM Cholesky cache | Execution time: $684\,\mu\text{s}$/sample; 71.77% NRMSE cut vs. Persistence |
| **Undeclared Support Inference** | Infers attack interval $[t_{\text{start}}, t_{\text{end}}]$ autonomously | Joint $\ell_1$ sparsity and 1D Total Variation regularization | Support $F_1 = 0.9000$ (zero oracle knowledge) |
| **Uncertainty Propagation Engine** | Generates calibrated confidence intervals | Posterior variance $\sigma_\epsilon^2$ propagated to bounds | Nominal 90% coverage measured at $94.01\% - 96.01\%$; sharpness $0.1375\,\text{m}$ |
| **Hydraulic Box Projection** | Prevents physically impossible tank levels | $y_t - x_{\max} \le \delta_t \le y_t - x_{\min}$ | Hard violations $\equiv 0$ ($[0.3201, 4.8713]\,\text{m}$) |
