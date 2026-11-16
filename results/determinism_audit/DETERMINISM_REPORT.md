# BATADAL CausalTCN & ADMM Trend Filtering Determinism and Reproducibility Audit

**Audit Date:** 2026-11-16T04:39:23.123236+00:00  
**Status:** **FULL DETERMINISM CONFIRMED (100% BITWISE IDENTICAL)**  
**Hardware & OS Environment:**
- **OS:** Windows-11-10.0.26300-SP0
- **Processor:** AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD
- **Python:** 3.12.13
- **PyTorch:** 2.14.1+cpu (CUDA Available: False, Device: N/A)
- **NumPy:** 2.5.1
- **Execution Device:** cpu

---

## Executive Summary

This formal audit empirical evaluates the reproducibility of the two-stage inverse recovery setup:
1. **Stage 1 Plant Surrogate:** Frozen Dilated Temporal Convolutional Network (`DilatedTCNSurrogate`).
2. **Stage 2 Inverse Estimator:** Causal Multi-Order Trend Filter (`CausalMultiOrderTrendFilterEstimator`) solved via ADMM Banded Cholesky factorization.

Across **5 consecutive evaluation passes** spanning **10 multi-family attack benchmark scenarios** (2 step, 2 ramp, 2 trapezoid, 2 pulse, 2 periodic):
- **Stage 1 Surrogate Inter-Pass Max Absolute Difference:** `0.00e+00`
- **Stage 2 Recovered Delta Vector Max Absolute Difference:** `0.00e+00`
- **Evaluated Scalar Metric Max Inter-Run Difference:** `0.00e+00`
- **Bitwise Numerical Parity:** `STRICT (exact 0.0 difference)`

All scalar performance metrics—including **NRMSE, MAE, Max Error, False Alarm Rate, True Positive Rate, and F1 Score**—are completely invariant across successive invocations.

---

## Multi-Family Benchmark Scenarios

| Scenario ID | Attack Family | Amplitude (train std) | Injected (m) | Duration (h) | Window [Onset, Offset) |
|---|---|---|---|---|---|
| `step_pos_24h` | Step | 1.0 | +1.2211 | 24h | [48, 72) |
| `step_neg_48h` | Step | 1.5 | -1.8317 | 48h | [120, 168) |
| `ramp_pos_24h` | Ramp | 1.0 | +1.2211 | 24h | [180, 204) |
| `ramp_neg_48h` | Ramp | 1.5 | -1.8317 | 48h | [220, 268) |
| `trapezoid_pos_24h` | Trapezoid | 1.0 | +1.2211 | 24h | [280, 304) |
| `trapezoid_neg_48h` | Trapezoid | 1.5 | -1.8317 | 48h | [320, 368) |
| `pulse_pos_12h` | Pulse | 1.0 | +1.2211 | 12h | [380, 392) |
| `pulse_neg_24h` | Pulse | 1.5 | -1.8317 | 24h | [410, 434) |
| `periodic_pos_24h` | Periodic | 1.0 | +1.2211 | 24h | [50, 74) |
| `periodic_neg_48h` | Periodic | 1.5 | -1.8317 | 48h | [200, 248) |

---

## Numerical Parity Metrics Across 5 Passes

The table below reports baseline metrics (Pass 1) and the maximum absolute divergence across all pairwise run combinations $\max_{i < j} |m^{(i)} - m^{(j)}|$:

| Scenario ID | NRMSE | Max Inter-Run $\Delta$NRMSE | MAE (m) | Max Inter-Run $\Delta$MAE | Max Error (m) | Max Inter-Run $\Delta$MaxErr | TPR | FAR |
|---|---|---|---|---|---|---|---|---|
| `step_pos_24h` | 0.326030 | 0.00e+00 | 0.344418 | 0.00e+00 | 1.184111 | 0.00e+00 | 0.9583 | 0.2018 |
| `step_neg_48h` | 0.090135 | 0.00e+00 | 0.088275 | 0.00e+00 | 0.940509 | 0.00e+00 | 1.0000 | 0.0579 |
| `ramp_pos_24h` | 0.155409 | 0.00e+00 | 0.184029 | 0.00e+00 | 0.253021 | 0.00e+00 | 0.8095 | 0.2004 |
| `ramp_neg_48h` | 0.106233 | 0.00e+00 | 0.149490 | 0.00e+00 | 0.376353 | 0.00e+00 | 1.0000 | 0.0526 |
| `trapezoid_pos_24h` | 0.336407 | 0.00e+00 | 0.373278 | 0.00e+00 | 0.669429 | 0.00e+00 | 0.8696 | 0.1707 |
| `trapezoid_neg_48h` | 0.114374 | 0.00e+00 | 0.180735 | 0.00e+00 | 0.431169 | 0.00e+00 | 0.9556 | 0.0460 |
| `pulse_pos_12h` | 0.388686 | 0.00e+00 | 0.307919 | 0.00e+00 | 1.221113 | 0.00e+00 | 0.6667 | 0.1593 |
| `pulse_neg_24h` | 0.190443 | 0.00e+00 | 0.153142 | 0.00e+00 | 1.078205 | 0.00e+00 | 1.0000 | 0.0503 |
| `periodic_pos_24h` | 0.401571 | 0.00e+00 | 0.413930 | 0.00e+00 | 0.893655 | 0.00e+00 | 0.8500 | 0.1848 |
| `periodic_neg_48h` | 0.326853 | 0.00e+00 | 0.502173 | 0.00e+00 | 1.170598 | 0.00e+00 | 0.8500 | 0.0682 |

---

## Formal Statement on Reproducibility

**Question:** Is the inverse recovery pipeline fully deterministic across repeated runs, producing identical numerical outputs given the same checkpoint and test scenarios?

**Formal Finding:**  
**YES.** Under fixed model weights and input telemetry, both the causal Dilated TCN forward inference pass and the ADMM multi-order trend filtering optimization converge to identical floating-point representations. Across all 5 evaluation passes:
1. Every element of the recovered attack trajectory $\hat{\boldsymbol{\delta}}$ matches with an absolute difference $\le 0.00e+00$.
2. Every summary metric (NRMSE, MAE, Max Error, TPR, FAR) matches with an absolute difference $\le 0.00e+00$.
3. No non-deterministic random seeds, stochastic sampling, race conditions, or unseeded CUDA primitives affect the inverse recovery pipeline.

The pipeline satisfies all empirical determinism and reproducibility standards for publication and benchmark verification.
