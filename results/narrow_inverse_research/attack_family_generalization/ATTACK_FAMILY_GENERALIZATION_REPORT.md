# Sprint 4B Technical Report: Attack Family Generalization via Multi-Order Trend Filtering

**Date:** November 2026  
**Status:** Complete & Sealed  
**Lead Subagent:** `developer`  
**Target Channel:** `L_T1` (Storage Tank 1 Water Level)  
**Plant Surrogate:** Frozen Stage 1 Champion Dilated TCN (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`)  
**Specification Tests:** `tests/spec/test_multi_order_trend_filter.py` (7/7 passing)  
**Full Regression Suite:** 284/284 tests passing (100%)  

---

## 1. Executive Summary

Sprint 4B addresses the limitation of pure first-order Total Variation (TV) filtering, which causes severe **staircasing artifacts** and degradation on non-step attack geometries (ramps, linear drifts, frequency sinusoids). We developed, mathematically locked, and benchmarked the **Composite Multi-Order Trend Filtering MAP Estimator** without modifying or retraining the frozen champion Dilated TCN plant surrogate.

The multi-order formulation incorporates a second-order difference penalty $\|\mathbf{D}^2 \boldsymbol{\delta}\|_1$ alongside zeroth-order sparsity $\|\boldsymbol{\delta}\|_1$ and first-order step discontinuities $\|\mathbf{D}^1 \boldsymbol{\delta}\|_1$, solved via an $O(N)$ pentadiagonal banded Cholesky ADMM algorithm.

### Key Empirical Metrics
Across 50 stratified scenarios (150 evaluations) spanning 5 attack morphologies on BATADAL validation telemetry (rows 480–960):

| Attack Family | Persistence Median NRMSE | StepMAP_TCN (TV, $\lambda_2=0$) | MultiOrder_TCN (TF, $\lambda_2>0$) | Marginal Reduction ($\lambda_2>0$ vs TV) | Target Acceptance Gate | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ramp** | 0.5589 | 0.2520 | **0.1820** | **+27.78%** | $\ge 15.0\%$ | **PASS** |
| **Drift** | 0.5263 | 0.2481 | **0.1801** | **+27.42%** | $\ge 15.0\%$ | **PASS** |
| **Sinusoid** | 0.7285 | 0.4644 | **0.3780** | **+18.60%** | Baseline | **PASS** |
| **Step** | 0.9962 | 0.2505 | **0.2226** | **+11.13%** | Non-inferiority | **PASS** |
| **Pulse** | 0.5109 | 0.2202 | **0.2223** | **-0.97%** | Non-inferiority | **PASS** |
| **Overall** | **0.5877** | **0.2713** | **0.2120** | **+21.85%** | $\ge 15.0\%$ | **PASS** |

Both ramp attacks (+27.78%) and linear drift attacks (+27.42%) significantly exceeded the precommitted $\ge 15\%$ NRMSE reduction gate, with 0 hard physical constraint violations across all scenarios and strict preservation of zero future lookahead invariance.

---

## 2. Mathematical Formulation & Solver Architecture

### 2.1 Optimization Problem
$$\min_{\boldsymbol{\delta}} \frac{1}{2\sigma_\epsilon^2} \|\mathbf{y} - \hat{\mathbf{x}} - \boldsymbol{\delta}\|_2^2 + \lambda_0 \|\boldsymbol{\delta}\|_1 + \lambda_1 \|\mathbf{D}^1 \boldsymbol{\delta}\|_1 + \lambda_2 \|\mathbf{D}^2 \boldsymbol{\delta}\|_1$$
$$\text{subject to} \quad x_{\min} \le \mathbf{y} - \boldsymbol{\delta} \le x_{\max} \iff \mathbf{y} - x_{\max} \le \boldsymbol{\delta} \le \mathbf{y} - x_{\min}$$

Where:
- $\mathbf{y} \in \mathbb{R}^N$: Corrupted telemetry.
- $\hat{\mathbf{x}} \in \mathbb{R}^N$: Causal plant state predictions from frozen champion Dilated TCN (`sigma_epsilon = 0.0418` m).
- $\|\boldsymbol{\delta}\|_1$: Zeroth-order $\ell_1$ norm enforcing null attack outside active intervals.
- $\|\mathbf{D}^1 \boldsymbol{\delta}\|_1$: First-order Total Variation ($(\mathbf{D}^1\boldsymbol{\delta})_t = \delta_{t+1} - \delta_t$) enforcing piecewise-constant transitions.
- $\|\mathbf{D}^2 \boldsymbol{\delta}\|_1$: Second-order discrete differences ($(\mathbf{D}^2\boldsymbol{\delta})_t = \delta_{t+2} - 2\delta_{t+1} + \delta_t$) penalizing slope inflections to enforce piecewise linearity.
- $[x_{\min}, x_{\max}] = [0.3201, 4.8713]\,\text{m}$: Physical tank capacity bounds.

### 2.2 ADMM Numerical Algorithm with Banded Cholesky
We introduce auxiliary splitting variables $\mathbf{z}_0 = \boldsymbol{\delta}$, $\mathbf{z}_1 = \mathbf{D}^1\boldsymbol{\delta}$, $\mathbf{z}_2 = \mathbf{D}^2\boldsymbol{\delta}$ with dual multipliers $\mathbf{u}_0, \mathbf{u}_1, \mathbf{u}_2$:

1. **Primal $\boldsymbol{\delta}$-update**:
   $$\left( \frac{1}{\sigma_\epsilon^2} \mathbf{I} + \rho_0 \mathbf{I} + \rho_1 (\mathbf{D}^1)^\top \mathbf{D}^1 + \rho_2 (\mathbf{D}^2)^\top \mathbf{D}^2 \right) \boldsymbol{\delta}^{(k+1)} = \mathbf{r}^{(k)}$$
   $$\mathbf{r}^{(k)} = \frac{1}{\sigma_\epsilon^2}(\mathbf{y} - \hat{\mathbf{x}}) + \rho_0(\mathbf{z}_0^{(k)} - \mathbf{u}_0^{(k)}) + \rho_1 (\mathbf{D}^1)^\top (\mathbf{z}_1^{(k)} - \mathbf{u}_1^{(k)}) + \rho_2 (\mathbf{D}^2)^\top (\mathbf{z}_2^{(k)} - \mathbf{u}_2^{(k)})$$
   The coefficient matrix $\mathbf{A}$ is symmetric pentadiagonal positive-definite (bandwidth $p=2$). It is factorized once in $O(N)$ time via `scipy.linalg.cholesky_banded` and backsolved in $O(N)$ time via `scipy.linalg.cho_solve_banded`.

2. **Dual Auxiliary Splitting Updates**:
   - $\mathbf{z}_0^{(k+1)} = \text{clip}\left( \mathcal{S}_{\lambda_0 / \rho_0}\left( \boldsymbol{\delta}^{(k+1)} + \mathbf{u}_0^{(k)} \right), \mathbf{y} - x_{\max}, \mathbf{y} - x_{\min} \right)$
   - $\mathbf{z}_1^{(k+1)} = \mathcal{S}_{\lambda_1 / \rho_1}\left( \mathbf{D}^1 \boldsymbol{\delta}^{(k+1)} + \mathbf{u}_1^{(k)} \right)$
   - $\mathbf{z}_2^{(k+1)} = \mathcal{S}_{\lambda_2 / \rho_2}\left( \mathbf{D}^2 \boldsymbol{\delta}^{(k+1)} + \mathbf{u}_2^{(k)} \right)$

3. **Dual Multiplier Updates**:
   - $\mathbf{u}_0^{(k+1)} = \mathbf{u}_0^{(k)} + \boldsymbol{\delta}^{(k+1)} - \mathbf{z}_0^{(k+1)}$
   - $\mathbf{u}_1^{(k+1)} = \mathbf{u}_1^{(k)} + \mathbf{D}^1 \boldsymbol{\delta}^{(k+1)} - \mathbf{z}_1^{(k+1)}$
   - $\mathbf{u}_2^{(k+1)} = \mathbf{u}_2^{(k)} + \mathbf{D}^2 \boldsymbol{\delta}^{(k+1)} - \mathbf{z}_2^{(k+1)}$

### 2.3 Strictly Causal Rolling Architecture
`CausalMultiOrderTrendFilterEstimator` processes streaming telemetry forward in time over a sliding lookback window $W=48\,\text{h}$. For each timestep $t$, the estimator solves the bounded MAP problem over window $[t - W + 1 : t + 1]$, committing only the estimate for current timestep $t$ ($\delta_t = \delta_W[-1]$). A precomputed Cholesky factor cache ensures sub-millisecond per-step inference.

---

## 3. Detailed Benchmark Results

### 3.1 Family-Stratified Performance Metrics
Comprehensive evaluation over 50 scenarios (10 per family) with amplitude ranges $\pm [0.5\sigma, 2.0\sigma]$ and durations $12\,\text{h}$ to $48\,\text{h}$:

| Family | Estimator | Median Delta NRMSE | Mean Delta NRMSE | Median Support F1 | Mean Support F1 | Sign Accuracy | Hard Violations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | Persistence | 0.9962 | 0.9645 | 0.0828 | 0.1096 | 70.0% | 66 |
| | StepMAP_TCN (TV) | 0.2505 | 0.2600 | 0.6880 | 0.6945 | 100.0% | 0 |
| | **MultiOrder_TCN (TF)** | **0.2226** | **0.2420** | **0.7729** | **0.7691** | **100.0%** | **0** |
| **Ramp** | Persistence | 0.5589 | 0.5342 | 0.0723 | 0.0966 | 90.0% | 30 |
| | StepMAP_TCN (TV) | 0.2520 | 0.2720 | 0.5081 | 0.5041 | 100.0% | 0 |
| | **MultiOrder_TCN (TF)** | **0.1820** | **0.2069** | **0.5982** | **0.5949** | **100.0%** | **0** |
| **Drift** | Persistence | 0.5263 | 0.5103 | 0.0723 | 0.0947 | 90.0% | 26 |
| | StepMAP_TCN (TV) | 0.2481 | 0.2638 | 0.4833 | 0.4919 | 100.0% | 0 |
| | **MultiOrder_TCN (TF)** | **0.1801** | **0.2027** | **0.5836** | **0.5898** | **100.0%** | **0** |
| **Pulse** | Persistence | 0.5109 | 0.4828 | 0.0190 | 0.0188 | 60.0% | 8 |
| | StepMAP_TCN (TV) | 0.2202 | 0.2429 | 0.2609 | 0.2305 | 100.0% | 0 |
| | **MultiOrder_TCN (TF)** | **0.2223** | **0.2468** | **0.3246** | **0.2914** | **100.0%** | **0** |
| **Sinusoid**| Persistence | 0.7285 | 0.7539 | 0.0974 | 0.1217 | 30.0% | 37 |
| | StepMAP_TCN (TV) | 0.4644 | 0.4521 | 0.5402 | 0.5166 | 100.0% | 0 |
| | **MultiOrder_TCN (TF)** | **0.3780** | **0.3880** | **0.6295** | **0.5900** | **90.0%** | **0** |

### 3.2 Key Performance Insights
1. **Mitigation of Staircasing Artifacts**:
   On linear drifts and ramps, pure TV forces $\delta_{t+1} - \delta_t = 0$ over intervals, producing unnatural staircase jumps. Second-order trend filtering ($\lambda_2 = 1.00$) penalizes second differences ($\mathbf{D}^2\boldsymbol{\delta} \to 0$), reconstructing smooth linear slopes and reducing reconstruction error by over 27%.
2. **Frequency Spoofing Robustness**:
   On smooth sinusoids, trend filtering reduces median NRMSE from 0.4644 down to 0.3780 (18.60% reduction), tracking continuous oscillation curves without flat-topping.
3. **Step Profile Retention**:
   On step attacks and transient pulses, maintaining $\lambda_1 = 0.50$ prevents over-smoothing of sharp step edges, achieving an improved Support F1 of 0.7729 on steps and statistical parity on 3-hour pulses.
4. **Physical Hydraulic Safety**:
   Across all 150 evaluations, `CausalMultiOrderTrendFilter_TCN` produced **zero hard physical box-constraint violations** ($x_{\min} \le x_{\text{hat}} \le x_{\max}$), whereas naive persistence produced 167 violations.

---

## 4. Acceptance Gate Audit

| Criterion | Target Specification | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| **Locked Spec Tests** | `tests/spec/test_multi_order_trend_filter.py` 100% pass | 7/7 passing (4.11s) | **PASSED** |
| **Ramp Generalization** | $\ge 15.0\%$ NRMSE reduction vs TV | **+27.78%** reduction | **PASSED** |
| **Drift Generalization** | $\ge 15.0\%$ NRMSE reduction vs TV | **+27.42%** reduction | **PASSED** |
| **Clean Negative Control** | Sparsity $P(\hat{\delta}=0 \mid \delta=0) > 95\%$ | **> 98.5%** sparsity | **PASSED** |
| **Causal Invariance** | Strict zero future lookahead invariance | Confirmed to machine precision ($10^{-10}$) | **PASSED** |
| **Physical Box Projection** | 0 box-constraint violations | Exactly 0 violations | **PASSED** |
| **Surrogate Invariance** | Dilated TCN weights 100% frozen | Verified completely unchanged | **PASSED** |
| **Regression Suite** | $\ge 284$ tests passing with 0 regressions | **284/284 passing** (28.84s) | **PASSED** |

---

## 5. Artifact Directory Inventory

The following versioned artifacts have been generated in `results/narrow_inverse_research/attack_family_generalization/`:

1. `summary.json`: Complete aggregate and family-stratified metrics, timings, and configuration metadata.
2. `summary.csv`: Full scenario-level record table (150 evaluations).
3. `plots/family_nrmse_comparison.png`: Grouped comparative bar chart across all 5 attack morphologies.
4. `plots/nrmse_reduction_by_family.png`: Marginal NRMSE improvement percentage chart by morphology.
5. `plots/attack_trajectories_sample.png`: Multi-panel temporal reconstruction traces comparing Ground Truth, Persistence, StepMAP (TV), and MultiOrder (TF).
6. `ATTACK_FAMILY_GENERALIZATION_REPORT.md`: This comprehensive technical milestone report.
