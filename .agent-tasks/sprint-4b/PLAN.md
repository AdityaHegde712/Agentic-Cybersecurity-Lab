# Sprint 4B: Attack Family Generalization via Multi-Order Trend Filtering

## 1. System Architecture & Mathematical Formulation

Sprint 4B generalizes additive sensor attack recovery from piecewise-constant step attacks to broad attack geometries (ramps, linear drifts, transient pulses, and smooth sinusoids) without retraining or modifying the frozen champion Dilated TCN plant surrogate.

### 1.1 Mathematical Objective
We formulate the **Composite Multi-Order Trend Filtering MAP Estimator**:
$$\min_{\boldsymbol{\delta}} \frac{1}{2\sigma_\epsilon^2} \| \mathbf{y} - \hat{\mathbf{x}} - \boldsymbol{\delta} \|_2^2 + \lambda_0 \|\boldsymbol{\delta}\|_1 + \lambda_1 \|\mathbf{D}^1 \boldsymbol{\delta}\|_1 + \lambda_2 \|\mathbf{D}^2 \boldsymbol{\delta}\|_1$$
$$\text{subject to} \quad x_{\min} \le \mathbf{y} - \boldsymbol{\delta} \le x_{\max}$$

Where:
- $\mathbf{y} \in \mathbb{R}^N$: Corrupted SCADA telemetry.
- $\hat{\mathbf{x}} \in \mathbb{R}^N$: Causal plant state predictions from frozen champion Dilated TCN surrogate (`sigma_epsilon = 0.0418` m).
- $\|\boldsymbol{\delta}\|_1$: Zeroth-order $\ell_1$ norm enforcing temporal sparsity (null unattacked periods $\delta_t = 0$).
- $\|\mathbf{D}^1 \boldsymbol{\delta}\|_1$: First-order Total Variation ($(\mathbf{D}^1\boldsymbol{\delta})_t = \delta_t - \delta_{t-1}$) inducing piecewise-constant step profiles.
- $\|\mathbf{D}^2 \boldsymbol{\delta}\|_1$: Second-order difference operator ($(\mathbf{D}^2\boldsymbol{\delta})_t = \delta_t - 2\delta_{t-1} + \delta_{t-2}$) inducing piecewise-linear profiles with sparse slope inflections (ramps, linear drifts).
- $[x_{\min}, x_{\max}]$: Physical hydraulic tank bounds ($[0.264, 4.908]\,\text{m}$ for `L_T1`).

### 1.2 ADMM Numerical Algorithm with Banded Cholesky
We introduce three auxiliary split variables $\mathbf{z}_0 = \boldsymbol{\delta}$, $\mathbf{z}_1 = \mathbf{D}^1\boldsymbol{\delta}$, $\mathbf{z}_2 = \mathbf{D}^2\boldsymbol{\delta}$ with dual variables $\mathbf{u}_0, \mathbf{u}_1, \mathbf{u}_2$:

1. **Primal $\boldsymbol{\delta}$-update**:
   $$\left( \frac{1}{\sigma_\epsilon^2} \mathbf{I} + \rho_0 \mathbf{I} + \rho_1 (\mathbf{D}^1)^\top \mathbf{D}^1 + \rho_2 (\mathbf{D}^2)^\top \mathbf{D}^2 \right) \boldsymbol{\delta}^{(k+1)} = \mathbf{r}^{(k)}$$
   $$\mathbf{r}^{(k)} = \frac{1}{\sigma_\epsilon^2}(\mathbf{y} - \hat{\mathbf{x}}) + \rho_0(\mathbf{z}_0^{(k)} - \mathbf{u}_0^{(k)}) + \rho_1 (\mathbf{D}^1)^\top (\mathbf{z}_1^{(k)} - \mathbf{u}_1^{(k)}) + \rho_2 (\mathbf{D}^2)^\top (\mathbf{z}_2^{(k)} - \mathbf{u}_2^{(k)})$$
   The coefficient matrix $\mathbf{A}$ is symmetric pentadiagonal positive-definite (bandwidth $p=2$). It is factorized once in $O(N)$ time via `scipy.linalg.cholesky_banded` and backsolved in $O(N)$ time via `scipy.linalg.cho_solve_banded`.

2. **Dual Auxiliary Splitting Updates**:
   - $\mathbf{z}_0^{(k+1)} = \text{clip}\left( \mathcal{S}_{\lambda_0 / \rho_0}\left( \boldsymbol{\delta}^{(k+1)} + \mathbf{u}_0^{(k)} \right), \mathbf{y} - x_{\max}, \mathbf{y} - x_{\min} \right)$
   - $\mathbf{z}_1^{(k+1)} = \mathcal{S}_{\lambda_1 / \rho_1}\left( \mathbf{D}^1 \boldsymbol{\delta}^{(k+1)} + \mathbf{u}_1^{(k)} \right)$
   - $\mathbf{z}_2^{(k+1)} = \mathcal{S}_{\lambda_2 / \rho_2}\left( \mathbf{D}^2 \boldsymbol{\delta}^{(k+1)} + \mathbf{u}_2^{(k)} \right)$
   where $\mathcal{S}_\gamma(v) = \text{sign}(v)\max(|v| - \gamma, 0)$.

3. **Dual Multiplier Updates**:
   - $\mathbf{u}_0^{(k+1)} = \mathbf{u}_0^{(k)} + \boldsymbol{\delta}^{(k+1)} - \mathbf{z}_0^{(k+1)}$
   - $\mathbf{u}_1^{(k+1)} = \mathbf{u}_1^{(k)} + \mathbf{D}^1 \boldsymbol{\delta}^{(k+1)} - \mathbf{z}_1^{(k+1)}$
   - $\mathbf{u}_2^{(k+1)} = \mathbf{u}_2^{(k)} + \mathbf{D}^2 \boldsymbol{\delta}^{(k+1)} - \mathbf{z}_2^{(k+1)}$

### 1.3 Online Causal Rolling Architecture
`CausalMultiOrderTrendFilterEstimator` processes streaming telemetry forward in time over a sliding lookback window $W=48\,\text{h}$, conditioning strictly on reconstructed history $\hat{x}_{<t}$, ensuring strict zero future lookahead invariance.

## 2. Attack Families for Generalization Bake-Off
We benchmark across 5 distinct synthetic attack families on validation telemetry (`data/processed/batadal/val.parquet` rows 480–960):
1. **Step Attacks**: Instantaneous step $A \cdot \mathbf{1}_{t \in [\text{onset}, \text{offset}]}$.
2. **Ramp Attacks**: Linearly increasing injection $\delta_t = A \cdot \frac{t - \text{onset}}{\text{offset} - \text{onset}}$.
3. **Linear Drifts**: Sustained unbounded slope $\delta_t = c \cdot (t - \text{onset})$.
4. **Transient Pulses**: Short boxcar pulses ($\tau = 3\,\text{h}$).
5. **Sinusoids**: Frequency spoofing $\delta_t = A \sin(2\pi (t - \text{onset}) / T_{\text{period}})$.

## 3. Strict Exit Criteria
1. Locked specification tests pass 100% in `tests/spec/test_multi_order_trend_filter.py`.
2. Second-order term ($\lambda_2 > 0$) demonstrates statistically significant NRMSE improvement over pure first-order TV ($\lambda_2 = 0$) on ramp and drift attack families ($\ge 15\%$ error reduction).
3. Zero false alarms on clean unattacked series ($P(\hat{\delta}=0 \mid \delta=0) > 95\%$).
4. Zero hard physical constraint violations across all scenarios.
5. Strict zero future lookahead invariance verified on causal filter.
6. Zero regressions across existing 277 test suite items (total tests $\ge 285$).
