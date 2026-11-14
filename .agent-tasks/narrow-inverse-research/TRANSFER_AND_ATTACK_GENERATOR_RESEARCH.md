# Research Report: SWaT & WADI Schema Profiling & Synthetic Attack Generator Architecture

**Created:** 2026-11-11  
**Lead Agent:** Primary Architectural Session  
**Scope:** Dataset Schema Ingestion for SWaT & WADI, and Parametric/Adversarial Attack Scenario Generator Design.

---

## 1. Executive Summary

This research specification establishes:
1. **Cross-Dataset Schema Ingestion Blueprints** for SWaT (Secure Water Treatment, 51 channels) and WADI (Water Distribution, 127 channels).
2. **Channel Role Stratification** separating physical target states ($x_1$), continuous companion manifolds ($\mathbf{x}_{\mathcal{C}}$), and discrete actuator commands ($\mathbf{u}$).
3. **High-Throughput Parametric Attack Generator Architecture** capable of generating $1,000+$ diverse synthetic and stealthy adversarial cyber-physical attack trajectories.

---

## 2. Dataset Schema Profiling: SWaT & WADI

### 2.1 SWaT (Secure Water Treatment Testbed)
- **Modality**: 1 Hz continuous physical telemetry logging across 6 sequential water purification stages:
  - **Stage P1 (Raw Water Storage)**: Target level `LIT101` ($[0.25\,\text{m}, 1.20\,\text{m}]$), flow `FIT101`, pumps `P101`/`P102`, valve `MV101`.
  - **Stage P2 (Chemical Dosing)**: Water quality analyzers `AIT201`–`AIT203` (pH, conductivity, ORP), dosing pumps `P201`–`P206`.
  - **Stage P3 (Ultrafiltration UF)**: UF feed flow `FIT301`, differential pressure `DPSH301`, UF feed pumps `P301`/`P302`.
  - **Stage P4 (Dechlorination)**: Analyzers `AIT401`/`AIT402`, flow `FIT401`, bisulfite dosing pumps `P401`–`P404`.
  - **Stage P5 (Reverse Osmosis RO)**: High pressure transmitters `PIT501`–`PIT503`, permeate flow `FIT501`–`FIT504`, RO booster pump `P501`.
  - **Stage P6 (Permeate Backwash)**: Permeate storage tank level `LIT601`, backwash pump `P601`/`P602`.
- **Target Inversion Candidate**: `LIT101` (Stage 1 Raw Water Buffer Tank) or `LIT601` (Stage 6 Treated Water Buffer Tank).
- **Companion Subspace $\mathbf{x}_{\mathcal{C}}$**: Direct upstream inflow `FIT101`, cross-stage transfer flow `FIT201`, pump status `P101`, and chemical dosing pressure heads.

### 2.2 WADI (Water Distribution Testbed)
- **Modality**: 1 Hz telemetry across 127 physical channels spanning 3 municipal supply grids:
  - **Grid 1 (Primary Intake & Elevated Reservoir)**: Primary tank levels `1_LT_001_PV`, `1_LT_002_PV`, supply flow `1_FIT_001_PV`, intake motorized valves `1_MV_001`–`1_MV_004`, booster pumps `1_P_001`–`1_P_006`.
  - **Grid 2 (Consumer Distribution & Pressure Zones)**: Differential pressure transmitters `2_DPIT_001_PV`, line pressure `2_PIT_001_PV`–`2_PIT_003_PV`, consumer branch flow controllers `2_FIC_101_PV`–`2_FIC_601_PV`.
  - **Grid 3 (Recirculation & Return)**: Return tank level `3_LT_001_PV`, flow `3_FIT_001_PV`, drain valves.
- **Target Inversion Candidate**: `1_LT_001_PV` (Primary Elevated Storage Tank Level).
- **Companion Subspace $\mathbf{x}_{\mathcal{C}}$**: 28 downstream flow sensors (`*_FIT_*`), 15 pressure transmitters (`*_PIT_*`, `*_DPIT_*`), and binary actuation statuses (`*_MV_*`, `*_P_*`).

---

## 3. High-Throughput Attack Scenario Generator Architecture

To rigorously evaluate neural surrogate resilience and inverse estimators across thousands of attack configurations, the generator produces parameterized trajectories across six core morphology families:

$$\delta_t(\boldsymbol{\theta}) = A \cdot \psi\left(\frac{t - t_{\mathrm{on}}}{T}\right) \cdot \mathbf{1}_{[t_{\mathrm{on}}, t_{\mathrm{off}}]} + \eta_t$$

### 3.1 Attack Family Parameterizations
1. **Piecewise Constant Step Attacks**:
   - $\psi(u) = 1$.
   - Amplitude: $A \in \pm [0.25\sigma, 4.0\sigma]$. Duration: $T \in [3\,\text{h}, 72\,\text{h}]$.
2. **Linear Ramps & Dynamic Drifts**:
   - $\psi(u) = u$ (finite ramp) or $\psi(u) = u \cdot \mathbf{1}_{t \ge t_{\mathrm{on}}}$ (continuous accumulation).
   - Slope rate: $\alpha = A / T \in [0.01\,\sigma/\text{h}, 0.50\,\sigma/\text{h}]$.
3. **Transient Pulses & Burst Spikes**:
   - $\psi(u) = 1$ with compact support duration $\tau \le 3\,\text{h}$.
4. **Harmonic & Resonance Sinusoids**:
   - $\psi(u) = \sin(2\pi \cdot f \cdot u + \phi)$, $f \in [1/24\,\text{h}^{-1}, 1/2\,\text{h}^{-1}]$.
   - Targets control-loop natural frequencies to induce actuator hunting.
5. **Stochastic Brownian Drifts & Composite Profiles**:
   - Compound waveforms: $\delta_t = \delta_t^{\text{step}} + \delta_t^{\text{ramp}} + W_t$, where $W_t = \sum_{k=1}^t \epsilon_k$ is a random walk.
6. **Stealthy Optimization-Based Adversarial Perturbations (PGD / FGSM)**:
   - Formulated by solving an adversarial maximization problem:
     $$\max_{\boldsymbol{\delta}} \|\hat{x}(\mathbf{y}) - x\|_2^2 \quad \text{s.t.} \quad |\delta_t| \le \gamma \cdot \sigma_\epsilon, \quad \|\mathbf{D}^1 \boldsymbol{\delta}\|_\infty \le \Delta_{\max}$$
   - Perturbations stay within the nominal confidence interval ($\le 3\sigma_\epsilon$) to evade standard threshold intrusion detectors while systematically misbiasing automated dispatch.

---

## 4. Implementation & Validation Roadmap

1. **Phase 4A**: Ingestion script `scripts/prepare_swat_wadi_telemetry.py` extracting clean and attacked segments into standardized Parquet partitions (`val.parquet`, `test.parquet`).
2. **Phase 4B**: High-throughput generator `src/attacks/massive_generator.py` producing deterministic scenario batches with verifiable random seeds.
3. **Phase 4C**: GPU-accelerated benchmark runner on Vista executing parallel batch evaluation across thousands of generated scenarios.
