::: center
**Inferring Latent Attacker Distributions from Time-Series Observations**\
NAIRR Pilot Start-Up Final Update \quad\textbar\quad November 18, 2026 \quad\textbar\quad Request ID: `NAIRR260371`\

--------------------------------------- ----------------------------------
Aditya Hegde (Graduate Student)         Saptarshi Sengupta (PI)
*Department of Computer Engineering*    *Department of Computer Science*
*San José State University*             *San José State University*
--------------------------------------- ----------------------------------
:::

### Problem Formulation & Causal Filter Failure
In industrial Cyber-Physical Systems (CPS) [1], adversarial sensor manipulation injects additive perturbations at discrete timesteps $t \in \{1, \dots, T\}$ according to $y_t = x_t + \delta_t$, where $x_t \in \mathbb{R}$ is the unobserved physical state and $\delta_t \in \mathbb{R}$ is the attack perturbation ($\delta_t = 0$ during clean operations). Disentangling true state from attack perturbation is non-identifiable from telemetry $y$ alone without prior structure. Standard causal estimators (Persistence, Seasonal Naive) fail under sustained attacks because corrupted measurements contaminate their autoregressive states, inducing severe divergence (Persistence median NRMSE $0.8616$; Seasonal Naive $1.4802$).

### Decoupled Inversion via Multi-Order Trend Filtering
To restore identifiability without ground truth, we decouple the target channel from its corrupted trajectory using uncompromised companion channels $z_t \in \mathbb{R}^d$ (flows, valve positions, adjacent pressures). A causal surrogate [5] predicts clean physical state $\hat{x}_{\text{surr}, t} = f_\theta(z_{t-W:t-1})$ with empirical surrogate residual variance $\sigma_\epsilon^2 = \mathbb{E}[(x_t - \hat{x}_{\text{surr}, t})^2]$, where $\epsilon_t = x_t - \hat{x}_{\text{surr}, t}$ represents nominal surrogate tracking error. To recover non-canonical, continuous attack profiles (steps, ramps, physical drift), we formulate a box-constrained Multi-Order Trend Filter [2,3] solved via the Alternating Direction Method of Multipliers (ADMM) [4]:
$$\begin{equation}
\min_{\delta \in \mathbb{R}^T} \;\; \frac{1}{2\sigma_\epsilon^2} \| y - \hat{x}_{\text{surr}} - \delta \|_2^2 + \lambda_0 \|\delta\|_1 + \lambda_1 \|\mathbf{D}^{(1)}\delta\|_1 + \lambda_2 \|\mathbf{D}^{(2)}\delta\|_1 \quad \text{s.t.} \quad y - \delta \in [x_{\min}, x_{\max}]
\end{equation}$$
The $\ell_1$ penalty enforces temporal attack sparsity; the first difference $\mathbf{D}^{(1)}$ captures abrupt step and ramp transitions; and the second difference $\mathbf{D}^{(2)}$ tracks smooth physical drift, while physical box constraints guarantee zero envelope violations.

### Multi-Testbed Empirical Benchmarks
Scaling beyond our initial single-tank BATADAL prototype, we evaluated the framework across three critical infrastructure testbeds: BATADAL [7] (43 channels), SWaT [8] (51 channels), and WADI [9] (122 channels). Multi-channel CausalTCN surrogates (trained on TACC Vista and cloud instances; 100k parameters, OneCycleLR [6]) compressed residual standard deviations to $0.0867\,\text{m}$ (BATADAL `L_T1`), $9.5677\,\text{mm}$ (SWaT `LIT101`), and $5.7187\,\text{mm}$ (WADI `1_LT_001_PV`). We benchmarked 5 canonical attack families (step, ramp, drift, pulse, periodic across 50 scenarios per dataset) and evaluated high-throughput stress sweeps across 3,000 randomized scenarios (1,000 per dataset).

Table 1: Multi-testbed attack recovery benchmarks across 5 canonical attack families (50 scenarios per testbed).

| Testbed | Telemetry Split | Target Sensor | Median NRMSE | Mean NRMSE | Sign Acc. | Box Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BATADAL Water Dist.** | 480 steps | `L_T1` (Tank 1) | **0.0418** (4.18%) | 0.0512 | 98.0% | **0 (0.0%)** |
| **SWaT Water Treatment** | 5,832 steps | `LIT101` (Raw Water) | **0.0381** (3.81%) | 0.0493 | 100.0% | **0 (0.0%)** |
| **WADI Water Dist.** | 249,371 steps | `1_LT_001_PV` (Elevated) | **0.0552** (5.52%) | 0.0725 | 96.0% | **0 (0.0%)** |

In addition, an isolated 5-pass determinism audit confirmed bitwise zero divergence ($|\Delta| = 0.00\times 10^0$) across sequential evaluations, verifying deterministic reproducibility. Across all 3,150 evaluated scenarios, physical box constraints maintained a strictly 0.0% violation rate.

### Deliverables & Open Science Release
Fulfilling our August NAIRR proposal commitments, all trained PyTorch model checkpoints, ADMM solver libraries, synthetic attack scenario generators, and benchmark logs are released publicly on GitHub under the MIT License at [https://github.com/sjsu-micosys-lab/LAD-TS.git](https://github.com/sjsu-micosys-lab/LAD-TS.git).

### Acknowledgement
The authors acknowledge the National Artificial Intelligence Research Resource (NAIRR) Pilot and the Texas Advanced Computing Center (TACC) Vista system for contributing to this research result.

::: thebibliography
[1] A. A. Giraldo *et al.*, "A Survey of Physics-Based Attack Detection in Cyber-Physical Systems," *ACM Comput. Surv.*, 51(4):1–36, 2018.  
[2] R. Tibshirani *et al.*, "Sparsity and Smoothness via the Fused Lasso," *J. R. Stat. Soc. B*, 67(1):91–108, 2005.  
[3] S.-J. Kim *et al.*, "$\ell_1$ Trend Filtering," *SIAM Review*, 51(2):339–360, 2009.  
[4] S. Boyd *et al.*, "Distributed Optimization and Statistical Learning via ADMM," *Found. Trends Mach. Learn.*, 3(1):1–122, 2011.  
[5] S. Bai, J. Z. Kolter, V. Koltun, "Generic Convolutional and Recurrent Networks for Sequence Modeling," *arXiv:1803.01271*, 2018.  
[6] L. N. Smith, N. Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates," *Proc. SPIE*, 11006:1100612, 2019.  
[7] R. Taormina *et al.*, "Battle of the Attack Detection Algorithms," *J. Water Resour. Plann. Manage.*, 144(8):04018048, 2018.  
[8] J. Goh *et al.*, "A Dataset to Support Research in the Design of Secure Water Treatment Systems," *CRITIS*, pp. 88–99, 2016.  
[9] C. M. Ahmed *et al.*, "WADI: A Water Distribution Testbed for Research in the Design of Secure Cyber Physical Systems," *CyberPnP*, pp. 25–28, 2017.
:::
