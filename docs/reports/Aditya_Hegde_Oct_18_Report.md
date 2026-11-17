::: center
**Inferring Latent Attacker Distributions from Time-Series
Observations**\
NAIRR Pilot Start-Up Two-Month Update October 19, 2026 Request ID:
`NAIRR260371`\

  -------------------------------------- ----------------------------------
     Aditya Hegde (Graduate Student)          Saptarshi Sengupta (PI)
   *Department of Computer Engineering*   *Department of Computer Science*
       *San José State University*          *San José State University*
  -------------------------------------- ----------------------------------
:::

In industrial Cyber-Physical Systems (CPS), adversarial additive
spoofing corrupts sensor telemetry at discrete timesteps
$t \in \{1, \dots, T\}$ according to $y_t = x_t + \delta_t$, where
$x_t \in \mathbb{R}$ is the unobserved physical plant state and
$\delta_t \in \mathbb{R}$ is the unknown attack perturbation
($\delta_t = 0$ during clean operations). Recovering both trajectories
from telemetry $y$ alone is non-identifiable without side information or
structural priors: any decomposition
$\tilde{\delta}_t = y_t - \tilde{x}_t$ yields identical observations.
Standard causal estimators (Persistence, Seasonal Naive) fail under
sustained spoofing because corrupted measurements directly contaminate
their autoregressive states, driving severe divergence (Persistence
median NRMSE $0.8616$; Seasonal Naive $1.4802$).

To restore identifiability without ground truth at test time, we
decouple the attacked channel from its corrupted history using
uncompromised companion channels $z_t \in \mathbb{R}^d$ (downstream
flows, pump statuses, adjacent pressures). A surrogate model yields
clean state estimates $\hat{x}_{\text{surr}, t} = f_\theta(z_t)$ with
residual variance
$\sigma_\epsilon^2 = \mathbb{E}[(x_t - \hat{x}_{\text{surr}, t})^2]$.
Exploiting piecewise-constant attack structure
($\sum_{t=1}^{T-1} |\delta_{t+1} - \delta_t| \ll T$), the perturbation
trajectory is recovered via 1D Total Variation (TV) MAP optimization
equivalently, the fused lasso signal approximator [@fused; @rof]:
$$\begin{equation}
\min_{\delta \in \mathbb{R}^T} \;\; \frac{1}{2\sigma_\epsilon^2} \sum_{t=1}^T \left( y_t - \hat{x}_{\text{surr}, t} - \delta_t \right)^2 + \lambda_{\text{TV}} \sum_{t=1}^{T-1} |\delta_{t+1} - \delta_t| + \lambda_{\ell_1} \sum_{t=1}^T |\delta_t|
\end{equation}$$ The TV and $\ell_1$ penalties collapse high-frequency
surrogate residual flutter into flat plateaus, recovering the plant
state as $\hat{x}_t = y_t - \hat{\delta}_t$.

Attacks are injected open-loop on the single Tank 1 channel of hourly
telemetry and evaluated across 120 deterministic scenarios on the sealed
BATADAL [@batadal] split (6h--48h durations,
$0.5\sigma$--$3.0\sigma_{\text{train}}$ amplitudes), StepMAP with a
linear surrogate achieves median NRMSE of 0.2031 (76.4% error reduction
over Persistence). Upgrading to a `DilatedTCNSurrogate` [@tcn] (48h
context, trained on TACC Vista GH200; validation RMSE
$0.0678\,\text{m}$, $R^2 = 0.9968$, compressing residual spread by
82.4%) reduces median NRMSE to 0.1605 ($-21.0\%$ vs. Linear StepMAP;
$-81.4\%$ vs. Persistence). For 48h attacks, error drops from 0.1689 to
0.0766 ($-54.6\%$), while weak $0.5\sigma$ attack NRMSE drops from
0.3832 to 0.2935 ($-23.4\%$) by lowering the surrogate noise floor.

  **Estimator**                 **Median NRMSE**   **Mean NRMSE**   **Median Amp. Err. ($\sigma$)**   **Sign Acc. ($\ge 1\sigma$)**
  ---------------------------- ------------------ ---------------- --------------------------------- -------------------------------
  Persistence Baseline               0.8616            1.1672                   0.6254                            92.5%
  Seasonal Naive Baseline            1.4802            1.7063                   1.2847                            77.5%
  Companion Ridge Regression         0.3087            0.3650                   0.0883                           100.0%
  StepMAP (Linear Surrogate)         0.2031            0.2575                   0.1441                           100.0%
  StepMAP (Neural TCN)               0.1605            0.2011                   0.1446                           100.0%

In the remaining month of the allocation, we will explore multi-channel
attacks and test robustness when companion channels are partially
compromised. Consistent with NAIRR Pilot open-science requirements, we
will release the implementation, trained surrogate and evaluation suite.

The authors acknowledge the National Artificial Intelligence Research
Resource (NAIRR) Pilot and the Texas Advanced Computing Center (TACC)
Vista system for contributing to this research result.

::: thebibliography
9 R. Tibshirani, M. Saunders, S. Rosset, J. Zhu, K. Knight, "Sparsity
and Smoothness via the Fused Lasso," *J. R. Stat. Soc. B*,
67(1):91--108, 2005. L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear Total
Variation Based Noise Removal Algorithms," *Physica D*, 60:259--268,
1992. R. Taormina *et al.*, "Battle of the Attack Detection Algorithms:
Disclosing Cyber Attacks on Water Distribution Networks," *J. Water
Resour. Plann. Manage.*, 144(8):04018048, 2018. S. Bai, J. Z. Kolter,
V. Koltun, "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018.
:::
