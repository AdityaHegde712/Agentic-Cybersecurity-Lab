# Research Master TODO & Next Session Blueprint
**Created:** 2026-11-10T02:08:00-08:00  
**Branch:** `feature/narrow-additive-inverse-research`  
**Status:** Locked for Session Handoff  

---

## 1. Immediate Next Morning Priorities (Report & Visuals)
- [ ] **Task 1: Human-Readable Advisor Visuals**
  - Re-render the 5 publication figures with clean, low-density layouts (single-trace curves, uncluttered typography, high contrast, readable by the advisor in 30 seconds).
- [ ] **Task 2: Mathematical Report Drafting (Sections 1 & 2)**
  - Pure mathematical SCADA observation model ($\mathbf{y}_t = \mathbf{x}_t + \boldsymbol{\delta}_t$) and causal lookback breakdown ($W=48\,\text{h}$) for Section 1.
  - Section 2 transformation pipeline: Clean Telemetry $\to$ Parameterized Corruption $\to$ TCN Surrogate $\to$ Banded ADMM Solver $\to$ GED Error Distribution.
- [ ] **Task 3: Hierarchical LaTeX Benchmark Table**
  - Integrate full-width multi-column table covering 5 attack families, 3 estimators, NRMSE, $F_1$, and physical bounds.

---

## 2. High-Capacity GPU & Scaling Track (Professor Directives)
- [ ] **Task 4: High-Throughput Parametric Attack Generator**
  - Build automated batch generator producing hundreds to thousands of synthetic attack trajectories (morphology sweeps, stochastic ramps, composite waveforms, multi-frequency sinusoids, adversarial noise).
  - Benchmark champion estimators under massive-scale stress tests.
- [ ] **Task 5: Gradient-Based Adversarial Attack Formulation**
  - Design stealthy optimization attacks (e.g., Projected Gradient Descent / FGSM tailored to cyber-physical time-series) that maximize estimation error while respecting residual detection thresholds ($\le 3\sigma_\epsilon$).
- [ ] **Task 6: Spatio-Temporal Transformer Surrogate (GPU-Intensive Training)**
  - Design and train a data-hungry Time-Series Transformer / PatchTST / Spatio-Temporal Graph Transformer on full multi-channel telemetry.
  - Maximize GPU cluster utilization across long sequence horizons and cross-attention companion features.
- [ ] **Task 7: Cross-Dataset Transfer (SWaT & WADI)**
  - Ingest SWaT (51 channels, water treatment) and WADI (127 channels, water distribution) clean partitions.
  - Train companion surrogates and evaluate inverse recovery generalization across distinct municipal network topologies.
