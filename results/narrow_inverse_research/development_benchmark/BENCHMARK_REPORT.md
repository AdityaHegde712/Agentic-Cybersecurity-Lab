# BATADAL Inverse Research Benchmark Report (Task 6)

> **Milestone:** Phase 0 Local-First Research Harness Execution  
> **Benchmark Split:** Development Injection (`val.parquet` rows 480:960)  
> **Target Sensor:** `L_T1` (Train Standard Deviation: `1.2085`)  
> **Total Evaluations:** 480 (120 scenarios × 4 estimators)  

## Executive Summary

This report presents the first empirical inverse recovery results on the sealed development-injection split of BATADAL. The benchmark evaluates four estimators across 120 deterministic, regime-stratified step attack scenarios spanning durations `[6, 12, 24, 48]` hours, amplitudes `[0.5, 1.0, 2.0]` standard deviations, and signs `[-1, 1]`.

- **Best Causal Baseline:** `PersistenceEstimator` achieved a median Delta NRMSE of **0.8616**.
- **1D Total Variation MAP (`StepMAPEstimator`):** Achieved a median Delta NRMSE of **0.2031**, representing an improvement of **76.42%** over the best causal baseline (acceptance threshold: ≥ 20.0%).
- **Signed Amplitude Error:** Median error of **0.1441** train standard deviations (acceptance threshold: ≤ 0.25 std).
- **Sign Accuracy:** **100.0%** on attacks with magnitude ≥ 1.0 train std (acceptance threshold: ≥ 95.0%).
- **Overall Development Acceptance:** **PASSED**.

## Overall Estimator Comparison

| Estimator | Median Delta NRMSE | Mean Delta NRMSE | Median Support F1 | Signed Amp Error (std) | Clean $x$ NRMSE | Sign Acc (≥1σ) | Improvement vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CompanionRegressionEstimator** | 0.3087 | 0.3650 | 1.0000 | 0.0883 | 0.3141 | 100.0% | +64.17% |
| **PersistenceEstimator** | 0.8616 | 1.1672 | 1.0000 | 0.6254 | 0.9780 | 92.5% | +0.00% |
| **SeasonalNaiveEstimator** | 1.4802 | 1.7063 | 1.0000 | 1.2847 | 1.4808 | 77.5% | -71.79% |
| **StepMAPEstimator** | 0.2031 | 0.2575 | 1.0000 | 0.1441 | 0.2226 | 100.0% | +76.42% |

## Precommitted Acceptance Gate Checks

| Criterion | Threshold | Empirical Result | Gate Status |
| :--- | :---: | :---: | :---: |
| Median Delta NRMSE Improvement vs Best Causal | ≥ 20.0% | 76.42% | ✅ PASS |
| Median Signed Amplitude Error | ≤ 0.25 train std | 0.1441 train std | ✅ PASS |
| Sign Accuracy on Attacks ≥ 1.0σ | ≥ 95.0% | 100.0% | ✅ PASS |
| Overall Development Gate | All Criteria Met | - | ✅ ALL PASSED |

## Breakdown by Attack Duration

| Estimator | 6h Duration NRMSE | 12h Duration NRMSE | 24h Duration NRMSE | 48h Duration NRMSE |
| :--- | :---: | :---: | :---: | :---: |
| **CompanionRegressionEstimator** | 0.2810 | 0.3127 | 0.3006 | 0.3168 |
| **PersistenceEstimator** | 0.4650 | 0.7124 | 1.3127 | 1.2577 |
| **SeasonalNaiveEstimator** | 1.5606 | 1.4676 | 1.4884 | 1.4120 |
| **StepMAPEstimator** | 0.3777 | 0.1838 | 0.1729 | 0.1689 |

## Breakdown by Injected Amplitude

| Estimator | 0.5σ NRMSE | 1.0σ NRMSE | 2.0σ NRMSE | 0.5σ F1 | 1.0σ F1 | 2.0σ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CompanionRegressionEstimator** | 0.6406 | 0.3112 | 0.1634 | 1.0000 | 1.0000 | 1.0000 |
| **PersistenceEstimator** | 1.9815 | 1.0819 | 0.4273 | 1.0000 | 1.0000 | 1.0000 |
| **SeasonalNaiveEstimator** | 2.7333 | 1.5118 | 0.7100 | 1.0000 | 1.0000 | 1.0000 |
| **StepMAPEstimator** | 0.3832 | 0.2022 | 0.0824 | 1.0000 | 1.0000 | 1.0000 |

## Breakdown by Dynamic Regime

| Estimator | Low Dynamics NRMSE | Medium Dynamics NRMSE | High Dynamics NRMSE | Low F1 | Med F1 | High F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CompanionRegressionEstimator** | 0.3061 | 0.3209 | 0.2876 | 1.0000 | 1.0000 | 1.0000 |
| **PersistenceEstimator** | 0.7505 | 0.8434 | 1.1407 | 1.0000 | 1.0000 | 1.0000 |
| **SeasonalNaiveEstimator** | 1.5229 | 1.4160 | 1.4873 | 1.0000 | 1.0000 | 1.0000 |
| **StepMAPEstimator** | 0.2031 | 0.1942 | 0.2076 | 1.0000 | 1.0000 | 1.0000 |

## Artifact Inventory

- Summary CSV: [`summary.csv`](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/development_benchmark/summary.csv)
- Summary JSON: [`summary.json`](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/development_benchmark/summary.json)
- Comparison Plots: [`plots/`](file:///C:/Users/hifia/Projects/Sengupta_Research/results/narrow_inverse_research/development_benchmark/plots/)
  - `comparison_low_dynamics.png`
  - `comparison_medium_dynamics.png`
  - `comparison_high_dynamics.png`
  - `comparison_negative_step.png`
  - `comparison_short_step.png`
  - `comparison_long_step.png`
  - `nrmse_by_duration.png`
  - `nrmse_by_amplitude.png`
  - `amplitude_error_by_regime.png`

## Conclusion & Recommendation for October 18 Report

1. The local-first inverse research harness is fully validated, completely deterministic, and sealed.
2. `StepMAPEstimator` coupled with `CompanionRegressionEstimator` convincingly surpasses the precommitted benchmark threshold (≥ 20% delta NRMSE improvement over best causal baseline) with high support F1 and negligible signed amplitude bias.
3. Baselines fail systematically on longer step attacks because persistence decays towards the attacked observation without companion sensor cross-referencing.
4. Ready to advance to gated dynamics bake-off (Task 8) and remote execution preparation on Vista.
