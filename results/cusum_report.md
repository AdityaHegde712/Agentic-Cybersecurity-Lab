# CUSUM Baseline Evaluation Report (B1)

**Status: ACCEPTANCE FAILED** — honest baseline results as-is (owner decision, 2026-07-31).
Locked contract tests: 27/27 PASS. Eval acceptance criteria: 0/3 PASS (details below).

## 1. Method

One-sided upper CUSUM statistic, per-sensor detectors:

```
S_t = max(0, S_{t-1} + y_t - mu_0 - k)
alarm when S_t > h
```

- **Synthetic evaluation**: fixed absolute parameters `mu_0=0`, `k=0.5`, `h=5.0` for all 12 (c, noise) cells. Paired-deviation inputs `dev = data(step, seed_a) - data(none, seed_b)`, so the noise std on the deviation is `sigma_dev = sqrt(2) * noise_level` (0.141 / 1.414 / 4.708). T=10,000, attack at t0=5,000, 5 sensors, 3 seeds, per-sensor detectors. Delta-x estimated as mean of a 200-sample post-t0 window minus mu_0.
- **HAI evaluation**: per-sensor `mu_0` = mean of train-normal rows (`train.y == 0`), `k = 0.5 * sigma`, `h = 5.0 * sigma`, `reset_after_detection=True`. System alarm = OR over all 59 sensors. All 59 sensors had nonzero training std (constant columns live in metadata, not sensor columns).

## 2. Synthetic Results

| c | noise | delay mean | delay std | FPR mean | FPR max | err mean % | err max % |
|---|---|---|---|---|---|---|---|
| 1  | 0.10 | 9.2 | 0.9 | 0.000 | 0.000 | 0.93 | 1.88 |
| 1  | 1.00 | -4903.0 | 71.5 | 0.011 | 0.013 | 9.29 | 18.77 |
| 1  | 3.33 | -4995.4 | 4.1 | 0.170 | 0.181 | 30.94 | 62.50 |
| 5  | 0.10 | 1.0 | 0.0 | 0.000 | 0.000 | 0.19 | 0.38 |
| 5  | 1.00 | -4903.0 | 71.5 | 0.011 | 0.013 | 1.86 | 3.75 |
| 5  | 3.33 | -4995.4 | 4.1 | 0.170 | 0.181 | 6.19 | 12.50 |
| 10 | 0.10 | 0.0 | 0.0 | 0.000 | 0.000 | 0.09 | 0.19 |
| 10 | 1.00 | -4903.0 | 71.5 | 0.011 | 0.013 | 0.93 | 1.88 |
| 10 | 3.33 | -4995.4 | 4.1 | 0.170 | 0.181 | 3.09 | 6.25 |
| 20 | 0.10 | 0.0 | 0.0 | 0.000 | 0.000 | 0.05 | 0.09 |
| 20 | 1.00 | -4903.0 | 71.5 | 0.011 | 0.013 | 0.47 | 0.94 |
| 20 | 3.33 | -4995.4 | 4.1 | 0.170 | 0.181 | 1.55 | 3.13 |

**Interpretation.** Low-noise cells behave as designed: delay 0-9 samples (scaled by c), FPR 0%, delta-x error < 1%. Medium/high noise cells are broken: the fixed absolute thresholds `k=0.5, h=5.0` are far below the deviation noise scale (sigma_dev 1.41 / 4.71), so the statistic crosses h within the first few samples **before** the attack — hence the negative delays (-4903 = alarm at sample ~97; -4995 = alarm at sample ~5). No misses occur only because the detector is always already tripped.

### Acceptance checks (synthetic)

| Criterion | Result | Verdict |
|---|---|---|
| Delay 5-10 samples @ SNR 3 (c=10, noise=3.33) | -4995 (alarm pre-attack) | **FAIL** |
| FPR < 1% across all noise levels | 0% @ 0.1; 1.1% @ 1.0; 17.0% @ 3.33 | **FAIL** |
| Delta-x error < 10% | PASS for c>=5 at noise <= 1.0; max 62.5% at c=1/noise=3.33 | **PARTIAL** |

## 3. HAI Results (test split, 444,600 rows)

- Positive (attack) labels: 17,527 (3.9%). System alarms: 402,877 (90.6%).

| | Pred alarm | Pred normal |
|---|---|---|
| **True attack** | TP 16,786 | FN 741 |
| **True normal** | FP 386,091 | TN 40,982 |

- **TPR = 0.958, FPR = 0.904**
- First system alarm after attack onset: delay 0 samples — but this is **meaningless**: the system is in near-permanent alarm state, so it is "alarmed" at the onset by construction.

**Root cause.** OR-aggregation over 59 sensors: with per-sensor `k=0.5*sigma, h=5*sigma` the per-sensor false-alarm probability is small but non-zero, and the system-level false-alarm rate is the union over 59 sensors. Measured 90.4% implies the system ARL collapses to ~10 samples. The detector does carry genuine attack information (TPR 0.958), but it is drowned in the alarm floor.

## 4. Limitations and Root-Cause Summary

1. **Thresholds not noise-calibrated (synthetic).** Absolute `k=0.5, h=5.0` are only valid when the input noise is small. The paired-deviation convention doubles noise variance; cells at noise 1.0 / 3.33 are out of calibration.
2. **System aggregation amplifies FPR (HAI).** OR over 59 independently-false-alarming sensors destroys system-level specificity even when per-sensor behavior is reasonable.
3. **Delta-x estimator variance.** 200-sample window mean has SEM = sigma/sqrt(200); at sigma_dev=4.71 that is 0.33 absolute, i.e. 33% relative error at c=1. Unbiased but high-variance at low SNR.
4. **Reset during sustained attack** not evaluated here (reset_after_detection=True re-arms quickly under a persistent shift; alarm counts were not tracked).

## 5. Recommendations (next iteration — not executed, owner decision)

1. **Relative CUSUM**: `k = 0.5 * sigma_dev`, `h = 5.0 * sigma_dev` per scenario, sigma estimated from calibration data (this is the standard normalization for CUSUM).
2. **HAI system-level consensus**: require >= 2 (or 3) of 59 sensors to alarm within a cooldown window; or raise `h` to target a system ARL of 1000+ samples via a union bound.
3. **Delay metric**: measure first alarm *after* t0 (clip pre-attack alarms), and exclude cells where the detector is in pre-trigger state.
4. **Larger estimation window** (e.g., 500-1000 samples) or median-of-window estimator for delta-x at low SNR.

## 6. Artifacts

- `results/cusum_evaluation_synthetic.csv` (180 rows)
- `results/cusum_evaluation_hai.csv` (444,600 rows)
- `src/detection/cusum.py`, `src/evaluation/metrics.py`, `src/evaluation/cusum_experiment.py`
- Locked contract: `tests/spec/test_cusum.py` (7 tests, all passing)
