# Research Findings — CUSUM Baseline and HAI Feasibility (Sengupta_Research ICS Sensor-Attack Detection PoC)

| Field    | Value                                      |
| -------- | ------------------------------------------ |
| Version  | 1.0                                        |
| Date     | 2026-08-05                                 |
| Status   | Maintained (append per build phase)        |

---

## 1. Purpose & Scope

Maintained research findings log for the PoC. The initial evaluation (CUSUM baseline) comprises two analyses: (a) synthetic acceptance evaluation (controlled), (b) HAI real-ICS-data feasibility evaluation.

Canonical attack model (per `docs/system-design.md`):

```
y(t) = x(t) + delta_x(t)
```

where `x` is the true value, `delta_x` the adversarial perturbation, `y` the observed reading.

Key question answered by this report: why the HAI acceptance target (system FPR <= 1% AND system TPR >= 0.90) is structurally out of reach for the CUSUM + consensus design, and what was learned from the attempt.

## 2. Experiment Structure

### 2.1 Synthetic Evaluation

- **Generator**: `SensorDataGenerator` (`src/data/synthetic.py`): 5 sensors, deterministic sine baseline (amplitude 10, integer periods), iid Gaussian noise with std = `noise_level`. Attack types:
  - `step` (magnitude `c`): a sudden constant offset added to the signal from `t0` onward.
  - `ramp` (rate `r`): an offset that grows linearly over time from `t0`.
  - `periodic` (amplitude `A`, frequency `f`): an oscillating offset applied from `t0`.
  - `coordinated` (sensors subset): a step applied only to a selected subset of sensors.
- **Paired-deviation design**: `dev = data_attack(seed_a) − data_none(seed_b)`. Here, `data_attack(seed_a)` and `data_none(seed_b)` are two runs of the same generator with the same deterministic sine baseline but different random seeds for the Gaussian noise. `data_attack` is the generator invoked with any of the listed attack types (`step`, `ramp`, `periodic`, `coordinated`); `data_none` is the generator invoked with `attack_type='none'` (no attack). Because the baseline sines are deterministic and identical in both runs, subtracting them cancels the baseline exactly. Two different seeds are used so the noise in the two runs is independent: the residual `dev` is then exactly the attack effect plus independent noise with `sigma_dev = sqrt(2) * noise_level` (independent Gaussian noises combine in quadrature). This makes the comparison unbiased — the "no-attack" reference is a clean, independent instance of the same physical signal, so `dev` is ~0 when no attack is present and equals the injected perturbation when one is.
- **Grid**: `c` in {1, 5, 10, 20} x `noise_level` in {0.1, 1.0, 3.33} x 3 seeds x 5 sensors; T = 10,000 samples, attack at `t0` = 5,000.
- **Detector**: one-sided upper CUSUM (`src/detection/cusum.py`): `S_t = max(0, S_{t-1} + y_t − mu0 − k)`; alarm when `S_t > h`; `reset_after_detection = True`.
- **Tuning policy** (tuning iteration): `k = 0.5 * sigma_dev`, `h = 12.0 * sigma_dev`; delay = first alarm at index > `t0` (pre-attack alarms clipped); runs with no post-`t0` alarm counted as misses; `delta-x` estimated as mean of a 4,000-sample post-`t0` window minus `mu0`; FPR measured on separate attack-free paired clean runs.

#### Acceptance Criteria and Results (3/3 PASS)

| Criterion                          | Target        | Measured                     | Verdict |
| ---------------------------------- | ------------- | ---------------------------- | ------- |
| C1 delay @ c=10, noise=3.33       | 5–10 samples  | 6.33                         | PASS    |
| C2 FPR per noise level (0.1 / 1.0 / 3.33) | < 1% | 0.0000 / 0.0000 / 0.0000 | PASS |
| C3 max delta-x mean cell error     | < 10%         | 7.77% (cell c=1, noise=3.33) | PASS    |

- **Known trade-off**: `h = 12 sigma` is conservative; the weakest cell (c=1, noise=3.33) has 12/15 missed detections because the shift (c=1) is below the CUSUM deadband `k = 0.5 * sigma_dev`.
- **Legitimacy note**: the zero FPR is the statistically expected outcome, not a bias artifact — the in-control average run length (ARL0) at `k=0.5σ`/`h=12σ` is roughly 10^6 samples, versus 10,000-sample runs, so ~0.14 false alarms are expected across the 15 clean runs at each noise level.

### 2.2 HAI Evaluation (Real ICS Data)

- **Dataset**: HAI 20.07 — hardware-in-the-loop OT testbed, ~1 Hz. 59 sensor columns (64 columns minus 5 metadata/time/attack columns). Train split 550,800 rows (80/20 chronological train/val); test split 444,600 rows with 38 attack onsets and 17,527 attack-labeled samples (3.9% of test).
- **Preprocessing**: per-sensor z-score fit on TRAIN rows only, applied to train/val/test (no leakage; enforced by locked tests).
- **Detector setup**: one CUSUM per sensor. Fixed-mu0 baseline: `mu0` = train-normal mean per sensor, `k = 0.5 * std`, `h = 5.0 * std`. System alarm = consensus: >= K distinct sensors alarm within a trailing C-sample window.
- **Target variable**: the dataset's `attack` label (0 = normal, 1 = attack). Metrics: system FPR = FP/(FP+TN), system TPR = TP/(TP+FN) over labeled samples; delay = first system alarm inside each attack segment after onset, median over onsets.

#### Results

- **Fixed mu0** (shipped K=7, C=5): FPR 0.8716, TPR 0.9049, delay 0 (vacuous — system near-permanently alarmed), 1/38 onsets missed.
- **Adaptive mu0** (gated EWMA, grid `g` in {1.6, 1.7, 2.0} x `alpha` in {0.001, 0.005, 0.01}; consensus K in {2..15} x C in {3, 5, 10}): shipped `g=2.0`, `alpha=0.01`, `K=8`, `C=3` — FPR 0.0127, TPR 0.1815, median delay 82 samples, 8/38 onsets missed.

## 3. Finding: The HAI Target Is Out of Reach for CUSUM + Consensus

### 3.1 The Target

System FPR <= 1% AND system TPR >= 0.90 on the HAI test split (sample-weighted).

### 3.2 Evidence

- **Achievable frontier**: max TPR ≈ 0.08 at FPR <= 1% (K=10); min FPR ≈ 0.66 at TPR >= 0.90 (K=2). No (K, C) config reaches the corner.
- **Fixed-mu0 offset floor**: 20 of 59 sensors have test-normal mean > 0.5 sigma away from the train-fit `mu0` (max |shift| = 1.55 sigma), producing a per-sensor false-trip floor of ~2.6 events/sample.
- **Gated-EWMA adaptation** removed roughly 10x of that offset floor but hit a structural wall (below).

### 3.3 Why It Is Structurally Out of Reach (Root-Cause Analysis)

1. **Non-stationary test-normal data**: slow multi-sigma level drift on many sensors plus a few permanent level steps up to ~4.3 sigma that exceed any attack-blocking gate; a baseline that never absorbs a >= 1.8 sigma attack also freezes on those normal steps and trips for the whole test.
2. **Short and weak attacks**: 8 of 38 attacks are < 1.8 sigma (partially inside the `k = 0.5 sigma` deadband); the shortest attack is 151 samples; one 2,888-sample attack carries 16.5% of all attack samples, so sample-weighted TPR is dominated by segment length.
3. **Distribution overlap**: normal-region active-sensor count reaches ~8–10 (p95–p99) while attack-region count starts at 1–2 (p10); no consensus threshold separates the classes.
4. **Gating dilemma**: a baseline gate wide enough to track the normal drift absorbs the weak attacks; a gate narrow enough to preserve the weak attacks freezes on the drift. A single per-sensor CUSUM baseline cannot satisfy both.

### 3.4 What This Means

- **Not a bug or mis-tuning**: the design space itself (per-sensor CUSUM + count-based consensus + fixed per-sensor k/h) cannot reach the FPR<=1%/TPR>=0.90 corner on this dataset.
- The **synthetic results (3/3 PASS)** confirm the detector machinery is correct under controlled conditions; HAI exposes the distribution-shift reality of real OT data.
- **Later-phase detectors** are the designed path: subsequent detector generations — BOCPD (drift-aware by construction) and an LSTM-AE ensemble. A segment-level (vs sample-weighted) TPR, or a two-tier target, may be a more meaningful operational metric.

## 4. Open Questions for the Owner

1. Should the HAI metric be reformulated (e.g., segment-level detection vs sample-weighted TPR)?
2. Is FPR <= 1% the right operating point for OT operations (cost of false alarms vs missed attacks)?
3. Is it acceptable to carry HAI hardening as an open item into subsequent detector generations?

## 5. Artifacts & References

- `results/cusum_report.md` (full numeric detail, honest verdicts)
- `results/cusum_evaluation_synthetic.csv`, `results/cusum_evaluation_hai.csv`, `results/cusum_hai_sensor_events.pkl`
- `src/evaluation/cusum_experiment.py` (evaluation harness)
- `src/detection/cusum.py`, `src/data/synthetic.py`, `src/data/hai_loader.py`, `src/evaluation/metrics.py`
- `docs/system-design.md` (attack model and development plan)
- `tests/spec/` (locked contract; 27/27 PASS as of 2026-08-05)

## Changelog

| Date       | Version | Description                                                                                           |
| ---------- | ------- | ----------------------------------------------------------------------------------------------------- |
| 2026-08-05 | 1.0     | Initial report — B1 synthetic 3/3 PASS, HAI feasibility analysis (target structurally out of reach for CUSUM+consensus), open questions. |
