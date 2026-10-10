# Temporal Additive Attack Modeling: Progress Report

**Status:** active research prototype on `dev`
**Scope of evidence:** reduced 10,000-row smoke runs on Vista; results are directional, not final cross-dataset claims.

## 1. Research objective and current boundary

The project is building a statistical model of observation-only temporal additive sensor attacks:

\[
y(t) = x(t) + \delta(t)
\]

Here, `y(t)` is the observed sensor value, `x(t)` is the unobserved clean value, and `delta(t)` is the additive attack. The final research objective is to characterize the temporal behavior of `delta(t)`: onset, duration, magnitude, sign, affected sensors, shape, and uncertainty.

HAI, SWaT, WADI, and BATADAL do not provide the clean counterfactual `x(t)`. Their labels therefore support detection, event recall, delay, and localization analysis, but cannot directly validate recovered attack magnitude. Direct delta-recovery metrics are reserved for injected attacks where ground truth is known.

**Implementation evidence:**

- Attack injection and recovery contract: `src/attacks/additive.py`
- Support, magnitude, and sign metrics: `src/evaluation/attack_metrics.py`
- Governing decision: `.agent-tasks/DECISIONS.md` (D-004)

## 2. Experimental trajectory

| Phase | Experiment | What was tested | Result and lesson | Evidence |
|---|---|---|---|---|
| A | Raw statistical baselines | CUSUM, EWMA, and PCA/SPE on standardized sensor levels | CUSUM accumulation was corrected, but raw level scores confused normal operating-regime shifts with attacks. HAI/SWaT EWMA and PCA/SPE reached approximately 99–100% false-positive rate. | `results/statistical_baselines/calibrated_smoke_analysis/INSPECTION.md`; `summary_metrics.png` |
| B | Temporal-residual baselines | CUSUM, EWMA, and PCA/SPE on one-step persistence residuals | Residuals removed the worst level-shift failure and produced attack-aligned spikes. They primarily detect transitions, however, rather than a persistent offset after the signal settles. | `results/statistical_baselines/residual_smoke_analysis/INSPECTION.md`; `score_distribution_shift.png`; `*_traces.png` |
| C | Normal-only LSTM forecaster | Next-step multivariate forecast trained on normal rows; observed minus forecast used as a residual | BATADAL learned a useful normal dynamic. HAI, SWaT, and WADI showed a large train/validation loss gap, so the current shared configuration is not yet transferable. | `results/lstm_forecaster/gh_dev_smoke_analysis/INSPECTION.md`; `*_loss.png`; `*_scores.png` |

## 3. Main results

### Statistical residual baseline

The residual formulation is the first reliable cross-dataset signal. It aligns validation and normal-test score distributions much more closely than raw level scoring.

| Dataset | Best residual observation from the 10k smoke | Interpretation |
|---|---|---|
| HAI | Residual EWMA: 0.17% FPR, 65.1% point TPR, 100% event recall; residual PCA/SPE: 0.85% FPR, 57.0% point TPR, 100% event recall | Strong correction of the raw-level failure. Residual activity rises in the labeled attack interval. |
| SWaT | Residual PCA/SPE: 0.89% FPR, 83.3% event recall, 69-sample median delay | Sparse transition evidence is present, but point coverage is only 1.15%. |
| WADI | Residual PCA/SPE: 0.17% FPR, 100% event recall, 591-sample median delay | A weak late event cue, not a usable detector. |
| BATADAL | Residual EWMA: 2.27% FPR, 40.6% point TPR, 100% event recall | Useful attack-aligned activity, though raw EWMA has higher point recall in this slice. |

The residual approach is a sound baseline for event localization. It is not the final additive-attack model: a one-step difference responds at attack onset and abrupt changes, then can return to normal while a persistent additive offset remains present.

**Evidence:** `results/statistical_baselines/residual_smoke_analysis/INSPECTION.md`, `summary_metrics.png`, `score_distribution_shift.png`, and the dataset trace plots in the same directory.

### LSTM forecaster smoke

The LSTM is a normal-only next-step predictor with train-derived normalization, early stopping, checkpoints, and residual scoring. The 10,000-row GPU smoke used five epochs, 32-step context windows, a 64-unit hidden state, and batch size 256.

| Dataset | Point FPR | Point TPR | Event recall | Delay | Reading |
|---|---:|---:|---:|---:|---|
| HAI | 1.0000 | 1.0000 | 1.0000 | 0 | Invalid operating point: normal test scores remain above the validation threshold. |
| SWaT | 0.0033 | 0.0076 | 0.1667 | 83 | Conservative and mostly misses labeled events. |
| WADI | 0.0006 | 0.0000 | 0.0000 | n/a | No usable signal in this smoke. |
| BATADAL | 0.0288 | 0.5160 | 1.0000 | 1 | Promising: all five events detected with aligned residual peaks. |

The loss curves explain the split. BATADAL training and validation loss both decline across the five epochs. HAI, SWaT, and WADI fit their training portions while validation loss remains much larger: roughly 129, 450, and 54 respectively. This points to normal-regime mismatch or insufficiently representative training segments, not merely a need for more epochs.

**Evidence:** `results/lstm_forecaster/gh_dev_smoke_analysis/INSPECTION.md`, `hai_loss.png`, `swat_loss.png`, `wadi_loss.png`, `batadal_loss.png`, and the corresponding score plots.

## 4. What the experiments establish

1. Raw sensor-level anomaly scores are not a stable proxy for additive attacks across these datasets. Operating regimes can dominate the score.
2. Temporal residuals are a better measurement layer. They generate attack-aligned evidence with bounded false-positive behavior on HAI and BATADAL, and partial event evidence on SWaT/WADI.
3. A learned normal-state predictor is feasible: BATADAL shows that forecast residuals can remain useful throughout labeled attack intervals.
4. The current one-size LSTM is not yet a cross-dataset model. HAI, SWaT, and WADI require representative normal-regime selection and per-dataset experiment configuration before architecture changes are justified.
5. None of the real-dataset results prove delta recovery. They justify building the next injected-attack benchmark around forecast residuals.

## 5. Next research step

Run a bounded BATADAL-focused LSTM ablation with live epoch reporting. Vary context length, hidden size, and training duration while preserving the normal-only train/validation split and blocked threshold policy. BATADAL is the current de-risking dataset because it has decreasing validation loss and attack-aligned residual peaks.

In parallel, add dataset configuration files for HAI, SWaT, and WADI that make normal-regime selection, sensor inclusion, context length, and training parameters explicit. The first question is whether better configuration closes the validation gap. Only if it does not should the project introduce dataset-specific neural architectures.

After a stable forecaster exists, inject step, ramp, periodic, and multi-sensor additive attacks into clean normal sequences. Evaluate `observed - predicted_clean` against known `delta(t)` using the existing recovery metrics. That experiment moves the project from anomaly detection evidence to the stated objective: a statistical model of temporal additive attack error and uncertainty.

## Meeting talking points

- We moved from detector benchmarking toward an explicit attack-modeling measurement layer.
- The important empirical result is not a single best detector. It is that temporal residuals separate normal operating-regime shifts from attack-related changes better than raw sensor levels.
- BATADAL provides a working forecaster case; the other datasets expose a normal-regime transfer problem that must be resolved before broad neural claims.
- The next deliverable is an injected-attack forecaster benchmark with known delta ground truth, not an agentic component. Multi-attack agent reasoning remains a later interpretation layer.
