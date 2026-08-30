# Dataset Consolidation Report

**Project**: Agentic Cyber-Defense for Sensor Data Attacks
**Date**: 2026-08-06
**Status**: Wave 1 complete (SWaT/WADI/BATADAL processed; HAI already available)
**Author**: Aditya Hegde, 019147609, SJSU

---

## 1. HAI Experiments Summary

A CUSUM baseline was evaluated on the HAI 20.07 hardware-in-the-loop testbed (59 sensors, ~1 Hz, 38 attack onsets in the test split). The synthetic evaluation passed all three acceptance criteria (delay, FPR, estimation error), confirming the detector machinery is correct under controlled conditions.

On HAI real data, the system FPR <= 1% / TPR >= 0.90 target is structurally out of reach for per-sensor CUSUM with count-based consensus:

- **Achievable frontier**: max TPR ~0.08 at FPR <= 1%; min FPR ~0.66 at TPR >= 0.90. No (K, C) configuration reaches the corner.
- **Root cause**: 20 of 59 sensors have test-normal means more than 0.5 sigma from the train-fit baseline, with permanent normal-level steps up to 4.3 sigma. Eight of 38 attacks are below 1.8 sigma (partially inside the CUSUM deadband), and one 2,888-sample attack carries 16.5% of all attack samples. The normal-region and attack-region active-sensor distributions overlap at every consensus threshold.
- **Not a tuning issue**: the design space itself (fixed per-sensor k/h, count-based consensus) cannot separate normal drift from weak attacks on this dataset.

**Literature-backed direction**: the ICS anomaly-detection community has converged on segment-level recall as the primary metric (Bauer et al. 2022; Kim & Park 2023; Heydari & Nyarko 2026). For safety-critical OT, an aspirational target of ~10% FPR / 98%+ TPR is realistic, selected via cost-ratio thresholding (Gaffney & Ulvila 2003). Gated rolling sigma (MAD/Qn robust estimators) can reduce false trips but will not close the structural distribution-overlap gap; that requires BOCPD or LSTM-AE detectors.

Full analysis: `docs/research_findings.md`.

---

## 2. Access to the Datasets

### SWaT (Secure Water Treatment, SUTD)

A1 & A2 (Dec 2015): primary benchmark. A3 through A12 are additional releases spanning 2017--2026. A10 (Dec 2023) was not downloaded (~100 GB 7z archives). All acquired files reside in `data/SWaT/`.

### WADI (Water Distribution, SUTD)

A1 (Oct 2017): primary release. A2 (Nov 2019) and A3 (Dec 2023) are additional releases. All acquired files reside in `data/WADI/`.

### BATADAL (Battle of the Attack Detection Algorithms)

Competition datasets: `BATADAL_dataset03.csv` (normal) and `BATADAL_dataset04.csv` (attack). Files reside in `data/BATADAL/extracted/`.

### HAI (Hardware-in-the-loop AI)

HAI 20.07: already available and processed by the existing loader (`src/data/hai_loader.py`). Raw CSVs in `data/hai/raw/hai-20.07/`.

---

## 3. Per-Provider Summary

### SWaT

| Split | Rows    | Time Range                           | Sensors | Sampling | Label Source                                                               |
| ----- | ------- | ------------------------------------ | ------- | -------- | -------------------------------------------------------------------------- |
| train | 365,400 | 2015-12-22 16:30 -- 2015-12-26 21:59 | 51      | 1 s      | All normal                                                                 |
| val   | 129,600 | 2015-12-26 22:00 -- 2015-12-28 09:59 | 51      | 1 s      | All normal                                                                 |
| test  | 449,919 | 2015-12-28 10:00 -- 2016-01-02 14:59 | 51      | 1 s      | 53,885 attack rows (generated from `List_of_attacks_Final.xlsx` intervals) |

- Timestamps: naive local SUTD time (no timezone shift).
- Label source: `List_of_attacks_Final.xlsx` (42 entries); the built-in "Attack" column in `SWaT_Dataset_Attack_v0.xlsx` is not used as the primary label (see Finding b).
- Additional releases (A3--A12) are on disk but have no usable per-sample attack labels; they are normal-only candidates (see Finding c).

### WADI

| Split | Rows      | Time Range                              | Sensors           | Sampling | Label Source                                         |
| ----- | --------- | --------------------------------------- | ----------------- | -------- | ---------------------------------------------------- |
| train | 1,080,001 | 2017-09-25 18:00 -- 2017-10-08 06:00    | 127 (34 constant) | 1 s      | All normal                                           |
| val   | 129,600   | 2017-10-08 06:00:01 -- 2017-10-09 18:00 | 127 (34 constant) | 1 s      | All normal                                           |
| test  | 172,801   | 2017-10-09 18:00 -- 2017-10-11 18:00    | 127 (34 constant) | 1 s      | 15 attack intervals (from `attack_description.xlsx`) |

- Timestamps: naive local SUTD time.
- 34 constant columns (mostly `*_STATUS`/`*_AL` bits plus `PLANT_START_STOP_LOG`); noted in `meta.json` for downstream feature selection.
- Val-end and test-start share one boundary timestamp (2017-10-09 18:00:00); this is inherent to the split carved from the 14-day normal run.
- Nulls are instrumentation artifacts (4 fully-null columns across both files, 5 transient dropouts in the normal period only); no null block aligns with any attack interval. NaN is preserved in the Parquet store.

### BATADAL

| Split | Rows  | Time Range               | Sensors | Sampling        | Label Source                                                       |
| ----- | ----- | ------------------------ | ------- | --------------- | ------------------------------------------------------------------ |
| train | 7,321 | 2014-01-06 -- 2014-11-07 | 43      | 3600 s (hourly) | All normal (`ATT_FLAG` = 0)                                        |
| val   | 1,440 | 2014-11-07 -- 2015-01-06 | 43      | 3600 s (hourly) | All normal                                                         |
| test  | 4,177 | 2016-07-04 -- 2016-12-25 | 43      | 3600 s (hourly) | Attack rows from `ATT_FLAG` column (`1` = attack, `-999` = normal) |

- Hourly cadence (not 1-minute as the original processing plan assumed). The 1,440-row validation split represents 60 days of hourly data.
- Small test set; event-level metrics recommended (noted in `meta.json`).

### HAI

| Split | Rows    | Time Range                           | Sensors | Sampling | Label Source                   |
| ----- | ------- | ------------------------------------ | ------- | -------- | ------------------------------ |
| train | 440,640 | 2019-09-11 20:00 -- 2019-11-03 08:23 | 59      | ~1 s     | 387 attack rows                |
| val   | 110,160 | 2019-11-03 08:24 -- 2019-11-04 14:59 | 59      | ~1 s     | 389 attack rows                |
| test  | 444,600 | 2019-10-29 11:00 -- 2019-11-06 09:29 | 59      | ~1 s     | 17,527 attack rows (38 onsets) |

- 8 zero-variance (constant) columns in raw data: `P1_PCV02D`, `P2_Auto`, `P2_Emgy`, `P2_On`, `P2_TripEx`, `P3_LH`, `P3_LL`, `P4_HT_PS`. These are mapped to unit-variance deterministically at z-score time.
- Z-scoring is performed at load time by `src/data/hai_loader.py` (fit on train only); the Parquet store holds raw values.

### Split Rationale (why the splits are not ratio-based, New Learning for me)

**Why 75/15/10 / 80/10/10 does not apply.** Standard ratio splits assume (a) rows are i.i.d. and shuffleable, (b) both classes appear in training, (c) labels are dense. All three are false for ICS time series: values are autocorrelated and drift, and detectors like CUSUM depend on row order (shuffling destroys the signal and leaks the future into training). Anomaly detection is one-class: the model learns only what normal looks like and flags deviations, because the threat model is novel attacks, not replayed training signatures. This means that the entire train split only consists of 'normal' samples, which was new to me.

**Three constraints that forced these splits.** (1) Chronology -- test must follow train; evaluating on earlier data would leak the future. (2) No attacks in train; validation is normal-only -- val exists to tune thresholds/FPR (CUSUM k/h, consensus K/C), and tuning on attack labels would be test-set leakage. (3) Provider protocol -- these are fixed physical-testbed recordings; the split boundaries are inherited from the vendor's experiment timeline, not chosen by us.

**Why each dataset's sizes look odd:**

| Dataset | Split sizes                                  | Explanation                                                                                                                                                                                                                                                                                                                      |
| ------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SWaT    | train 365,400 / val 129,600 / test 449,919   | 11-day SUTD experiment: ~5.7 days normal, attacks in the final ~5.2 days. Test > train because the provider's attack phase outlasted the normal phase. This is the canonical split used by published SWaT work, so results stay comparable.                                                                                      |
| WADI    | train 1,080,001 / val 129,600 / test 172,801 | 16-day run: 14 days normal + 2 days attack. Test << train (the inverse of SWaT), same protocol logic.                                                                                                                                                                                                                            |
| BATADAL | train 7,321 / val 1,440 / test 4,177         | Competition design: dataset03 = one full normal year (train+val), dataset04 = ~6 attack months (test). Test ~ half of train because the organizers ran the attack phase ~6 months; hourly cadence makes absolute counts small.                                                                                                   |
| HAI     | train 440,640 / val 110,160 / test 444,600   | Vendor-shipped official train/val/test files, boundaries inherited. Note the quirk: test (Oct 29 -- Nov 6) calendar-overlaps train/val (train ends Nov 3), yet zero shared timestamps -- the provider recorded test logs during a concurrent operational window. Disjointness is verified; the overlap is inherited, not chosen. |

**"Train/val normal-only, test attack-heavy" -- what it actually means.** Test is not mostly attacks; it is the attack _window_, and normal rows still dominate:

| Dataset | Attack rows         | Test total | Attack fraction            |
| ------- | ------------------- | ---------- | -------------------------- |
| SWaT    | 53,885              | 449,919    | ~12%                       |
| WADI    | 15 intervals        | 172,801    | ~5.8% in the 2-day window  |
| BATADAL | `ATT_FLAG`=1 subset | 4,177      | subset of the 6-month file |
| HAI     | 17,527              | 444,600    | ~3.9%                      |

The normal >> attack imbalance is inherent to ICS and part of the detection challenge. Training on all-normal is the problem definition: the detector must flag deviations it has never seen.

**Why forcing ratio splits would be wrong.** Shuffling destroys temporal structure and leaks the future into train; a 10% random test of a ~12%-attack split would contain too few positives to estimate TPR/FPR; and abandoning provider boundaries makes results incomparable with the literature, which evaluates on the official splits.

---

## 4. Operations, Processing, and Consolidation

### 4.1 Wave-1 Pipeline

The processing pipeline converts raw source files into a canonical Parquet store:

```
data/{swat,wadi,batadal}/raw/
    -->  data/processed/<dataset>/{train,val,test}.parquet + meta.json
```

**Column contract** (all datasets):

| Column         | dtype       | Description                               |
| -------------- | ----------- | ----------------------------------------- |
| `timestamp`    | datetime64  | Strictly monotonic within each split      |
| `label`        | int8        | 0 = normal, 1 = under attack              |
| `attack_id`    | Int32 / NaN | Attack scenario id (NaN on normal rows)   |
| `<sensor>` x N | float32     | Original raw values, cleaned column names |

**meta.json** per dataset records: sampling interval, sensor column list, split boundaries (path, start, end, rows, label source), raw file SHA-256 checksums, and dataset-specific notes.

**Deterministic rebuild**: same raw inputs produce byte-identical Parquet output. SHA-256 checksums of every raw source file are recorded in `meta.json`.

**Normalization deferred**: raw values are stored. Z-scoring happens at load time using train-only statistics, enforced by locked tests.

### 4.2 Acceptance Validation

`scripts/validate_processed.py` runs 44 automated checks across all four datasets (SWaT: 11, WADI: 11, BATADAL: 10, HAI: 12), verifying:

- Row counts match documented figures
- Sensor counts and column contracts
- Strictly monotonic timestamps with expected cadence
- Label/attack_id dtype and consistency
- No cross-split timestamp overlap (WADI allows 1 shared boundary)
- Raw values unnormalized
- `meta.json` integrity (SHA-256 checksums, required keys)
- Parquet reload equality (loader output matches direct read)

All 44 checks pass (exit 0).

**Test suite**: 67 tests total (`pytest`), including 27 locked specification tests (`tests/spec/`, immutable) and 40 new tests (`tests/test_processed_data.py`). All pass.

### 4.3 Consolidated Store Structure

```
data/processed/
    swat/
        train.parquet
        val.parquet
        test.parquet
        meta.json
    wadi/
        train.parquet
        val.parquet
        test.parquet
        meta.json
    batadal/
        train.parquet
        val.parquet
        test.parquet
        meta.json
    hai/
        train.parquet
        val.parquet
        test.parquet
        meta.json
```

### 4.4 Mandatory Findings

1. **BATADAL cadence is hourly, not 1-minute.** The original processing plan assumed 1-minute sampling. Actual sampling interval is 3600 seconds (hourly). The 1,440-row validation split covers 60 days, not 1 day. `sampling_interval_s=3600` recorded in `meta.json`.

2. **SWaT label discrepancy.** The generated attack list (`List_of_attacks_Final.xlsx` intervals) produces 53,885 attack rows; the built-in "Attack" column in `SWaT_Dataset_Attack_v0.xlsx` reports 54,584. The difference (736 rows) breaks down as:
   - 721 rows: attack #21 has a genuine 12-hour AM/PM discrepancy (attack list says 18:30, built-in says 06:30).
   - 15 rows: a short tail after attack #41 where the built-in label extends ~15 seconds past the list interval.
   - False positives = 0 (the generated labels never mark a normal row as attack).
   - The built-in "Attack" column is retained as the canonical label source; the discrepancy is documented in `data/eda/SWaT_label_crosscheck.md`.

3. **No usable per-sample attack labels in SWaT A3--A12.** A9's `Attack Patterns.txt` contains association rules (antecedent-consequent state patterns), not timestamped attack intervals. None of the A3--A12 releases provides derivable per-timestamp labels. All are normal-only candidates for training. By contrast, WADI A2 has a usable `Attack LABLE` column (9,977 attack rows, ~5.8% of the attack file), though timestamps must be reconstructed from the `Row` column because the `Time` field records only MM:SS (hour-of-day is lost). WADI A3 is clean normal-only.

4. **SWaT A10 skipped.** The ~100 GB 7z archives were not downloaded. Only the split plan document (`SWaT.A10_Split_Archives_Plan.md`) exists on disk. This release is documented but not available for processing.

---

## 5. Next Steps

1. **Dataset-agnostic CUSUM refactor** (before any new runs): parameterize `cusum_experiment.py` by dataset; wire `load_dataset`/`get_meta` from the registry; implement gated rolling sigma (MAD/Qn robust estimators or EWMA of squared deviations).
2. **CUSUM cross-dataset evaluation** on SWaT, WADI, and BATADAL using segment-level recall as the primary metric, with sample-weighted TPR as supplementary. Target: ~10% FPR / 98%+ TPR.
3. **BOCPD detector** (drift-aware by construction) -- HAI first, then cross-dataset.
4. **Autoencoder / sequence model** ensemble as the final detector generation.
5. **Wave 2 data processing** (pending Architect plan extension): SWaT A3--A12 (all normal-only), WADI A2 (labels usable), WADI A3 (normal-only).
