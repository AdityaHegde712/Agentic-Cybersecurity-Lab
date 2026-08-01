# LOCKED — do not modify (TDD contract)

"""LOCKED contract tests for ``src.detection.cusum.CUSUMDetector`` (B1).

RED phase (now): these tests MUST fail at collection time because
``src/detection/cusum.py`` does not exist yet.
GREEN phase (after B1): they must pass UNCHANGED against the implementation.
Do not relax or weaken these assertions. This file is an immutable contract.

Public API under test (exact):
    from src.detection.cusum import CUSUMDetector
    det = CUSUMDetector(mu0: float, k: float, h: float,
                        reset_after_detection: bool = True)
    is_cp, stat = det.update(y: float)      # -> tuple[bool, float]
    delta = det.estimate_delta_x(window: np.ndarray)   # -> float
    det.reset()                             # clears internal state (BaseDetector)

Semantics (one-sided upper CUSUM, standard):
    S_t = max(0, S_{t-1} + y_t - mu0 - k); alarm fires when S_t > h
    reset_after_detection=True -> state resets after an alarm fires
    estimate_delta_x estimates the post-shift level relative to mu0
    (e.g. mean(window) - mu0).

Parameter choices and safety margins (noise std = 0.1):
    k = 0.5 : half the smallest target shift used here (+2.0). In-control
              increments y - mu0 - k ~ -0.5 +/- 0.1 are strongly negative, so
              S sits at 0 on clean data (max stat simulated = 0.0 over 5000
              samples at the locked seed). FPR bound is exact: 0 alarms.
    h = 5.0 : 50x the noise std. A +2.0 shift gives expected drift
              1.5/sample -> alarm at ~sample 3 (simulated: 203, shift at 200).
              The locked bound of 50 is ~17x the expected latency, so any
              reasonable S_t parametrization (including a sigma-scaled ARL
              form) passes.
    estimate_delta_x: a window mean has std 0.1/sqrt(200) ~ 0.007, so the 10%
              relative bound on a 5.0 shift leaves ~70 sigma of slack and the
              clean-window sanity bound [-1.0, 1.0] leaves ~140 sigma.

Generator tests (6-7) — deliberate, documented adaptation of the draft spec:
    The B0 generator baseline is a deterministic sine of amplitude 10 (locked
    B0 design). A fixed-mu0 CUSUM fed the RAW sensor columns false-alarms
    massively on the clean 'none' run (simulated: ~750 alarms over 2000
    samples, first at sample 10-21) for ANY k/h, so that reading of the draft
    is unsatisfiable. These tests therefore feed each detector the PAIRED
    DEVIATION between two same-parameter generator runs — the exact isolation
    convention B0 already locks in test_synthetic.py ("the difference isolates
    the injected delta_x and is robust to the unspecified baseline signal").
    The paired runs use DIFFERENT seeds so the deviation retains real noise of
    std 0.1*sqrt(2) ~ 0.141: the FPR assertion is genuine, not vacuous.
    k=0.5, h=5.0 are retained as specified:
        - 'step' deviation = 3.0 + noise_diff (SNR ~21) -> alarm at ~t0+1
          (locked bound: within 200 samples of t0; simulated t0+1).
        - clean deviation = noise_diff: max stat simulated 0.04 vs h=5.0;
          locked FPR = 0 alarms over 2000 samples x 3 sensors.
"""

import numpy as np

from src.detection.cusum import CUSUMDetector

MU0 = 0.0
K = 0.5
H = 5.0
NOISE = 0.1


def _feed(det, y):
    """Feed a 1-D array through det; return (first_alarm_index, stats).

    ``stats`` holds every stat returned by ``update``. ``first_alarm`` is the
    index of the first sample that triggered an alarm, or None if none did.
    """
    first_alarm = None
    stats = np.empty(len(y), dtype=float)
    for i, val in enumerate(y):
        is_cp, stat = det.update(float(val))
        stats[i] = stat
        if is_cp and first_alarm is None:
            first_alarm = i
    return first_alarm, stats


# ---------------------------------------------------------------------------
# 1. Clean data -> zero false alarms (exact FPR = 0)
# ---------------------------------------------------------------------------

def test_no_false_alarm_on_clean_data():
    det = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=True)
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, NOISE, 5000)
    first_alarm, _ = _feed(det, y)
    assert first_alarm is None


# ---------------------------------------------------------------------------
# 2. Step shift +2.0 detected within 50 samples, no pre-shift alarm
# ---------------------------------------------------------------------------

def test_step_shift_detection_within_50_samples():
    det = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=True)
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0.0, NOISE, 200), np.full(100, 2.0)])
    first_alarm, _ = _feed(det, y)
    assert first_alarm is not None
    assert 200 <= first_alarm < 200 + 50  # no alarm pre-shift; alarm within 50


# ---------------------------------------------------------------------------
# 3. Cumulative statistic shape: 0 <= S_t <= h on clean, S_t > h on shift
#    (reset_after_detection=False so the crossing stat is observable)
# ---------------------------------------------------------------------------

def test_statistic_shape_clamped_and_crosses_threshold():
    det_clean = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=False)
    rng = np.random.default_rng(0)
    _, stats_clean = _feed(det_clean, rng.normal(0.0, NOISE, 5000))
    assert np.all(stats_clean >= 0.0)
    assert np.all(stats_clean <= H)

    det_shift = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=False)
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0.0, NOISE, 200), np.full(100, 2.0)])
    first_alarm, stats_shift = _feed(det_shift, y)
    assert first_alarm is not None
    assert np.all(stats_shift >= 0.0)
    assert np.any(stats_shift > H)  # the crossing stat is observed


# ---------------------------------------------------------------------------
# 4. reset() clears internal state (explicit path + auto-reset path)
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    # Explicit reset: state is high after a detection (no auto-reset), then
    # reset() must clear it so 1000 clean samples produce no alarm.
    det = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=False)
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0.0, NOISE, 200), np.full(100, 2.0)])
    first_alarm, _ = _feed(det, y)
    assert first_alarm is not None
    det.reset()
    rng = np.random.default_rng(1)
    first_alarm_after, _ = _feed(det, rng.normal(0.0, NOISE, 1000))
    assert first_alarm_after is None

    # Auto-reset path: with reset_after_detection=True the same holds WITHOUT
    # an explicit reset() call (state already cleared by the detector).
    det2 = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=True)
    rng = np.random.default_rng(0)
    y2 = np.concatenate([rng.normal(0.0, NOISE, 200), np.full(100, 2.0)])
    first_alarm2, _ = _feed(det2, y2)
    assert first_alarm2 is not None
    rng = np.random.default_rng(1)
    first_alarm_after2, _ = _feed(det2, rng.normal(0.0, NOISE, 1000))
    assert first_alarm_after2 is None


# ---------------------------------------------------------------------------
# 5. estimate_delta_x: 10% relative bound on a real shift, sanity on clean
# ---------------------------------------------------------------------------

def test_estimate_delta_x_accuracy():
    det = CUSUMDetector(mu0=MU0, k=K, h=H)

    # Shifted window: constant 5.0 + noise; estimate within 10% of 5.0.
    rng = np.random.default_rng(0)
    window = np.full(200, 5.0) + rng.normal(0.0, NOISE, 200)
    est = det.estimate_delta_x(window)
    assert isinstance(est, (float, np.floating)), type(est)
    assert abs(est - 5.0) / 5.0 < 0.1

    # Clean window: must not hallucinate a large shift.
    clean = np.random.default_rng(0).normal(0.0, NOISE, 200)
    est_clean = det.estimate_delta_x(clean)
    assert isinstance(est_clean, (float, np.floating)), type(est_clean)
    assert -1.0 <= est_clean <= 1.0


# ---------------------------------------------------------------------------
# 6. Multivariate per-sensor application on the project generator (step)
# ---------------------------------------------------------------------------

def test_multivariate_per_sensor_step_detection_on_generator():
    from src.data.synthetic import SensorDataGenerator

    T = 2000
    step_sg = SensorDataGenerator(n_sensors=3, sampling_rate=1.0,
                                  noise_level=NOISE, correlation=0.0, seed=1)
    data_step, meta = step_sg.generate(n_timesteps=T, attack_type='step',
                                       t0=None, c=3.0)
    none_sg = SensorDataGenerator(n_sensors=3, sampling_rate=1.0,
                                  noise_level=NOISE, correlation=0.0, seed=2)
    data_none, _ = none_sg.generate(n_timesteps=T, attack_type='none')
    t0 = meta['t0']
    assert t0 == T // 2

    # Paired deviation isolates the injected step (B0 isolation convention).
    dev = data_step - data_none
    for s in range(3):
        det = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=True)
        first_alarm, _ = _feed(det, dev[:, s])
        assert first_alarm is not None, f"sensor {s}: no alarm on attacked column"
        assert t0 <= first_alarm < t0 + 200, \
            f"sensor {s}: first alarm at {first_alarm} (t0={t0})"


# ---------------------------------------------------------------------------
# 7. FPR bound on generator data: strict zero alarms over all sensors
# ---------------------------------------------------------------------------

def test_zero_false_alarms_on_clean_generator_run():
    from src.data.synthetic import SensorDataGenerator

    T = 2000
    sg_a = SensorDataGenerator(n_sensors=3, sampling_rate=1.0,
                               noise_level=NOISE, correlation=0.0, seed=3)
    sg_b = SensorDataGenerator(n_sensors=3, sampling_rate=1.0,
                               noise_level=NOISE, correlation=0.0, seed=4)
    da, _ = sg_a.generate(n_timesteps=T, attack_type='none')
    db, _ = sg_b.generate(n_timesteps=T, attack_type='none')
    dev = da - db  # clean deviation: real noise of std 0.1*sqrt(2)

    total_alarms = 0
    for s in range(3):
        det = CUSUMDetector(mu0=MU0, k=K, h=H, reset_after_detection=True)
        for val in dev[:, s]:
            is_cp, _ = det.update(float(val))
            if is_cp:
                total_alarms += 1
    assert total_alarms == 0
