# LOCKED — do not modify (TDD contract)

"""LOCKED contract tests for ``src.data.synthetic.SensorDataGenerator`` (B0).

RED phase (now): these tests MUST fail because ``src/`` does not exist yet.
GREEN phase (after B0): they must pass UNCHANGED against the implementation.
Do not relax or weaken these assertions. This file is an immutable contract.

Public API under test (exact):
    SensorDataGenerator(n_sensors=5, sampling_rate=1.0, noise_level=0.1,
                        correlation=0.0, seed=42)
    sg.generate(n_timesteps=10000, attack_type='step', t0=None, **attack_params)
        -> (data: np.ndarray float, shape (T, N); meta: dict)

Contract rules encoded below:
    - attack_type in {'none', 'step', 'ramp', 'periodic', 'coordinated'}
    - t0 default: n_timesteps // 2
    - meta must contain keys: attack_type, t0, n_timesteps, n_sensors,
      attack_params, noise_level, seed, changepoints (list of int indices)
    - step:      delta_x = c * 1(t >= t0),            c default 1.0
    - ramp:      delta_x = r * (t - t0) * 1(t >= t0), r default 0.01
    - periodic:  delta_x = A * sin(2*pi*f*(t - t0)) * 1(t >= t0),
                 A default 1.0, f default 0.01
    - coordinated: same injection applied across attack_params['sensors']
                 (list of indices, default all sensors)
    - same seed -> identical arrays (np.array_equal)
    - correlation=0.9 must be visible in the generated data (>0.7 empirical)

Baseline assumption (architect plan B0): normal operation = low-frequency sine
wave + Gaussian noise. Tests 4-6 additionally compare an attack run against a
paired 'none' run with the same seed: because generation is deterministic under
a fixed seed (assertion 8), the difference isolates the injected delta_x and is
robust to the unspecified baseline signal. Noise-level verification (test 10)
uses the first-difference estimator, valid for a smooth (low-frequency) signal.
"""

import numpy as np
import pytest

from src.data.synthetic import SensorDataGenerator

N_SENSORS = 5
NOISE = 0.1
T = 20000  # large T for statistical stability (contract requirement)
SEED = 42


def _sg(n_sensors=N_SENSORS, sampling_rate=1.0, noise_level=NOISE,
        correlation=0.0, seed=SEED):
    return SensorDataGenerator(n_sensors=n_sensors, sampling_rate=sampling_rate,
                               noise_level=noise_level, correlation=correlation,
                               seed=seed)


def _none_run(sg, n_timesteps=T):
    """Paired attack-free run with the same seed; isolates delta_x additively."""
    return sg.generate(n_timesteps=n_timesteps, attack_type='none')[0]


def _assert_int_like(x):
    assert isinstance(x, (int, np.integer)), f"expected int, got {type(x)}"


# ---------------------------------------------------------------------------
# 1. Shape + metadata
# ---------------------------------------------------------------------------

def test_returns_shape_and_metadata():
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type='step', t0=None, c=2.0)

    # (1) shape == (n_timesteps, n_sensors), float
    assert isinstance(data, np.ndarray)
    assert data.shape == (T, N_SENSORS)
    assert data.dtype.kind == 'f'

    # (2) metadata keys and values
    assert isinstance(meta, dict)
    assert meta['attack_type'] == 'step'
    assert meta['t0'] == T // 2
    _assert_int_like(meta['t0'])
    assert meta['n_timesteps'] == T
    assert meta['n_sensors'] == N_SENSORS
    assert meta['noise_level'] == NOISE
    assert meta['seed'] == SEED
    assert isinstance(meta['attack_params'], dict)
    assert meta['attack_params'].get('c') == 2.0
    cp = meta['changepoints']
    assert isinstance(cp, list) and len(cp) > 0  # non-empty for attack types
    assert all(isinstance(i, (int, np.integer)) for i in cp)
    assert all(0 <= i < T for i in cp)


@pytest.mark.parametrize('attack_type', ['none', 'step', 'ramp', 'periodic', 'coordinated'])
def test_metadata_for_all_attack_types(attack_type):
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type=attack_type)
    assert data.shape == (T, N_SENSORS)
    assert meta['attack_type'] == attack_type
    assert meta['t0'] == T // 2
    _assert_int_like(meta['t0'])
    assert isinstance(meta['changepoints'], list)
    if attack_type != 'none':
        assert len(meta['changepoints']) > 0


# ---------------------------------------------------------------------------
# 3. attack_type='none' -> no shift
# ---------------------------------------------------------------------------

def test_attack_none_no_shift():
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type='none')
    assert meta['attack_type'] == 'none'
    first = np.mean(data[:T // 2])
    second = np.mean(data[T // 2:])
    assert abs(first - second) < 3.0 * NOISE


# ---------------------------------------------------------------------------
# 4. step attack: mean shift ~= c after t0, no shift before t0
# ---------------------------------------------------------------------------

def test_step_attack_shift():
    c = 2.0
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type='step', t0=None, c=c)
    t0 = meta['t0']
    assert t0 == T // 2

    none_data = _none_run(_sg())
    dev = data - none_data

    for s in range(N_SENSORS):
        # mean shift after t0 ~= c per attacked sensor (tolerance ~0.5)
        shift = np.mean(data[t0:, s]) - np.mean(data[:t0, s])
        assert abs(shift - c) < 0.5, f"sensor {s}: within-run shift {shift}"
        # NO shift before t0: pre-t0 level ~ baseline, i.e. ~0 residual
        assert abs(np.mean(data[:t0, s])) < 0.5, f"sensor {s}: pre-t0 mean"
        # paired isolation: deviation vs the none-run equals the step exactly
        assert abs(np.mean(dev[t0:, s]) - c) < 0.5, f"sensor {s}: paired shift"
        assert abs(np.mean(dev[:t0, s])) < 0.3, f"sensor {s}: paired pre-t0"


# ---------------------------------------------------------------------------
# 5. ramp attack: endpoint deviation ~= r * (T - t0)
# ---------------------------------------------------------------------------

def test_ramp_attack_endpoint_deviation():
    r = 0.05
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type='ramp', t0=None, r=r)
    t0 = meta['t0']
    expected = r * (T - 1 - t0)  # delta_x at the final sample

    none_data = _none_run(_sg())
    dev = data - none_data

    for s in range(N_SENSORS):
        # paired: deviation at the final sample equals the ramp height
        assert abs(dev[-1, s] - expected) <= 0.10 * expected + 1.0, f"sensor {s}"
        # within-run: endpoint deviation vs the pre-t0 level (linear fit)
        tt = np.arange(t0, T)
        slope = np.polyfit(tt, data[t0:, s], 1)[0]
        end_dev = slope * (T - 1 - t0)
        assert abs(end_dev - expected) <= 0.15 * expected + 0.5, f"sensor {s}"


# ---------------------------------------------------------------------------
# 6. periodic attack: post-t0 residual amplitude ~= A
# ---------------------------------------------------------------------------

def test_periodic_attack_amplitude():
    A = 2.0
    f = 0.01
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type='periodic', t0=None, A=A, f=f)
    t0 = meta['t0']

    none_data = _none_run(_sg())
    dev = data - none_data

    for s in range(N_SENSORS):
        post = dev[t0:, s]
        amp = (np.max(post) - np.min(post)) / 2.0
        assert abs(amp - A) < 0.5, f"sensor {s}: amplitude {amp}"
        # sanity: the post-t0 oscillation is real, not just noise
        assert amp > 1.0, f"sensor {s}: no oscillation detected"


# ---------------------------------------------------------------------------
# 7. coordinated attack: only the listed sensors show the shift
# ---------------------------------------------------------------------------

def test_coordinated_attack_sensor_selection():
    n_sensors = 6
    attacked = [0, 1, 2]
    sg = _sg(n_sensors=n_sensors)
    data, meta = sg.generate(n_timesteps=T, attack_type='coordinated', t0=None,
                             sensors=attacked, c=1.0)
    t0 = meta['t0']
    assert meta['attack_params']['sensors'] == attacked

    for s in range(n_sensors):
        shift = np.mean(data[t0:, s]) - np.mean(data[:t0, s])
        if s in attacked:
            assert abs(shift - 1.0) < 0.5, f"attacked sensor {s}: shift {shift}"
        else:
            assert abs(shift) < 0.3, f"clean sensor {s}: shift {shift}"


# ---------------------------------------------------------------------------
# 8. same seed -> identical arrays
# ---------------------------------------------------------------------------

def test_same_seed_identical_arrays():
    d1, _ = _sg().generate(n_timesteps=T, attack_type='step', t0=None, c=2.0)
    d2, _ = _sg().generate(n_timesteps=T, attack_type='step', t0=None, c=2.0)
    assert np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# 9. correlation structure is respected
# ---------------------------------------------------------------------------

def test_correlated_sensors():
    n_sensors = 2
    T_corr = 30000
    sg = _sg(n_sensors=n_sensors, correlation=0.9)
    data, meta = sg.generate(n_timesteps=T_corr, attack_type='none')
    assert data.shape == (T_corr, n_sensors)
    corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    assert corr > 0.7, f"empirical correlation too low: {corr}"


# ---------------------------------------------------------------------------
# 10. noise_level is respected (residual std ~ noise_level, within 2x)
# ---------------------------------------------------------------------------

def test_noise_level_respected():
    sg = _sg()
    data, meta = sg.generate(n_timesteps=T, attack_type='none')
    for s in range(N_SENSORS):
        # First-difference estimator: for a smooth underlying signal + iid
        # noise, std(diff)/sqrt(2) ~ noise_level. Valid within 2x.
        est = np.std(np.diff(data[:, s])) / np.sqrt(2.0)
        assert NOISE / 2.0 <= est <= 2.0 * NOISE, f"sensor {s}: noise est {est}"
