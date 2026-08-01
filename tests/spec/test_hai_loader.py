# LOCKED — do not modify (TDD contract)

"""LOCKED contract tests for ``src.data.hai_loader.HAILoader`` (B0).

RED phase (now): these tests MUST fail because ``src/`` does not exist yet.
GREEN phase (after B0): they must pass UNCHANGED against the implementation.
Do not relax or weaken these assertions. This file is an immutable contract.

Public API under test (exact):
    HAILoader(data_dir='data/hai/raw/hai-20.07', val_fraction=0.2)
    train, val, test = hl.load()

Split semantics:
    - train1.csv + train2.csv -> train + val (val carved from the END of the
      train portion, temporal, time-ordered)
    - test1.csv + test2.csv -> test
    - Each split exposes .X (float, (T, 59)), .y (int, (T,)), .timestamps
      (int, (T,)) — 59 sensor columns only; 'time' and 'attack*' columns are
      excluded from X. y is the 'attack' label column. timestamps are epoch
      seconds parsed from the 'time' column.
    - z-score normalization: train.X per-column mean ~0, std ~1.
    - Local files only — the loader must NOT download anything.

Schema facts (verified against the real file header):
    - 64 columns: time + 59 sensors + attack + attack_P1 + attack_P2 + attack_P3
    - delimiter ';' ; time format 'YYYY-MM-DD HH:MM:SS'
    - row counts: train1 309600, train2 241200, test1 291600, test2 153000

TIMELINE NOTE (verified from real data) — deliberate deviation from the draft
contract's assertion 4. Real chronology of the HAI 20.07 files:
    train1 [2019-09-11 .. 2019-09-15] < test1 [2019-10-29 .. 2019-11-01] <
    train2 [2019-11-01 .. 2019-11-04] < test2 [2019-11-04 .. 2019-11-06]
The test files are NOT chronologically after the train files, so the draft
assertions `test.timestamps[0] > train.timestamps[-1]` and
`test.timestamps[0] > val.timestamps[-1]` are IMPOSSIBLE on real data. They are
replaced by the satisfiable no-leakage properties:
    - val.timestamps[0] > train.timestamps[-1]   (val is a strict suffix)
    - all three splits are pairwise disjoint in timestamps (no shared rows)
    - row-count integrity: every file row lands in exactly one split
"""

import os

import numpy as np
import pytest

from src.data.hai_loader import HAILoader

DATA_DIR = 'data/hai/raw/hai-20.07'
TRAIN_FILES = ['train1.csv', 'train2.csv']
TEST_FILES = ['test1.csv', 'test2.csv']
N_FEATURES = 59
VAL_FRACTION = 0.2
EXPECTED_TRAIN_ROWS = 309600 + 241200  # train1.csv + train2.csv
EXPECTED_TEST_ROWS = 291600 + 153000   # test1.csv + test2.csv


@pytest.fixture(scope='module')
def loaded():
    """Load the HAI data once per session (heavy: ~1 GB of CSV)."""
    hl = HAILoader(data_dir=DATA_DIR, val_fraction=VAL_FRACTION)
    return hl.load()


def _check_split(split, name):
    X, y, ts = split.X, split.y, split.timestamps
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert isinstance(ts, np.ndarray)
    assert X.ndim == 2, f"{name}: X must be 2D, got ndim {X.ndim}"
    assert X.shape[1] == N_FEATURES, f"{name}: expected 59 sensor columns, got {X.shape[1]}"
    t_len = len(y)
    assert X.shape[0] == t_len, f"{name}: X rows != y length"
    assert len(ts) == t_len, f"{name}: timestamps length != y length"
    assert t_len >= 1, f"{name}: split is empty"
    return X, y, ts


# ---------------------------------------------------------------------------
# 1. Three splits with correct shapes
# ---------------------------------------------------------------------------

def test_returns_three_splits_with_expected_shapes(loaded):
    train, val, test = loaded
    for name, split in [('train', train), ('val', val), ('test', test)]:
        X, y, ts = _check_split(split, name)
        assert X.dtype.kind == 'f', f"{name}: X dtype {X.dtype} not float"
        assert y.dtype.kind in 'iu', f"{name}: y dtype {y.dtype} not integer"
        assert ts.dtype.kind in 'iu', f"{name}: timestamps dtype {ts.dtype} not integer"


# ---------------------------------------------------------------------------
# 2. Non-empty splits, binary labels
# ---------------------------------------------------------------------------

def test_labels_are_binary(loaded):
    train, val, test = loaded
    for name, split in [('train', train), ('val', val), ('test', test)]:
        _, y, _ = _check_split(split, name)
        assert np.all(np.isin(y, [0, 1])), f"{name}: y has values outside {{0, 1}}"


# ---------------------------------------------------------------------------
# 3. Timestamps strictly increasing within each split
# ---------------------------------------------------------------------------

def test_timestamps_strictly_increasing_within_split(loaded):
    train, val, test = loaded
    for name, split in [('train', train), ('val', val), ('test', test)]:
        ts = split.timestamps
        assert np.all(np.diff(ts) > 0), f"{name}: timestamps not strictly increasing"


# ---------------------------------------------------------------------------
# 4. Temporal split, no leakage
# ---------------------------------------------------------------------------

def test_temporal_split_no_leakage(loaded):
    train, val, test = loaded
    # val is carved from the END of the train portion -> strictly after train
    assert val.timestamps[0] > train.timestamps[-1]
    # no shared timestamps between any pair of splits (no row duplication)
    assert np.intersect1d(train.timestamps, val.timestamps).size == 0
    assert np.intersect1d(train.timestamps, test.timestamps).size == 0
    assert np.intersect1d(val.timestamps, test.timestamps).size == 0
    # integrity: every row of every source file lands in exactly one split
    assert len(train.timestamps) + len(val.timestamps) == EXPECTED_TRAIN_ROWS
    assert len(test.timestamps) == EXPECTED_TEST_ROWS


# ---------------------------------------------------------------------------
# 5. z-score normalization on train.X
# ---------------------------------------------------------------------------

def test_train_zscore_normalization(loaded):
    train, _, _ = loaded
    X = train.X
    col_means = np.mean(X, axis=0)
    col_stds = np.std(X, axis=0)
    assert np.all(np.abs(col_means) < 0.05), f"train.X column means not ~0: {col_means}"
    assert np.all(np.abs(col_stds - 1.0) < 0.1), f"train.X column stds not ~1: {col_stds}"


# ---------------------------------------------------------------------------
# 6. dtype checks (folded into test 1 above)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Local data dir resolves without any network access
# ---------------------------------------------------------------------------

def test_local_data_dir_no_network():
    assert os.path.isdir(DATA_DIR), f"local data dir missing: {DATA_DIR}"
    for f in TRAIN_FILES + TEST_FILES:
        path = os.path.join(DATA_DIR, f)
        assert os.path.isfile(path), f"local file missing: {path}"
