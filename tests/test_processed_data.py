"""Acceptance tests for the Wave 1 consolidated parquet store.

Tests run against the exported store via ``load_dataset`` (fast parquet reads;
the slow raw xlsx/csv builds are NOT re-run here). One test class per dataset.
The canonical column contract, monotonicity, split row counts, label sanity,
cross-split uniqueness, and meta checksums are asserted per the documented
store-level truths in .agent-tasks/architect/DATA_PREP_PLAN.md section 6 and
data/eda/SWaT_label_crosscheck.md.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import registry

SPLITS = ("train", "val", "test")


def _assert_strict_monotonic(ts: pd.Series) -> None:
    assert ts.is_monotonic_increasing, "timestamps not monotonic increasing"
    assert ts.is_unique, "timestamps contain duplicates"


def _assert_label_contract(df: pd.DataFrame) -> None:
    assert str(df["label"].dtype) == "int8", f"label dtype {df['label'].dtype}"
    assert set(df["label"].unique()).issubset({0, 1}), "label values outside {0, 1}"
    assert str(df["attack_id"].dtype) == "Int32", f"attack_id dtype {df['attack_id'].dtype}"
    normal_mask = df["label"] == 0
    attack_mask = df["label"] == 1
    assert df.loc[normal_mask, "attack_id"].isna().all(), "normal rows must have NaN attack_id"
    assert df.loc[attack_mask, "attack_id"].notna().all(), "attack rows must have int attack_id"


def _assert_schema(df: pd.DataFrame, sensor_cols: list) -> None:
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp not datetime64"
    _assert_label_contract(df)
    for col in sensor_cols:
        assert str(df[col].dtype) == "float32", f"{col} dtype {df[col].dtype}"


def _assert_cadence(df: pd.DataFrame, expected: pd.Timedelta) -> None:
    diffs = df["timestamp"].diff().dropna()
    assert len(diffs) > 0, "empty split"
    assert diffs.mode().iloc[0] == expected, "cadence mismatch"


def _assert_meta_checksums(meta: dict) -> None:
    checksums = meta.get("raw_file_checksums", {})
    assert checksums, "meta.json missing raw_file_checksums"
    for key, value in checksums.items():
        assert isinstance(value, str) and len(value) == 64, f"bad sha256 for {key}"


@pytest.fixture(scope="module")
def swat():
    return {s: registry.load_dataset("swat", s) for s in SPLITS}


@pytest.fixture(scope="module")
def wadi():
    return {s: registry.load_dataset("wadi", s) for s in SPLITS}


@pytest.fixture(scope="module")
def batadal():
    return {s: registry.load_dataset("batadal", s) for s in SPLITS}


@pytest.fixture(scope="module")
def hai():
    return {s: registry.load_dataset("hai", s) for s in SPLITS}


class TestSWaT:
    def test_row_counts(self, swat):
        assert len(swat["train"]) == 365400
        assert len(swat["val"]) == 129600
        assert len(swat["test"]) == 449919
        assert len(swat["train"]) + len(swat["val"]) == 495000

    def test_sensor_count(self, swat):
        assert len(registry.get_meta("swat")["sensor_columns"]) == 51

    def test_train_val_labels_zero(self, swat):
        for s in ("train", "val"):
            assert (swat[s]["label"] == 0).all()

    def test_test_attack_rows(self, swat):
        n_attack = int((swat["test"]["label"] == 1).sum())
        assert n_attack >= 50000
        assert n_attack == 53885

    def test_strict_monotonic(self, swat):
        for s in SPLITS:
            _assert_strict_monotonic(swat[s]["timestamp"])

    def test_cadence(self, swat):
        for s in SPLITS:
            _assert_cadence(swat[s], pd.Timedelta(seconds=1))

    def test_raw_values_unnormalized(self, swat):
        lit = swat["train"]["LIT101"]
        assert lit.min() > 50
        assert lit.max() < 5000

    def test_schema(self, swat):
        sensor_cols = registry.get_meta("swat")["sensor_columns"]
        for s in SPLITS:
            _assert_schema(swat[s], sensor_cols)

    def test_no_cross_split_overlap(self, swat):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(swat[a]["timestamp"]) & set(swat[b]["timestamp"])
            assert not shared

    def test_meta_checksums(self):
        _assert_meta_checksums(registry.get_meta("swat"))


class TestWadi:
    def test_row_counts(self, wadi):
        assert len(wadi["train"]) == 1080001
        assert len(wadi["val"]) == 129600
        assert len(wadi["test"]) == 172801

    def test_sensor_count(self, wadi):
        assert len(registry.get_meta("wadi")["sensor_columns"]) == 127

    def test_constant_columns(self, wadi):
        assert len(registry.get_meta("wadi")["constant_columns"]) == 34

    def test_train_labels_zero(self, wadi):
        assert (wadi["train"]["label"] == 0).all()

    def test_test_attack(self, wadi):
        n_attack = int((wadi["test"]["label"] == 1).sum())
        assert n_attack > 0
        attack_ids = wadi["test"]["attack_id"].dropna().unique()
        assert all(1 <= int(a) <= 15 for a in attack_ids)

    def test_strict_monotonic(self, wadi):
        for s in SPLITS:
            _assert_strict_monotonic(wadi[s]["timestamp"])

    def test_cadence(self, wadi):
        for s in SPLITS:
            _assert_cadence(wadi[s], pd.Timedelta(seconds=1))

    def test_cross_split_overlap(self, wadi):
        shared_tv = set(wadi["train"]["timestamp"]) & set(wadi["val"]["timestamp"])
        shared_tt = set(wadi["train"]["timestamp"]) & set(wadi["test"]["timestamp"])
        shared_vt = set(wadi["val"]["timestamp"]) & set(wadi["test"]["timestamp"])
        assert len(shared_tv) == 0
        assert len(shared_tt) == 0
        assert len(shared_vt) <= 1  # single shared boundary row is inherent to source

    def test_schema(self, wadi):
        sensor_cols = registry.get_meta("wadi")["sensor_columns"]
        for s in SPLITS:
            _assert_schema(wadi[s], sensor_cols)

    def test_meta_checksums(self):
        _assert_meta_checksums(registry.get_meta("wadi"))


class TestBatadal:
    def test_row_counts(self, batadal):
        assert len(batadal["train"]) == 7321
        assert len(batadal["val"]) == 1440
        assert len(batadal["test"]) == 4177
        assert len(batadal["train"]) + len(batadal["val"]) == 8761

    def test_sensor_count(self, batadal):
        assert len(registry.get_meta("batadal")["sensor_columns"]) == 43

    def test_hourly_cadence(self, batadal):
        assert registry.get_meta("batadal")["sampling_interval_s"] == 3600
        for s in SPLITS:
            _assert_cadence(batadal[s], pd.Timedelta(hours=1))

    def test_train_labels_zero(self, batadal):
        assert (batadal["train"]["label"] == 0).all()

    def test_test_attack(self, batadal):
        n_attack = int((batadal["test"]["label"] == 1).sum())
        assert n_attack > 0

    def test_strict_monotonic(self, batadal):
        for s in SPLITS:
            _assert_strict_monotonic(batadal[s]["timestamp"])

    def test_no_cross_split_overlap(self, batadal):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(batadal[a]["timestamp"]) & set(batadal[b]["timestamp"])
            assert len(shared) == 0

    def test_schema(self, batadal):
        sensor_cols = registry.get_meta("batadal")["sensor_columns"]
        for s in SPLITS:
            _assert_schema(batadal[s], sensor_cols)

    def test_meta_checksums(self):
        _assert_meta_checksums(registry.get_meta("batadal"))


class TestHAI:
    def test_row_counts(self, hai):
        assert len(hai["train"]) == 440640
        assert len(hai["val"]) == 110160
        assert len(hai["test"]) == 444600

    def test_sensor_count(self, hai):
        assert len(registry.get_meta("hai")["sensor_columns"]) == 59

    def test_constant_columns(self, hai):
        assert len(registry.get_meta("hai")["constant_columns"]) == 8

    def test_attack_rows(self, hai):
        expected = {"train": 387, "val": 389, "test": 17527}
        for s in SPLITS:
            assert int((hai[s]["label"] == 1).sum()) == expected[s]

    def test_label_attack_consistency(self, hai):
        for s in SPLITS:
            _assert_label_contract(hai[s])

    def test_strict_monotonic(self, hai):
        for s in SPLITS:
            _assert_strict_monotonic(hai[s]["timestamp"])

    def test_cadence(self, hai):
        for s in SPLITS:
            _assert_cadence(hai[s], pd.Timedelta(seconds=1))

    def test_raw_values_unnormalized(self, hai):
        lit = hai["train"]["P1_LIT01"]
        assert lit.min() > 300
        assert lit.max() < 600

    def test_no_cross_split_overlap(self, hai):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(hai[a]["timestamp"]) & set(hai[b]["timestamp"])
            assert len(shared) == 0

    def test_schema(self, hai):
        sensor_cols = registry.get_meta("hai")["sensor_columns"]
        for s in SPLITS:
            _assert_schema(hai[s], sensor_cols)

    def test_meta_checksums(self):
        meta = registry.get_meta("hai")
        _assert_meta_checksums(meta)
        assert len(meta["raw_file_checksums"]) == 4