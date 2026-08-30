"""Validate the consolidated parquet store against the Wave 1 acceptance criteria.

Loads each dataset's splits through the registry, asserts the documented
store-level truths (see .agent-tasks/architect/DATA_PREP_PLAN.md section 6 and
data/eda/SWaT_label_crosscheck.md), prints a per-dataset PASS/FAIL table, and
exits non-zero if any check fails.

Run:  python scripts/validate_processed.py
"""

from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import registry  # noqa: E402

SPLITS = ("train", "val", "test")


def _check(name: str, fn: Callable[[], None]) -> Tuple[str, bool, str]:
    try:
        fn()
        return name, True, ""
    except AssertionError as exc:
        return name, False, str(exc)


def _run_checks(dataset: str, checks: List[Tuple[str, Callable[[], None]]]) -> List[Tuple[str, bool, str]]:
    results = []
    for name, fn in checks:
        results.append(_check(f"{dataset}: {name}", fn))
    return results


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


def _assert_sensor_dtypes(df: pd.DataFrame, sensor_cols: List[str]) -> None:
    for col in sensor_cols:
        assert str(df[col].dtype) == "float32", f"{col} dtype {df[col].dtype}"


def _assert_schema_dtypes(df: pd.DataFrame, sensor_cols: List[str]) -> None:
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp not datetime64"
    _assert_label_contract(df)
    _assert_sensor_dtypes(df, sensor_cols)


def _assert_cadence(df: pd.DataFrame, expected: pd.Timedelta) -> None:
    diffs = df["timestamp"].diff().dropna()
    assert len(diffs) > 0, "empty split"
    mode = diffs.mode().iloc[0]
    assert mode == expected, f"cadence mode {mode} != {expected}"


def _assert_meta_checksums(meta: dict) -> None:
    checksums = meta.get("raw_file_checksums", {})
    assert checksums, "meta.json missing raw_file_checksums"
    for key, value in checksums.items():
        assert isinstance(value, str) and len(value) == 64, f"bad sha256 for {key}"


def _assert_reload_equality(dataset: str, split: str) -> None:
    via_loader = registry.load_dataset(dataset, split)
    path = registry.PROCESSED_DIR / dataset / f"{split}.parquet"
    direct = pd.read_parquet(path)
    pd.testing.assert_frame_equal(via_loader, direct)


def _swat_checks() -> List[Tuple[str, Callable[[], None]]]:
    meta = registry.get_meta("swat")
    splits = {s: registry.load_dataset("swat", s) for s in SPLITS}
    sensor_cols = meta["sensor_columns"]

    def row_counts():
        assert len(splits["train"]) == 365400, f"train rows {len(splits['train'])}"
        assert len(splits["val"]) == 129600, f"val rows {len(splits['val'])}"
        assert len(splits["test"]) == 449919, f"test rows {len(splits['test'])}"
        assert len(splits["train"]) + len(splits["val"]) == 495000, "train+val != 495000"

    def sensor_count():
        assert len(sensor_cols) == 51, f"sensor count {len(sensor_cols)}"

    def train_val_labels_zero():
        for s in ("train", "val"):
            assert (splits[s]["label"] == 0).all(), f"{s} has non-zero labels"

    def test_attack_rows():
        n_attack = int((splits["test"]["label"] == 1).sum())
        assert n_attack >= 50000, f"test attack rows {n_attack} < 50000"
        assert n_attack == 53885, f"test attack rows {n_attack} != 53885"

    def monotonic():
        for s in SPLITS:
            _assert_strict_monotonic(splits[s]["timestamp"])

    def cadence():
        for s in SPLITS:
            _assert_cadence(splits[s], pd.Timedelta(seconds=1))

    def raw_values():
        lit = splits["train"]["LIT101"]
        assert lit.min() > 50, f"LIT101 min {lit.min()} not raw-scale"
        assert lit.max() < 5000, f"LIT101 max {lit.max()} not raw-scale"

    def schema():
        for s in SPLITS:
            _assert_schema_dtypes(splits[s], sensor_cols)
            _assert_label_contract(splits[s])

    def no_cross_split_overlap():
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(splits[a]["timestamp"]) & set(splits[b]["timestamp"])
            assert not shared, f"swat {a}/{b} share {len(shared)} timestamps"

    def meta_ok():
        _assert_meta_checksums(meta)
        assert meta["sampling_interval_s"] == 1

    def reload():
        for s in SPLITS:
            _assert_reload_equality("swat", s)

    return [
        ("row counts", row_counts),
        ("sensor count", sensor_count),
        ("train/val labels all 0", train_val_labels_zero),
        ("test attack rows >= 50000 and == 53885", test_attack_rows),
        ("strict monotonic timestamps", monotonic),
        ("1 s cadence", cadence),
        ("raw values unnormalized", raw_values),
        ("column contract + dtypes", schema),
        ("no cross-split overlap", no_cross_split_overlap),
        ("meta checksums + cadence", meta_ok),
        ("parquet re-load equality", reload),
    ]


def _wadi_checks() -> List[Tuple[str, Callable[[], None]]]:
    meta = registry.get_meta("wadi")
    splits = {s: registry.load_dataset("wadi", s) for s in SPLITS}
    sensor_cols = meta["sensor_columns"]

    def row_counts():
        assert len(splits["train"]) == 1080001, f"train rows {len(splits['train'])}"
        assert len(splits["val"]) == 129600, f"val rows {len(splits['val'])}"
        assert len(splits["test"]) == 172801, f"test rows {len(splits['test'])}"

    def sensor_count():
        assert len(sensor_cols) == 127, f"sensor count {len(sensor_cols)}"

    def constant_cols():
        assert len(meta.get("constant_columns", [])) == 34, "expected 34 constant columns"

    def train_labels():
        assert (splits["train"]["label"] == 0).all(), "train has non-zero labels"

    def test_attack():
        n_attack = int((splits["test"]["label"] == 1).sum())
        assert n_attack > 0, "test has no attack rows"
        attack_ids = splits["test"]["attack_id"].dropna().unique()
        assert all(1 <= int(a) <= 15 for a in attack_ids), "attack_id outside 1..15"

    def monotonic():
        for s in SPLITS:
            _assert_strict_monotonic(splits[s]["timestamp"])

    def cadence():
        for s in SPLITS:
            _assert_cadence(splits[s], pd.Timedelta(seconds=1))

    def cross_split_overlap():
        shared_tv = set(splits["train"]["timestamp"]) & set(splits["val"]["timestamp"])
        shared_tt = set(splits["train"]["timestamp"]) & set(splits["test"]["timestamp"])
        shared_vt = set(splits["val"]["timestamp"]) & set(splits["test"]["timestamp"])
        assert len(shared_tv) == 0, f"train/val share {len(shared_tv)}"
        assert len(shared_tt) == 0, f"train/test share {len(shared_tt)}"
        assert len(shared_vt) <= 1, f"val/test share {len(shared_vt)} (allow 1 boundary)"

    def schema():
        for s in SPLITS:
            _assert_schema_dtypes(splits[s], sensor_cols)
            _assert_label_contract(splits[s])

    def meta_ok():
        _assert_meta_checksums(meta)
        assert meta["sampling_interval_s"] == 1

    def reload():
        for s in SPLITS:
            _assert_reload_equality("wadi", s)

    return [
        ("row counts", row_counts),
        ("sensor count", sensor_count),
        ("34 constant columns in meta", constant_cols),
        ("train labels all 0", train_labels),
        ("test attack rows > 0, attack_id in 1..15", test_attack),
        ("strict monotonic timestamps", monotonic),
        ("1 s cadence", cadence),
        ("cross-split overlap (<=1 boundary)", cross_split_overlap),
        ("column contract + dtypes", schema),
        ("meta keys + cadence", meta_ok),
        ("parquet re-load equality", reload),
    ]


def _batadal_checks() -> List[Tuple[str, Callable[[], None]]]:
    meta = registry.get_meta("batadal")
    splits = {s: registry.load_dataset("batadal", s) for s in SPLITS}
    sensor_cols = meta["sensor_columns"]

    def row_counts():
        assert len(splits["train"]) == 7321, f"train rows {len(splits['train'])}"
        assert len(splits["val"]) == 1440, f"val rows {len(splits['val'])}"
        assert len(splits["test"]) == 4177, f"test rows {len(splits['test'])}"
        assert len(splits["train"]) + len(splits["val"]) == 8761, "train+val != 8761"

    def sensor_count():
        assert len(sensor_cols) == 43, f"sensor count {len(sensor_cols)}"

    def hourly_cadence():
        assert meta["sampling_interval_s"] == 3600, "sampling_interval_s != 3600"
        for s in SPLITS:
            _assert_cadence(splits[s], pd.Timedelta(hours=1))

    def train_labels():
        assert (splits["train"]["label"] == 0).all(), "train has non-zero labels"

    def test_attack():
        n_attack = int((splits["test"]["label"] == 1).sum())
        assert n_attack > 0, "test has no attack rows"

    def monotonic():
        for s in SPLITS:
            _assert_strict_monotonic(splits[s]["timestamp"])

    def cross_split_overlap():
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(splits[a]["timestamp"]) & set(splits[b]["timestamp"])
            assert len(shared) == 0, f"batadal {a}/{b} share {len(shared)} timestamps"

    def schema():
        for s in SPLITS:
            _assert_schema_dtypes(splits[s], sensor_cols)
            _assert_label_contract(splits[s])

    def meta_ok():
        _assert_meta_checksums(meta)

    def reload():
        for s in SPLITS:
            _assert_reload_equality("batadal", s)

    return [
        ("row counts", row_counts),
        ("sensor count", sensor_count),
        ("hourly cadence (3600 s)", hourly_cadence),
        ("train labels all 0", train_labels),
        ("test attack rows > 0", test_attack),
        ("strict monotonic timestamps", monotonic),
        ("no cross-split overlap", cross_split_overlap),
        ("column contract + dtypes", schema),
        ("meta checksums", meta_ok),
        ("parquet re-load equality", reload),
    ]


def _hai_checks() -> List[Tuple[str, Callable[[], None]]]:
    meta = registry.get_meta("hai")
    splits = {s: registry.load_dataset("hai", s) for s in SPLITS}
    sensor_cols = meta["sensor_columns"]

    def row_counts():
        assert len(splits["train"]) == 440640, f"train rows {len(splits['train'])}"
        assert len(splits["val"]) == 110160, f"val rows {len(splits['val'])}"
        assert len(splits["test"]) == 444600, f"test rows {len(splits['test'])}"

    def sensor_count():
        assert len(sensor_cols) == 59, f"sensor count {len(sensor_cols)}"

    def constant_cols():
        assert len(meta.get("constant_columns", [])) == 8, "expected 8 constant columns"

    def attack_rows():
        expected = {"train": 387, "val": 389, "test": 17527}
        for s in SPLITS:
            n_attack = int((splits[s]["label"] == 1).sum())
            assert n_attack == expected[s], f"{s} attack rows {n_attack} != {expected[s]}"

    def label_attack_consistency():
        for s in SPLITS:
            _assert_label_contract(splits[s])

    def monotonic():
        for s in SPLITS:
            _assert_strict_monotonic(splits[s]["timestamp"])

    def cadence():
        for s in SPLITS:
            _assert_cadence(splits[s], pd.Timedelta(seconds=1))

    def raw_values():
        lit = splits["train"]["P1_LIT01"]
        assert lit.min() > 300, f"P1_LIT01 min {lit.min()} not raw-scale"
        assert lit.max() < 600, f"P1_LIT01 max {lit.max()} not raw-scale"

    def no_cross_split_overlap():
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(splits[a]["timestamp"]) & set(splits[b]["timestamp"])
            assert len(shared) == 0, f"hai {a}/{b} share {len(shared)} timestamps"

    def schema():
        for s in SPLITS:
            _assert_schema_dtypes(splits[s], sensor_cols)
            _assert_label_contract(splits[s])

    def meta_ok():
        _assert_meta_checksums(meta)
        assert meta["sampling_interval_s"] == 1
        assert len(meta["raw_file_checksums"]) == 4, "expected 4 raw CSV checksums"

    def reload():
        for s in SPLITS:
            _assert_reload_equality("hai", s)

    return [
        ("row counts", row_counts),
        ("sensor count", sensor_count),
        ("8 constant columns in meta", constant_cols),
        ("attack rows per split (387/389/17527)", attack_rows),
        ("label/attack_id consistency", label_attack_consistency),
        ("strict monotonic timestamps", monotonic),
        ("1 s cadence", cadence),
        ("raw values unnormalized", raw_values),
        ("no cross-split overlap", no_cross_split_overlap),
        ("column contract + dtypes", schema),
        ("meta checksums + cadence", meta_ok),
        ("parquet re-load equality", reload),
    ]


def _print_table(results: List[Tuple[str, bool, str]]) -> None:
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok and msg:
            print(f"         {msg}")


def main() -> int:
    all_results: Dict[str, List[Tuple[str, bool, str]]] = {
        "swat": _run_checks("swat", _swat_checks()),
        "wadi": _run_checks("wadi", _wadi_checks()),
        "batadal": _run_checks("batadal", _batadal_checks()),
        "hai": _run_checks("hai", _hai_checks()),
    }

    print("=" * 60)
    print("Processed store acceptance validation")
    print("=" * 60)
    overall_ok = True
    for dataset, results in all_results.items():
        passed = sum(1 for _, ok, _ in results if ok)
        total = len(results)
        overall_ok = overall_ok and passed == total
        print(f"\n[{dataset}] {passed}/{total} checks passed")
        _print_table(results)

    print("\n" + "=" * 60)
    if overall_ok:
        print("RESULT: ALL DATASETS PASS")
        return 0
    print("RESULT: FAILURE(S) DETECTED")
    return 1


if __name__ == "__main__":
    sys.exit(main())