"""Build the consolidated HAI parquet store (raw values) + meta.json.

Replicates ONLY the temporal split logic of src/data/hai_loader.py (train/val
boundary at the last 20% of the combined train files; test = combined test
files) but exports RAW, unnormalized sensor values. Z-scoring stays at load
time in src/data/hai_loader.py (fit on train only).

Run:  python scripts/build_hai_store.py
"""

from pathlib import Path
import hashlib
import json
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "hai" / "raw" / "hai-20.07"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hai"

VAL_FRACTION = 0.2
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
NON_SENSOR_COLS = ("time", "timestamps", "timestamp", "attack", "attack_P1", "attack_P2", "attack_P3")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["timestamp"] = pd.to_datetime(df["time"], format=TIMESTAMP_FORMAT)
    return df


def _sensor_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in NON_SENSOR_COLS]


def _build_attack_id(df: pd.DataFrame) -> pd.array:
    """Map phase flags to a single attack_id via bitmask (P1=4, P2=2, P3=1).

    attack == 1 iff any phase flag is set (verified across all four source
    files), so the bitmask is a faithful, lossless encoding of the phase
    combination. Normal rows (attack == 0) get NaN.
    """
    raw = (
        df["attack_P1"].astype(np.int8) * 4
        + df["attack_P2"].astype(np.int8) * 2
        + df["attack_P3"].astype(np.int8)
    )
    attack_id = np.where(df["attack"].astype(int) == 1, raw, np.nan)
    return pd.array(attack_id, dtype="Int32")


def _canonicalize(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    out["label"] = df["attack"].astype(np.int8)
    out["attack_id"] = _build_attack_id(df)
    for col in sensor_cols:
        out[col] = df[col].astype(np.float32)
    return out


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_meta(meta: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    train1 = _read_raw(RAW_DIR / "train1.csv")
    train2 = _read_raw(RAW_DIR / "train2.csv")
    test1 = _read_raw(RAW_DIR / "test1.csv")
    test2 = _read_raw(RAW_DIR / "test2.csv")

    sensor_cols = _sensor_columns(train1)
    if len(sensor_cols) != 59:
        raise ValueError(f"expected 59 sensor columns, got {len(sensor_cols)}")

    # Replicate the loader's split logic exactly (see src/data/hai_loader.py).
    train_df = pd.concat([train1, train2], ignore_index=True)
    test_df = pd.concat([test1, test2], ignore_index=True)
    val_size = int(len(train_df) * VAL_FRACTION)
    train_split_df = train_df.iloc[:-val_size]
    val_split_df = train_df.iloc[-val_size:]

    # Loader boundary guard: ensure val starts strictly after train ends.
    if len(val_split_df) > 0 and len(train_split_df) > 0:
        if val_split_df["timestamp"].iloc[0] <= train_split_df["timestamp"].iloc[-1]:
            min_val_ts = train_split_df["timestamp"].iloc[-1] + pd.Timedelta(seconds=1)
            val_split_df = val_split_df[val_split_df["timestamp"] > min_val_ts]

    splits = {
        "train": _canonicalize(train_split_df, sensor_cols),
        "val": _canonicalize(val_split_df, sensor_cols),
        "test": _canonicalize(test_df, sensor_cols),
    }

    for split_name, df in splits.items():
        _write_parquet(df, OUT_DIR / f"{split_name}.parquet")

    meta = {
        "dataset": "hai",
        "sampling_interval_s": 1,
        "sensor_columns": sensor_cols,
        "constant_columns": [
            col for col in sensor_cols
            if train_split_df[col].std() < 1e-12
        ],
        "timestamps": "naive local HAI plant time; no tz shift",
        "label_source": "HAI attack flags: attack + attack_P1..P3 (label = 1 iff attack flag set; attack_id = bitmask of phase flags P1=4, P2=2, P3=1)",
        "zscore_note": "HAI z-scoring is done at load time by src/data/hai_loader.py (fit on train only); store holds raw values",
        "splits": {
            split_name: {
                "path": str(OUT_DIR / f"{split_name}.parquet"),
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "rows": int(len(df)),
                "label_source": "HAI attack flags: attack + attack_P1..P3",
            }
            for split_name, df in splits.items()
        },
        "raw_file_checksums": {
            "train1.csv": _sha256(RAW_DIR / "train1.csv"),
            "train2.csv": _sha256(RAW_DIR / "train2.csv"),
            "test1.csv": _sha256(RAW_DIR / "test1.csv"),
            "test2.csv": _sha256(RAW_DIR / "test2.csv"),
        },
    }
    _write_meta(meta, OUT_DIR / "meta.json")
    print("Wrote HAI store to", OUT_DIR)


if __name__ == "__main__":
    main()