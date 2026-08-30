from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import registry

DATASET = "wadi"
SAMPLING_INTERVAL_S = 1
VAL_HOURS = 36

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "WADI" / "WADI.A1_9 Oct 2017"
_14DAYS_PATH = _RAW_DIR / "WADI_14days.csv"
_ATTACKDATA_PATH = _RAW_DIR / "WADI_attackdata.csv"
_ATTACK_DESC_PATH = _RAW_DIR / "attack_description.xlsx"

_UNC_PREFIX = "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\"
_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S.%f %p"


def _read_wadi_csv(path: Path, skiprows: int) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=skiprows)
    df.columns = [str(c).replace(_UNC_PREFIX, "").strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str), format=_DATETIME_FORMAT
    )
    sensor_cols = [c for c in df.columns if c not in ("Row", "Date", "Time", "timestamp")]
    out = df[["timestamp"] + sensor_cols].copy()
    for col in sensor_cols:
        out[col] = out[col].astype(np.float32)
    return out


def _correct_attack_date(d: pd.Timestamp) -> pd.Timestamp:
    # Source typos: row 9 dated 1947-10-10 (year), rows 24-28 dated 2017-07-11
    # (month). The attack window is 2017-10-09 .. 2017-10-11.
    if d.year == 1947 and d.month == 10 and d.day == 10:
        return d.replace(year=2017)
    if d.year == 2017 and d.month == 7 and d.day == 11:
        return d.replace(month=10)
    return d


def _build_attack_intervals(data_start: pd.Timestamp, data_end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    df = pd.read_excel(_ATTACK_DESC_PATH, header=4)
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for _, row in df.iterrows():
        sno = row["S.No"]
        start_raw = row["Start Time"]
        end_raw = row["End Time"]
        if pd.isna(sno) or pd.isna(start_raw) or pd.isna(end_raw):
            continue
        date = _correct_attack_date(pd.Timestamp(row["Date"]))
        start_time = str(start_raw).replace(".", ":")
        end_time = str(end_raw).replace(".", ":")
        start = pd.Timestamp(date.date()) + pd.to_timedelta(start_time)
        end = pd.Timestamp(date.date()) + pd.to_timedelta(end_time)
        if end <= start:
            end += pd.Timedelta(days=1)
        intervals.append((start, end, int(sno)))
    return intervals


def _apply_labels(ts: pd.Series, intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]]) -> Tuple[np.ndarray, pd.array]:
    label = np.zeros(len(ts), dtype=np.int8)
    attack_id = np.full(len(ts), np.nan)
    for start, end, aid in intervals:
        mask = (ts >= start) & (ts <= end)
        label[mask] = 1
        attack_id[mask] = aid
    return label, pd.array(attack_id, dtype="Int32")


def _constant_columns(df: pd.DataFrame, sensor_cols: List[str]) -> List[str]:
    return [c for c in sensor_cols if df[c].nunique(dropna=True) <= 1]


def build_store() -> dict:
    normal = _read_wadi_csv(_14DAYS_PATH, skiprows=4)
    attack = _read_wadi_csv(_ATTACKDATA_PATH, skiprows=0)
    normal = normal.sort_values("timestamp").reset_index(drop=True)
    attack = attack.sort_values("timestamp").reset_index(drop=True)

    intervals = _build_attack_intervals(attack["timestamp"].min(), attack["timestamp"].max())
    label, attack_id = _apply_labels(attack["timestamp"], intervals)
    attack["label"] = label
    attack["attack_id"] = attack_id
    normal["label"] = np.zeros(len(normal), dtype=np.int8)
    normal["attack_id"] = pd.array(np.full(len(normal), np.nan), dtype="Int32")

    val_size = VAL_HOURS * 3600
    train = normal.iloc[:-val_size].reset_index(drop=True)
    val = normal.iloc[-val_size:].reset_index(drop=True)
    test = attack

    sensor_cols = [c for c in train.columns if c not in ("timestamp", "label", "attack_id")]
    out_dir = registry.PROCESSED_DIR / DATASET
    splits = {"train": train, "val": val, "test": test}
    for split_name, df in splits.items():
        registry._write_parquet(df, out_dir / f"{split_name}.parquet")

    meta = {
        "dataset": DATASET,
        "sampling_interval_s": SAMPLING_INTERVAL_S,
        "sensor_columns": sensor_cols,
        "constant_columns": _constant_columns(train, sensor_cols),
        "timestamps": "naive local SUTD time; no tz shift",
        "splits": {
            split_name: {
                "path": str(out_dir / f"{split_name}.parquet"),
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "rows": int(len(df)),
                "label_source": "attack_description.xlsx intervals",
            }
            for split_name, df in splits.items()
        },
        "label_source": "attack_description.xlsx intervals",
        "raw_file_checksums": {
            "WADI_14days.csv": registry._sha256(_14DAYS_PATH),
            "WADI_attackdata.csv": registry._sha256(_ATTACKDATA_PATH),
            "attack_description.xlsx": registry._sha256(_ATTACK_DESC_PATH),
        },
        "other_releases": {
            "wadi_a2": {"path": "data/WADI/WADI.A2_19 Nov 2019", "format": "csv"},
            "wadi_a3": {"path": "data/WADI/WaDi.A3_Dec 2023", "format": "csv"},
        },
    }
    registry._write_meta(meta, out_dir / "meta.json")
    return meta


def load(split: str) -> pd.DataFrame:
    return registry._load_parquet(DATASET, split)


def get_meta() -> dict:
    return registry._read_meta(DATASET)