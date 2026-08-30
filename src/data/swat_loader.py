from datetime import time as dtime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import registry

DATASET = "swat"
SAMPLING_INTERVAL_S = 1
VAL_HOURS = 36

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "SWaT" / "SWaT.A1 & A2 (Dec 2015)" / "Physical"
_ATTACK_LIST_PATH = Path(__file__).resolve().parents[2] / "data" / "SWaT" / "SWaT.A1 & A2 (Dec 2015)" / "List_of_attacks_Final.xlsx"
_NORMAL_PATH = _RAW_DIR / "SWaT_Dataset_Normal_v1.xlsx"
_ATTACK_PATH = _RAW_DIR / "SWaT_Dataset_Attack_v0.xlsx"
_UNUSED_NORMAL_V0_PATH = _RAW_DIR / "SWaT_Dataset_Normal_v0.xlsx"

_TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"


def _read_sensors(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["Timestamp"].str.strip(), format=_TIMESTAMP_FORMAT)
    sensor_cols = [c for c in df.columns if c not in ("timestamp", "Timestamp", "Normal/Attack")]
    out = df[["timestamp"] + sensor_cols].copy()
    for col in sensor_cols:
        out[col] = out[col].astype(np.float32)
    return out


def _time_to_timedelta(value) -> pd.Timedelta:
    if isinstance(value, dtime):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    parsed = pd.Timestamp(value)
    return parsed - parsed.normalize()


def _build_attack_intervals(data_start: pd.Timestamp, data_end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    df = pd.read_excel(_ATTACK_LIST_PATH)
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for _, row in df.iterrows():
        attack_id = row["Attack #"]
        start_raw = row["Start Time"]
        end_raw = row["End Time"]
        if pd.isna(attack_id) or pd.isna(start_raw) or pd.isna(end_raw):
            continue
        start = pd.Timestamp(start_raw)
        end = start.normalize() + _time_to_timedelta(end_raw)
        if end <= start:
            end += pd.Timedelta(days=1)
        # Source year typo: rows 37-41 are dated 2015-01-02 but belong to the
        # 2016-01-02 campaign window (data spans 2015-12-28 -> 2016-01-02).
        if start < data_start and (data_start - start) < pd.Timedelta(days=365):
            start += pd.DateOffset(years=1)
            end += pd.DateOffset(years=1)
        intervals.append((start, end, int(attack_id)))
    return intervals


def _apply_labels(ts: pd.Series, intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]]) -> Tuple[np.ndarray, pd.array]:
    label = np.zeros(len(ts), dtype=np.int8)
    attack_id = np.full(len(ts), np.nan)
    for start, end, aid in intervals:
        mask = (ts >= start) & (ts <= end)
        label[mask] = 1
        attack_id[mask] = aid
    return label, pd.array(attack_id, dtype="Int32")


def build_store() -> Path:
    normal = _read_sensors(_NORMAL_PATH)
    attack = _read_sensors(_ATTACK_PATH)
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
        "constant_columns": [],
        "timestamps": "naive local SUTD time; no tz shift",
        "splits": {
            split_name: {
                "path": str(out_dir / f"{split_name}.parquet"),
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "rows": int(len(df)),
                "label_source": "List_of_attacks_Final.xlsx intervals",
            }
            for split_name, df in splits.items()
        },
        "label_source": "List_of_attacks_Final.xlsx intervals",
        "raw_file_checksums": {
            "SWaT_Dataset_Normal_v1.xlsx": registry._sha256(_NORMAL_PATH),
            "SWaT_Dataset_Attack_v0.xlsx": registry._sha256(_ATTACK_PATH),
            "List_of_attacks_Final.xlsx": registry._sha256(_ATTACK_LIST_PATH),
            "SWaT_Dataset_Normal_v0.xlsx (unused)": registry._sha256(_UNUSED_NORMAL_V0_PATH),
        },
        "other_releases": {
            "swat_a3": {"path": "data/SWaT/SWaT.A3 (Jun 2017)", "format": "xlsx"},
            "swat_a4_a5": {"path": "data/SWaT/SWaT.A4 & A5_Jul 2019", "format": "xlsx"},
            "swat_a6": {"path": "data/SWaT/SWaT.A6 (Dec 2019)", "format": "xlsx"},
            "swat_a7": {"path": "data/SWaT/SWaT.A7 (Jun 2020)", "format": "xlsx"},
            "swat_a8": {"path": "data/SWaT/SWaT.A8 (Jun 2021)", "format": "xlsx"},
            "swat_a9": {"path": "data/SWaT/SWaT.A9 (Nov 2022)", "format": "xlsx"},
            "swat_a11": {"path": "data/SWaT/SWaT.A11_OTDataset_Feb_26", "format": "xlsx"},
            "swat_a12": {"path": "data/SWaT/SWaT.A12_OTDataset_Mar_26", "format": "xlsx"},
        },
    }
    registry._write_meta(meta, out_dir / "meta.json")
    return meta


def load(split: str) -> pd.DataFrame:
    return registry._load_parquet(DATASET, split)


def get_meta() -> dict:
    return registry._read_meta(DATASET)