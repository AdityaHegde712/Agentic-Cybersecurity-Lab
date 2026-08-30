from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import registry

DATASET = "batadal"
SAMPLING_INTERVAL_S = 3600
VAL_ROWS = 1440

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "BATADAL" / "extracted"
_DATASET03_PATH = _RAW_DIR / "BATADAL_dataset03.csv"
_DATASET04_PATH = _RAW_DIR / "BATADAL_dataset04.csv"

_DATETIME_FORMAT = "%d/%m/%y %H"


def _read_batadal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["DATETIME"], format=_DATETIME_FORMAT)
    sensor_cols = [c for c in df.columns if c not in ("DATETIME", "ATT_FLAG", "timestamp")]
    out = df[["timestamp"] + sensor_cols + ["ATT_FLAG"]].copy()
    for col in sensor_cols:
        out[col] = out[col].astype(np.float32)
    out["ATT_FLAG"] = out["ATT_FLAG"].astype(np.int32)
    return out


def _constant_columns(df: pd.DataFrame, sensor_cols: List[str]) -> List[str]:
    return [c for c in sensor_cols if df[c].nunique(dropna=True) <= 1]


def build_store() -> dict:
    normal = _read_batadal(_DATASET03_PATH)
    attack = _read_batadal(_DATASET04_PATH)
    normal = normal.sort_values("timestamp").reset_index(drop=True)
    attack = attack.sort_values("timestamp").reset_index(drop=True)

    normal["label"] = np.zeros(len(normal), dtype=np.int8)
    normal["attack_id"] = pd.array(np.full(len(normal), np.nan), dtype="Int32")
    attack["label"] = (attack["ATT_FLAG"] > 0).astype(np.int8)
    attack["attack_id"] = pd.array(
        np.where(attack["ATT_FLAG"] > 0, attack["ATT_FLAG"], np.nan), dtype="Int32"
    )
    normal = normal.drop(columns=["ATT_FLAG"])
    attack = attack.drop(columns=["ATT_FLAG"])

    train = normal.iloc[:-VAL_ROWS].reset_index(drop=True)
    val = normal.iloc[-VAL_ROWS:].reset_index(drop=True)
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
                "label_source": "ATT_FLAG column (dataset03 all 0; dataset04 -999/1)",
            }
            for split_name, df in splits.items()
        },
        "label_source": "ATT_FLAG column (dataset03 all 0; dataset04 -999/1)",
        "eval_note": "small test set; use event-level metrics",
        "cadence_note": "hourly sampling (DATETIME %d/%m/%y %H, day-first); plan assumed 1-min; val=1440 rows = 60 days of hourly data, not 1 day",
        "raw_file_checksums": {
            "BATADAL_dataset03.csv": registry._sha256(_DATASET03_PATH),
            "BATADAL_dataset04.csv": registry._sha256(_DATASET04_PATH),
        },
        "other_releases": {},
    }
    registry._write_meta(meta, out_dir / "meta.json")
    return meta


def load(split: str) -> pd.DataFrame:
    return registry._load_parquet(DATASET, split)


def get_meta() -> dict:
    return registry._read_meta(DATASET)