"""Prepare clean, standardized parquet files for SWaT and WADI datasets.

Processes:
1. SWaT A10/A11 OT telemetry:
   - Coerces 'Bad Input' strings to NaN, forward-fills and backward-fills.
   - Extracts the 51 canonical physical tags and synthesizes a standardized schema.
   - Chronologically splits into 80% train and 20% val.
   - Saves to data/processed/swat/train.parquet and data/processed/swat/val.parquet.

2. WADI 14-days A2 telemetry:
   - Drops 4 uninstalled columns (2_LS_001_AL, 2_LS_002_AL, 2_P_001_STATUS, 2_P_002_STATUS).
   - Synthesizes continuous monotonic datetime timestamps.
   - Interpolates transient calibration nulls.
   - Chronologically splits into first 10 calendar days train (~70%) and remaining days val (~30%).
   - Saves to data/processed/wadi/train.parquet and data/processed/wadi/val.parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

SWAT_51_TAGS = [
    "FIT101",
    "LIT101",
    "MV101",
    "P101",
    "P102",
    "AIT201",
    "AIT202",
    "AIT203",
    "FIT201",
    "MV201",
    "P201",
    "P202",
    "P203",
    "P204",
    "P205",
    "P206",
    "DPIT301",
    "FIT301",
    "LIT301",
    "MV301",
    "MV302",
    "MV303",
    "MV304",
    "P301",
    "P302",
    "AIT401",
    "AIT402",
    "FIT401",
    "LIT401",
    "P401",
    "P402",
    "P403",
    "P404",
    "UV401",
    "AIT501",
    "AIT502",
    "AIT503",
    "AIT504",
    "FIT501",
    "FIT502",
    "FIT503",
    "FIT504",
    "P501",
    "P502",
    "PIT501",
    "PIT502",
    "PIT503",
    "FIT601",
    "P601",
    "P602",
    "P603",
]

WADI_DROP_COLUMNS = [
    "2_LS_001_AL",
    "2_LS_002_AL",
    "2_P_001_STATUS",
    "2_P_002_STATUS",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare SWaT and WADI parquets")
    parser.add_argument(
        "--swat-raw",
        type=str,
        default="data/SWaT/SWaT.A11_OTDataset_Feb_26/SWaT.A10_OTDataset_19-Feb-2026_0930_1735.csv",
        help="Path to raw SWaT CSV",
    )
    parser.add_argument(
        "--wadi-raw",
        type=str,
        default="data/WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv",
        help="Path to raw WADI CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Root processed output directory",
    )
    return parser.parse_args()


def process_swat(raw_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Process SWaT CSV into train and val parquet splits."""
    print(f"Loading SWaT raw telemetry from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)

    # 1. Parse timestamp
    timestamp = pd.to_datetime(df["t_stamp"])

    # 2. Map 51 physical channels
    channel_data: dict[str, pd.Series] = {"timestamp": timestamp}
    for tag in SWAT_51_TAGS:
        source_col = None
        for candidate in [f"{tag}.Pv", f"{tag}.Status", tag]:
            if candidate in df.columns:
                source_col = candidate
                break

        if source_col is None:
            raise KeyError(f"Could not map tag '{tag}' to any column in SWaT CSV")

        # Coerce 'Bad Input' or non-numeric strings to NaN
        series = pd.to_numeric(
            df[source_col].replace(r"(?i)bad\s*input", np.nan, regex=True),
            errors="coerce",
        )
        # Forward fill and backward fill missing values
        series = series.ffill().bfill().astype(np.float32)
        channel_data[tag] = series

    swat_clean = pd.DataFrame(channel_data)

    # 3. Chronological split: 80% train, 20% val
    split_idx = int(len(swat_clean) * 0.8)
    train_df = swat_clean.iloc[:split_idx].reset_index(drop=True)
    val_df = swat_clean.iloc[split_idx:].reset_index(drop=True)

    dest_dir = out_dir / "swat"
    dest_dir.mkdir(parents=True, exist_ok=True)
    train_path = dest_dir / "train.parquet"
    val_path = dest_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(
        f"Saved SWaT: train={len(train_df)} rows ({train_path.stat().st_size} bytes), "
        f"val={len(val_df)} rows ({val_path.stat().st_size} bytes)"
    )
    return train_path, val_path


def process_wadi(raw_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Process WADI CSV into train and val parquet splits."""
    print(f"Loading WADI raw telemetry from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 1. Drop uninstalled columns and metadata
    cols_to_drop = [c for c in WADI_DROP_COLUMNS if c in df.columns]
    if "Row" in df.columns:
        cols_to_drop.append("Row")
    df = df.drop(columns=cols_to_drop)

    # 2. Synthesize continuous timestamps per calendar date group
    print("Synthesizing WADI timestamps across calendar date groups...")
    timestamps: list[pd.Timestamp] = []
    for _date_val, group in df.groupby("Date", sort=False):
        start_ts = pd.to_datetime(_date_val)
        delta_ts = pd.to_timedelta(np.arange(len(group)), unit="s")
        timestamps.extend(start_ts + delta_ts)

    # 3. Clean numeric sensor channels and interpolate calibration nulls
    sensor_cols = [c for c in df.columns if c not in ("Date", "Time")]
    numeric_df = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    numeric_clean = (
        numeric_df.interpolate(method="linear").ffill().bfill().astype(np.float32)
    )

    df_clean = pd.concat(
        [pd.Series(timestamps, name="timestamp"), numeric_clean], axis=1
    )

    # 4. Chronological split: first 10 calendar days train (~70%), remaining days val (~30%)
    # Train dates: Sept 25 to Oct 4 (8 recording days, 535,200 rows = 68.2%)
    # Val dates: Oct 5 to Oct 7 (3 recording days, 249,371 rows = 31.8%)
    cutoff_time = pd.Timestamp("2017-10-05 00:00:00")
    train_df = df_clean[df_clean["timestamp"] < cutoff_time].reset_index(drop=True)
    val_df = df_clean[df_clean["timestamp"] >= cutoff_time].reset_index(drop=True)

    dest_dir = out_dir / "wadi"
    dest_dir.mkdir(parents=True, exist_ok=True)
    train_path = dest_dir / "train.parquet"
    val_path = dest_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(
        f"Saved WADI: train={len(train_df)} rows ({train_path.stat().st_size} bytes), "
        f"val={len(val_df)} rows ({val_path.stat().st_size} bytes)"
    )
    return train_path, val_path


def main() -> None:
    """Execute end-to-end data preparation for SWaT and WADI."""
    args = parse_args()
    out_dir = Path(args.out_dir)

    print("=" * 80)
    print("PREPARE SWAT & WADI PARQUET DATASETS")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 80)

    swat_raw = Path(args.swat_raw)
    wadi_raw = Path(args.wadi_raw)

    t0 = time.perf_counter()
    process_swat(swat_raw, out_dir)
    process_wadi(wadi_raw, out_dir)
    elapsed = time.perf_counter() - t0

    print("=" * 80)
    print(f"Data preparation complete in {elapsed:.2f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
