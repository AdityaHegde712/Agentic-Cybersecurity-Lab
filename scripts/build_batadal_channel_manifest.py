"""Build and seal the BATADAL channel-constraint manifest.

The manifest is estimated only from the authorized normal model-fit prefix of
the processed BATADAL training split.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import get_meta, load_dataset
from src.inverse.batadal_manifest import build_batadal_channel_manifest
from src.inverse.channel_constraints import ChannelConstraintManifest


FIT_ROW_COUNT = 5_840
DEFAULT_MANIFEST_OUTPUT_PATH = Path("configs/manifests/batadal-channel-manifest-v1.json")


def parse_args() -> argparse.Namespace:
    """Parse the destination for the sealed channel manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT_PATH,
        help="Repository-relative or absolute manifest output path.",
    )
    return parser.parse_args()


def load_fit_values() -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load exactly the authorized BATADAL model-fit prefix and sensor schema."""
    meta = get_meta("batadal")
    sensor_columns = _sensor_columns(meta)
    train = load_dataset("batadal", "train")
    if len(train) < FIT_ROW_COUNT:
        raise ValueError(
            "processed BATADAL training data has fewer than "
            f"{FIT_ROW_COUNT} rows: {len(train)}"
        )

    missing_columns = tuple(
        column for column in sensor_columns if column not in train.columns
    )
    if missing_columns:
        raise ValueError(
            "processed BATADAL train schema is missing meta sensor columns: "
            f"{', '.join(missing_columns)}"
        )

    return train.iloc[:FIT_ROW_COUNT].loc[:, sensor_columns].copy(), sensor_columns


def write_manifest(manifest: ChannelConstraintManifest, output_path: Path) -> str:
    """Atomically persist a deterministic manifest and return its byte digest."""
    if not isinstance(manifest, ChannelConstraintManifest):
        raise ValueError("manifest must be a ChannelConstraintManifest")

    output_path = Path(output_path)
    serialized = (manifest.to_json() + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return hashlib.sha256(serialized).hexdigest()


def _sensor_columns(meta: object) -> tuple[str, ...]:
    if not isinstance(meta, dict):
        raise ValueError("processed BATADAL metadata must be a JSON object")
    raw_columns = meta.get("sensor_columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("processed BATADAL metadata must declare sensor_columns")
    if not all(isinstance(column, str) and column for column in raw_columns):
        raise ValueError("processed BATADAL metadata sensor_columns must be non-empty strings")

    sensor_columns = tuple(raw_columns)
    if len(sensor_columns) != len(set(sensor_columns)):
        raise ValueError("processed BATADAL metadata sensor_columns must not contain duplicates")
    return sensor_columns


def main() -> None:
    """Build the manifest from the sealed model-fit block and write it."""
    settings = parse_args()
    fit_values, sensor_columns = load_fit_values()
    manifest = build_batadal_channel_manifest(fit_values, sensor_columns)
    digest = write_manifest(manifest, settings.output)
    print(f"Manifest path: {settings.output}", flush=True)
    print(f"SHA-256: {digest}", flush=True)
    print(f"Model-fit rows: {len(fit_values)}", flush=True)


if __name__ == "__main__":
    main()
