from pathlib import Path
import hashlib
import json
from typing import Dict, Union

import pandas as pd

from ..config import get_processed_dir

PROCESSED_DIR = get_processed_dir()


def set_processed_dir(path: Union[str, Path]) -> None:
    """Dynamically override the processed store directory at runtime."""
    global PROCESSED_DIR
    PROCESSED_DIR = Path(path).resolve()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_meta(meta: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _load_parquet(name: str, split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {path}")
    return pd.read_parquet(path)


def _read_meta(name: str) -> dict:
    path = PROCESSED_DIR / name / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"meta.json not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _get_loader(name: str):
    if name == "swat":
        from . import swat_loader
        return swat_loader
    if name == "wadi":
        from . import wadi_loader
        return wadi_loader
    if name == "batadal":
        from . import batadal_loader
        return batadal_loader
    if name == "hai":
        from . import hai_store_loader
        return hai_store_loader
    raise ValueError(f"Unknown dataset name: {name!r}")


def load_dataset(name: str, split: str) -> pd.DataFrame:
    """Return canonical columns: timestamp, label, attack_id, <sensors>."""
    return _get_loader(name).load(split)


def get_meta(name: str) -> dict:
    """Return the meta.json contents for the dataset."""
    return _get_loader(name).get_meta()