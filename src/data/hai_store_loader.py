"""Store-format loader for the consolidated HAI parquet store.

Reads the exported raw-value store (data/processed/hai) built by
scripts/build_hai_store.py. This is distinct from src/data/hai_loader.py,
which z-scores at load time; the store holds raw values and this loader only
reads parquet, so it is safe for acceptance validation.
"""

import pandas as pd

from . import registry

DATASET = "hai"


def load(split: str) -> pd.DataFrame:
    return registry._load_parquet(DATASET, split)


def get_meta() -> dict:
    return registry._read_meta(DATASET)