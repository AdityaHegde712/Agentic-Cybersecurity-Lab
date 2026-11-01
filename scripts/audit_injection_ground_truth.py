"""Audit the configured BATADAL injections without overwriting recovery artifacts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# ruff: noqa: E402
# The CLI makes the repository importable before package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import get_meta, load_dataset
from src.evaluation.baseline_runner import prepare_feature_matrices
from src.experiments.injection_ground_truth_audit import audit_injection
from src.experiments.injected_recovery_config import (
    DEFAULT_INJECTED_RECOVERY_CONFIG_PATH,
    InjectedRecoveryConfig,
    load_injected_recovery_config,
)
from src.models.lstm_forecaster import FeatureNormalizer, load_forecaster_checkpoint


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "narrow_inverse_research" / "ground_truth_audit"


def parse_args() -> argparse.Namespace:
    """Parse the source recovery configuration and dedicated audit output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_INJECTED_RECOVERY_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--normalizer-source",
        choices=("checkpoint", "train"),
        default="checkpoint",
    )
    return parser.parse_args()


def load_clean_values(config: InjectedRecoveryConfig) -> np.ndarray:
    """Load the configured clean BATADAL sequence through the canonical loader."""
    if config.dataset != "batadal" or config.source_split != "val":
        raise ValueError("the current ground-truth audit supports BATADAL validation data only")
    sensors = list(get_meta(config.dataset)["sensor_columns"])
    train = load_dataset(config.dataset, "train")
    validation = load_dataset(config.dataset, config.source_split)
    train_normal = train.loc[train.label == 0, sensors]
    validation_normal = validation.loc[validation.label == 0, sensors]
    _, clean_values, _, _ = prepare_feature_matrices(
        train_normal,
        validation_normal,
        validation_normal,
        sensors,
    )
    if len(clean_values) < config.limit:
        raise ValueError("configured limit exceeds available clean validation rows")
    return clean_values[: config.limit]


def load_training_normal_values(config: InjectedRecoveryConfig) -> np.ndarray:
    """Load canonical normal BATADAL training values for an audit-only fallback."""
    sensors = list(get_meta(config.dataset)["sensor_columns"])
    train = load_dataset(config.dataset, "train")
    return train.loc[train.label == 0, sensors].to_numpy(dtype=np.float64)


def write_report(
    records: list[dict[str, object]],
    output_dir: Path,
    normalizer_source: str,
) -> None:
    """Write compact verification evidence without serializing source data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth_audit.json").write_text(
        json.dumps(
            {"normalizer_source": normalizer_source, "records": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Injected Ground-Truth Audit",
        "",
        "All rows below are aggregate invariants; no time-series values are included.",
        f"Normalizer source: `{normalizer_source}`.",
        "",
        "| Attack | Support cells | New off-support nonzeros | Clean plus delta | "
        "Declared delta | Legacy off-support nonzeros | Legacy max residue |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    lines.extend(
        "| {attack_id} | {declared_support_count} | "
        "{delta_nonzero_outside_declared_support} | "
        "{observed_equals_clean_plus_delta} | {declared_delta_matches_expected} | "
        "{legacy_round_trip_nonzero_outside_declared_support} | "
        "{legacy_round_trip_max_abs_outside_declared_support:.6g} |".format(**record)
        for record in records
    )
    (output_dir / "GROUND_TRUTH_AUDIT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the configured scenarios through the support-integrity audit."""
    settings = parse_args()
    config_path = settings.config if settings.config.is_absolute() else PROJECT_ROOT / settings.config
    config = load_injected_recovery_config(config_path)
    clean_values = load_clean_values(config)
    if settings.normalizer_source == "checkpoint":
        forecaster = load_forecaster_checkpoint(
            PROJECT_ROOT / config.checkpoint_path,
            device=settings.device,
        )
        normalizer = forecaster.normalizer
    else:
        normalizer = FeatureNormalizer.fit(load_training_normal_values(config))
    records = [
        audit_injection(clean_values, normalizer, scenario)
        for scenario in config.scenarios
    ]
    write_report(records, settings.output_dir, settings.normalizer_source)
    print(f"Saved ground-truth audit to {settings.output_dir}", flush=True)


if __name__ == "__main__":
    main()
