"""Write aggregate BATADAL constraint evidence for inverse-benchmark design."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# ruff: noqa: E402
# The CLI makes the repository importable before package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import get_meta, load_dataset
from src.experiments.batadal_constraint_audit import summarize_constraints


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "narrow_inverse_research" / "data_audit"


def parse_args() -> argparse.Namespace:
    """Parse target selection and aggregate-report output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--companion-count", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_clean_training_values() -> pd.DataFrame:
    """Load normal BATADAL training rows without exposing raw records."""
    sensor_columns = list(get_meta("batadal")["sensor_columns"])
    train = load_dataset("batadal", "train")
    return train.loc[train.label == 0, sensor_columns]


def write_report(summary: dict[str, object], output_dir: Path) -> None:
    """Persist machine-readable and concise human-readable aggregate evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "constraint_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    target_range = summary["target_range"]
    candidates = summary["companion_candidates"]
    report_lines = [
        "# BATADAL Narrow-Inverse Constraint Audit",
        "",
        "This report contains aggregate training-data evidence only. Correlation-ranked",
        "companions are candidates for investigation, not trusted measurements.",
        "",
        f"- Target sensor: `{summary['target_sensor']}`",
        f"- Clean training rows: {summary['row_count']}",
        f"- Sensor count: {summary['sensor_count']}",
        f"- Excluded constant sensors: {summary['constant_sensor_count']}",
        "",
        "## Target Range",
        "",
        "| Minimum | Q01 | Median | Q99 | Maximum | Std. dev. | Missing |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {minimum:.6g} | {quantile_01:.6g} | {median:.6g} | {quantile_99:.6g} | "
        "{maximum:.6g} | {standard_deviation:.6g} | {missing_count} |".format(**target_range),
        "",
        "## Correlation-Ranked Companion Candidates",
        "",
        "| Sensor | Correlation | Absolute correlation |",
        "|---|---:|---:|",
    ]
    report_lines.extend(
        f"| `{candidate['sensor']}` | {candidate['correlation']:.6f} | "
        f"{candidate['absolute_correlation']:.6f} |"
        for candidate in candidates
    )
    (output_dir / "CONSTRAINT_AUDIT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Inspect the requested target sensor and write compact evidence artifacts."""
    settings = parse_args()
    clean_values = load_clean_training_values()
    if not 0 <= settings.target_index < len(clean_values.columns):
        raise ValueError("target-index must select an available BATADAL sensor")
    target_sensor = str(clean_values.columns[settings.target_index])
    summary = summarize_constraints(
        clean_values,
        target_sensor=target_sensor,
        companion_count=settings.companion_count,
    )
    write_report(summary, settings.output_dir)
    print(f"Saved aggregate constraint audit to {settings.output_dir}", flush=True)


if __name__ == "__main__":
    main()
