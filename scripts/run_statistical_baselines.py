"""Run leakage-safe statistical baselines on canonical processed datasets."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baseline_runner import run_dataset_statistical_baselines


ALL_DATASETS = ("hai", "swat", "wadi", "batadal")


def parse_args() -> argparse.Namespace:
    """Parse bounded, reproducible baseline-run arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_DATASETS],
        help="Canonical datasets to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional rows per split for a smoke run.")
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--ewma-alpha", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "statistical_baselines")
    return parser.parse_args()


def main() -> None:
    """Write per-timestep scores and summary metrics for requested datasets."""
    args = parse_args()
    datasets = ALL_DATASETS if "all" in args.datasets else tuple(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for dataset in datasets:
        print(f"Running statistical baselines for {dataset}...", flush=True)
        run = run_dataset_statistical_baselines(
            dataset,
            limit=args.limit,
            threshold_quantile=args.threshold_quantile,
            ewma_alpha=args.ewma_alpha,
            pca_components=args.pca_components,
        )
        run.scores.to_csv(args.output_dir / f"{dataset}_scores.csv", index=False)
        summaries.append(run.summary)
        print(f"Completed {dataset}: {len(run.scores)} test rows.", flush=True)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"Summary saved to {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
