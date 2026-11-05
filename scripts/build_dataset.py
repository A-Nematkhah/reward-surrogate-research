#!/usr/bin/env python3
"""Validate Phase 1 runs and build the reward–performance dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.builder import build_and_save_phase1_dataset
from src.dataset.outlier_filter import drop_flagged_seeds
from src.dataset.validator import DatasetValidationError, validate_dataset
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <experiments-root>/dataset",
    )
    parser.add_argument(
        "--min-completed",
        type=int,
        default=1,
        help="Minimum completed rows required for validation",
    )
    parser.add_argument(
        "--drop-outliers-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1" / "outlier_seeds.csv",
        help=(
            "Outlier seeds CSV from flag_ppo_outliers.py (default path). "
            "Pass an empty string to disable dropping."
        ),
    )
    parser.add_argument(
        "--no-drop-outliers",
        action="store_true",
        help="Do not drop outlier seeds even if the default CSV exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("build_dataset")

    df, paths = build_and_save_phase1_dataset(
        args.experiments_root,
        output_dir=args.output_dir,
    )
    drop_csv = None if args.no_drop_outliers else args.drop_outliers_csv
    if drop_csv is not None and str(drop_csv).strip() and Path(drop_csv).is_file():
        flagged = pd.read_csv(drop_csv)
        before = len(df)
        df = drop_flagged_seeds(df, flagged)
        # Rewrite artifacts with filtered rows
        out_dir = Path(paths["csv"]).parent
        csv_path = out_dir / "reward_performance_dataset.csv"
        df.to_csv(csv_path, index=False)
        logger.info(
            "Dropped outlier seeds: %d -> %d rows (using %s)",
            before,
            len(df),
            drop_csv,
        )
    elif not args.no_drop_outliers:
        logger.info(
            "No outlier CSV at %s — dataset kept unfiltered. "
            "Run scripts/flag_ppo_outliers.py first for filter-by-default.",
            args.drop_outliers_csv,
        )
    logger.info("Dataset shape: %s", df.shape)

    try:
        report = validate_dataset(df, min_completed=args.min_completed)
    except DatasetValidationError as exc:
        logger.error("Dataset validation failed: %s", exc)
        return 1

    report_path = paths["csv"].parent / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Validation OK: %s", report["stats"])
    logger.info("Wrote %s", report_path)
    for kind, path in paths.items():
        logger.info("  %s: %s", kind, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
