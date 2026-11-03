#!/usr/bin/env python3
"""Validate Phase 1 runs and build the reward–performance dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.builder import build_and_save_phase1_dataset
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("build_dataset")

    df, paths = build_and_save_phase1_dataset(
        args.experiments_root,
        output_dir=args.output_dir,
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
