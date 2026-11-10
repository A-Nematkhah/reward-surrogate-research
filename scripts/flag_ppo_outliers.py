#!/usr/bin/env python3
"""Flag unstable PPO seed runs for **filter-by-default** (not full retrain).

Writes ``experiments/phase1/outlier_seeds.csv`` plus a short JSON summary.

Default next step is to drop these seeds when building the dataset::

    python scripts/build_dataset.py --min-completed 50 \\
      --drop-outliers-csv experiments/phase1/outlier_seeds.csv

Optional (rarely needed):
  --invalidate   mark runs needs_retrain for full PPO redo
  then optionally re-eval only those seeds:
    python scripts/reeval_phase1_checkpoints.py --outliers-csv ... --skip-restore
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.dataset.outlier_filter import flag_seed_outliers, summarize_flags
from rsr.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "phase1"
            / "dataset"
            / "reward_performance_dataset.csv"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1" / "outlier_seeds.csv",
    )
    p.add_argument("--metric", default="success_rate")
    p.add_argument("--max-deviation", type=float, default=0.30)
    p.add_argument("--min-span", type=float, default=0.35)
    p.add_argument(
        "--invalidate",
        action="store_true",
        help=(
            "OPT-IN: mark flagged seed runs incomplete for full PPO retrain. "
            "Prefer dataset drop instead unless flags are very few."
        ),
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1" / "runs",
    )
    return p.parse_args()


def _invalidate(runs_root: Path, flagged: pd.DataFrame) -> int:
    n = 0
    for _, row in flagged.iterrows():
        run_dir = runs_root / str(row["candidate_id"]) / f"seed_{int(row['training_seed'])}"
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            continue
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        status["training_status"] = "needs_retrain"
        status["failure"] = {
            "type": "OutlierSeed",
            "message": (
                f"Flagged outlier on {row['metric']}: value={row['metric_value']:.3f}, "
                f"median={row['candidate_median']:.3f}, span={row['candidate_span']:.3f}"
            ),
        }
        with status_path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2)
        marker = run_dir / "OUTLIER_INVALIDATED"
        marker.write_text(status["failure"]["message"], encoding="utf-8")
        n += 1
    return n


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("flag_ppo_outliers")

    if not args.dataset.is_file():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    df = pd.read_csv(args.dataset)
    flagged = flag_seed_outliers(
        df,
        metric=args.metric,
        max_deviation_from_median=args.max_deviation,
        min_seed_span=args.min_span,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if flagged.empty:
        flagged = pd.DataFrame(
            columns=[
                "candidate_id",
                "family_id",
                "training_seed",
                "metric",
                "metric_value",
                "candidate_median",
                "candidate_span",
                "abs_dev_from_median",
                "reason",
            ]
        )
    flagged.to_csv(args.output, index=False)
    summary = summarize_flags(flagged)
    summary["policy"] = "filter_by_default"
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Flagged %s seeds across %s candidates",
        summary["n_flagged_seeds"],
        summary["n_candidates"],
    )
    logger.info("Wrote %s", args.output)
    logger.info(
        "Default next step (filter, no full retrain):\n"
        "  python scripts/build_dataset.py --min-completed 50 "
        "--drop-outliers-csv %s",
        args.output.as_posix(),
    )

    if args.invalidate and not flagged.empty:
        n = _invalidate(args.runs_root, flagged)
        logger.warning(
            "Invalidated %d seed runs for FULL retrain under %s "
            "(expensive; prefer filter unless n is tiny)",
            n,
            args.runs_root,
        )
        logger.info(
            "Optional outlier-only re-eval after retrain:\n"
            "  python scripts/reeval_phase1_checkpoints.py --outliers-csv %s "
            "--skip-restore --workers 4",
            args.output.as_posix(),
        )
        logger.info("Re-run Phase 1: python scripts/run_phase1.py")
    elif not flagged.empty:
        logger.info(
            "Optional (only if you insist on repairing labels, not dropping):\n"
            "  python scripts/reeval_phase1_checkpoints.py --outliers-csv %s "
            "--skip-restore --workers 4",
            args.output.as_posix(),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
