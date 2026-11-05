#!/usr/bin/env python
"""Re-evaluate Phase 1 PPO checkpoints without retraining.

By default this is **optional** in the quality pipeline (labels already use 160
episodes). Prefer ``--outliers-csv`` to re-eval only flagged unstable seeds.

Use ``--workers N`` for process-parallel evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.training.parallel_run_jobs import reeval_run_job
from src.utils.config import apply_dot_overrides, load_merged_configs
from src.utils.logging import get_logger, setup_logging
from src.utils.parallel_jobs import map_maybe_parallel

logger = get_logger("reeval_phase1")

DEFAULT_ARCHIVE = ROOT / "docs" / "archive" / "phase1_model_zips"
DEFAULT_RUNS = ROOT / "experiments" / "phase1" / "runs"
DEFAULT_OUTLIERS = ROOT / "experiments" / "phase1" / "outlier_seeds.csv"


def restore_model_zips(*, archive_root: Path, runs_root: Path) -> int:
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Archive not found: {archive_root}")
    restored = 0
    for src in archive_root.rglob("model.zip"):
        rel = src.relative_to(archive_root)
        dst = runs_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)
        restored += 1
    return restored


def _run_dirs_from_outliers(runs_root: Path, outliers_csv: Path) -> list[Path]:
    flagged = pd.read_csv(outliers_csv)
    if flagged.empty:
        return []
    dirs: list[Path] = []
    for _, row in flagged.iterrows():
        run_dir = (
            runs_root
            / str(row["candidate_id"])
            / f"seed_{int(row['training_seed'])}"
        )
        if run_dir.is_dir():
            dirs.append(run_dir)
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--skip-restore",
        action="store_true",
        help="Do not copy model.zip from archive (use existing checkpoints)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of runs to re-evaluate (debug)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (1 = sequential). Default 4.",
    )
    parser.add_argument(
        "--outliers-csv",
        type=Path,
        default=None,
        help=(
            "If set, only re-eval flagged seeds from this CSV "
            f"(typical: {DEFAULT_OUTLIERS.name}). Preferred over full reeval."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, e.g. evaluation.n_eval_episodes=160",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    cfg = load_merged_configs()
    if args.override:
        cfg = apply_dot_overrides(cfg, args.override)
    env_config = cfg["environment"]
    eval_config = cfg["evaluation"]
    workers = max(1, int(args.workers))

    if not args.skip_restore and args.outliers_csv is None:
        n_restored = restore_model_zips(
            archive_root=args.archive_root, runs_root=args.runs_root
        )
        logger.info("Restored/copied %d model.zip files from archive", n_restored)

    if args.outliers_csv is not None:
        if not args.outliers_csv.is_file():
            logger.error("Outliers CSV not found: %s", args.outliers_csv)
            return 1
        run_dirs = _run_dirs_from_outliers(args.runs_root, args.outliers_csv)
        logger.info(
            "Outlier-only re-eval: %d runs from %s",
            len(run_dirs),
            args.outliers_csv,
        )
    else:
        run_dirs = sorted(
            p.parent for p in args.runs_root.rglob("status.json") if p.parent.is_dir()
        )
        logger.warning(
            "Full re-eval of %d runs — usually unnecessary if labels already use "
            "n_eval_episodes=160. Prefer --outliers-csv for flagged seeds only.",
            len(run_dirs),
        )

    if args.limit is not None:
        run_dirs = run_dirs[: max(0, int(args.limit))]

    jobs = [
        {
            "run_dir": str(run_dir),
            "env_config": env_config,
            "eval_config": eval_config,
        }
        for run_dir in run_dirs
    ]
    logger.info(
        "Re-evaluating %d runs with n_eval_episodes=%s workers=%d",
        len(jobs),
        eval_config.get("n_eval_episodes"),
        workers,
    )

    results: list[dict[str, Any]] = []

    def _on_result(i: int, _job: dict[str, Any], row: dict[str, Any]) -> None:
        results.append(row)
        if i % 10 == 0 or i == len(jobs):
            done = sum(1 for r in results if not r.get("skipped"))
            logger.info("Progress %d/%d (completed_reevals=%d)", i, len(jobs), done)

    ordered = map_maybe_parallel(
        reeval_run_job,
        jobs,
        workers=workers,
        on_result=_on_result,
    )

    out_path = ROOT / "experiments" / "phase1" / "reeval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_runs": len(jobs),
                "n_eval_episodes": eval_config.get("n_eval_episodes"),
                "workers": workers,
                "outliers_csv": (
                    str(args.outliers_csv) if args.outliers_csv is not None else None
                ),
                "results": ordered,
            },
            handle,
            indent=2,
        )
    logger.info("Wrote %s", out_path)
    n_fail = sum(1 for r in ordered if r.get("ok") is False)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
