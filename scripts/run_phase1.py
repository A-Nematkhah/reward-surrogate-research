#!/usr/bin/env python3
"""Run Phase 1: train and evaluate all reward candidates (resumable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.experiment_runner import load_phase1_candidates, run_phase1
from src.utils.config import apply_dot_overrides, load_named_config
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1",
    )
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=None,
        help="Defaults to <experiments-root>/candidates.json",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="Restrict to one or more candidate IDs (repeatable)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional cap on (candidate, seed) jobs for pilot runs",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. ppo.total_timesteps=100000",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip completed runs (will retrain)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort on first training failure",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel (candidate, seed) processes. Default 1. Does not change n_envs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("run_phase1")

    env_config = load_named_config("environment")
    ppo_config = load_named_config("ppo")
    eval_config = load_named_config("evaluation")

    if args.override:
        merged = apply_dot_overrides(
            {
                "environment": env_config,
                "ppo": ppo_config,
                "evaluation": eval_config,
            },
            args.override,
        )
        env_config = merged["environment"]
        ppo_config = merged["ppo"]
        eval_config = merged["evaluation"]

    candidates = load_phase1_candidates(
        args.experiments_root, candidates_json=args.candidates_json
    )
    if args.candidate_id:
        wanted = set(args.candidate_id)
        candidates = [c for c in candidates if c.candidate_id in wanted]
        missing = wanted - {c.candidate_id for c in candidates}
        if missing:
            raise SystemExit(f"Unknown candidate_ids: {sorted(missing)}")

    logger.info(
        "Starting Phase 1 with %d candidates, timesteps=%s, seeds=%s, max_jobs=%s, workers=%s",
        len(candidates),
        ppo_config["total_timesteps"],
        ppo_config.get("seeds"),
        args.max_jobs,
        args.workers,
    )

    summary = run_phase1(
        candidates=candidates,
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        experiments_root=args.experiments_root,
        candidate_ids=None,
        max_jobs=args.max_jobs,
        resume=not args.no_resume,
        stop_on_failure=args.stop_on_failure,
        workers=int(args.workers),
    )

    if summary["n_failed"] > 0:
        logger.warning("Phase 1 finished with %d failures", summary["n_failed"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
