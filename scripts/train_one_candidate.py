#!/usr/bin/env python3
"""Train and evaluate a single reward candidate (Step 4 smoke / debug entrypoint)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rewards.candidate_generator import load_candidates
from src.training.ppo_trainer import load_trained_model, run_dir_for, train_candidate_seed
from src.utils.config import apply_dot_overrides, load_named_config
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", type=str, default="balanced_00")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1" / "candidates.json",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override nested under section, e.g. ppo.total_timesteps=5000",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a completed run already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("train_one")

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

    candidates = load_candidates(args.candidates_json)
    matching = [c for c in candidates if c.candidate_id == args.candidate_id]
    if not matching:
        available = ", ".join(c.candidate_id for c in candidates[:8])
        raise SystemExit(
            f"Unknown candidate_id '{args.candidate_id}'. Examples: {available}, ..."
        )
    spec = matching[0]

    run_dir = run_dir_for(args.experiments_root, spec.candidate_id, args.seed)
    logger.info(
        "Training %s seed=%s timesteps=%s -> %s",
        spec.candidate_id,
        args.seed,
        ppo_config["total_timesteps"],
        run_dir,
    )

    status = train_candidate_seed(
        spec,
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        seed=args.seed,
        run_dir=run_dir,
        resume_if_complete=not args.force,
    )

    # Verify checkpoint loads
    if status.get("training_status") == "completed":
        model = load_trained_model(run_dir)
        logger.info("Checkpoint OK: %s (device=%s)", run_dir / "model.zip", model.device)
        logger.info("Metrics: %s", json.dumps(status.get("metrics", {}), indent=2))
    else:
        logger.error("Run did not complete: %s", status.get("failure"))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
