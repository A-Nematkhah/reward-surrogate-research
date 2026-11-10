#!/usr/bin/env python3
"""Refresh multi-checkpoint early behavioral features without full 50k retrain.

For each completed Phase-1 seed run, trains a fresh PPO only up to
max(early checkpoints) (default 12k), writes schema-v2
``early_behavioral_features.json``, and leaves ``model.zip`` / eval intact.

Resume-safe: skips runs whose early payload already matches config checkpoints.

Use ``--workers N`` for process-parallel refresh (recommended: 4–6 on multi-core).
When workers > 1, PPO ``n_envs`` is forced to 1 to avoid CPU oversubscription.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.features.early_behavioral import (
    EARLY_FEATURES_FILENAME,
    early_payload_matches,
    load_early_payload,
    resolve_early_checkpoints,
)
from rsr.rewards.reward_function import RewardSpec
from rsr.training.parallel_run_jobs import refresh_early_job
from rsr.utils.config import apply_dot_overrides, load_merged_configs
from rsr.utils.logging import get_logger, setup_logging
from rsr.utils.parallel_jobs import map_maybe_parallel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1" / "runs",
    )
    p.add_argument("--limit", type=int, default=None, help="Max runs to process")
    p.add_argument("--force", action="store_true", help="Rewrite even if schema matches")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (1 = sequential). Default 4.",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. ppo.early_behavioral.n_episodes=20",
    )
    p.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="Only refresh this candidate (repeatable)",
    )
    return p.parse_args()


def _load_spec(run_dir: Path) -> RewardSpec:
    spec_path = run_dir / "reward_spec.json"
    if spec_path.is_file():
        with spec_path.open("r", encoding="utf-8") as handle:
            return RewardSpec.from_dict(json.load(handle))
    with (run_dir / "status.json").open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    return RewardSpec(
        candidate_id=str(status["candidate_id"]),
        family_id=str(status.get("family_id", "unknown")),
        weights=dict(status.get("reward_weights") or {}),
        scales={k: 1.0 for k in (status.get("reward_weights") or {})},
    )


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("refresh_early")

    configs = load_merged_configs()
    if args.override:
        configs = apply_dot_overrides(configs, args.override)
    env_config = configs["environment"]
    ppo_config = dict(configs["ppo"])
    early_cfg = dict(ppo_config.get("early_behavioral") or {})
    if not bool(early_cfg.get("enabled", False)):
        logger.error("ppo.early_behavioral.enabled is false; aborting")
        return 1
    checkpoints = resolve_early_checkpoints(early_cfg)
    workers = max(1, int(args.workers))
    if workers > 1:
        # Avoid n_envs × workers CPU thrash.
        ppo_config["n_envs"] = 1
    logger.info(
        "Refreshing early features checkpoints=%s workers=%d n_envs=%s",
        checkpoints,
        workers,
        ppo_config.get("n_envs"),
    )

    run_dirs = sorted(args.runs_root.glob("*/seed_*"))
    if args.candidate_id:
        allow = set(args.candidate_id)
        run_dirs = [d for d in run_dirs if d.parent.name in allow]

    jobs: list[dict[str, Any]] = []
    skipped = 0
    for run_dir in run_dirs:
        if args.limit is not None and len(jobs) >= int(args.limit):
            break
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            skipped += 1
            continue
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("training_status") != "completed":
            skipped += 1
            continue
        seed = int(status.get("training_seed", run_dir.name.split("_")[-1]))
        early_path = run_dir / EARLY_FEATURES_FILENAME
        if early_path.is_file() and not args.force:
            try:
                payload = load_early_payload(early_path)
                if early_payload_matches(payload, expected_checkpoints=checkpoints):
                    skipped += 1
                    continue
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        reward_spec = _load_spec(run_dir)
        jobs.append(
            {
                "run_dir": str(run_dir),
                "seed": seed,
                "candidate_id": reward_spec.candidate_id,
                "reward_spec": reward_spec.to_dict(),
                "env_config": env_config,
                "ppo_config": ppo_config,
            }
        )

    logger.info("Queued %d runs (skipped=%d)", len(jobs), skipped)
    t0 = time.perf_counter()
    done = failed = 0

    def _on_result(i: int, _job: dict[str, Any], result: dict[str, Any]) -> None:
        nonlocal done, failed
        if result.get("ok"):
            done += 1
            logger.info(
                "[%d/%d] Refreshed %s seed=%s",
                i,
                len(jobs),
                result.get("candidate_id"),
                result.get("seed"),
            )
        else:
            failed += 1
            logger.error(
                "[%d/%d] Failed %s: %s",
                i,
                len(jobs),
                result.get("run_dir"),
                result.get("error"),
            )

    map_maybe_parallel(
        refresh_early_job,
        jobs,
        workers=workers,
        on_result=_on_result,
    )

    logger.info(
        "Done. refreshed=%d skipped=%d failed=%d workers=%d elapsed=%.1fs",
        done,
        skipped,
        failed,
        workers,
        time.perf_counter() - t0,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
