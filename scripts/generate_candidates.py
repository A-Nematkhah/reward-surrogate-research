#!/usr/bin/env python3
"""Generate and save the Phase 1 reward candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rewards.candidate_generator import (
    generate_candidates,
    save_candidates,
    summarize_candidates,
)
from src.utils.config import apply_dot_overrides, load_named_config
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs",
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override generation_seed",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. n_candidates=32 (repeatable)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override output JSON path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override per-candidate JSON directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("generate_candidates")

    config = load_named_config("reward_generation", config_dir=args.config_dir)
    if args.override:
        # apply_dot_overrides expects nested keys; reward_generation is flat at top
        config = apply_dot_overrides(config, args.override)
    if args.seed is not None:
        config["generation_seed"] = args.seed

    candidates = generate_candidates(config)
    summary = summarize_candidates(candidates)

    out_json = args.out_json
    if out_json is None:
        out_json = PROJECT_ROOT / config["output"]["candidates_json"]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = PROJECT_ROOT / config["output"]["candidates_dir"]

    path = save_candidates(candidates, json_path=out_json, candidates_dir=out_dir)
    summary_path = path.with_name("candidates_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Generated %d candidates across %d families", summary["n_candidates"], summary["n_families"])
    for family, count in summary["by_family"].items():
        logger.info("  family %-18s n=%d", family, count)
    logger.info("Weight spread (min..max):")
    for name, stats in summary["weight_stats"].items():
        logger.info(
            "  %-12s [%7.3f, %7.3f] mean=%7.3f std=%7.3f (+%d/-%d/0=%d)",
            name,
            stats["min"],
            stats["max"],
            stats["mean"],
            stats["std"],
            stats["n_positive"],
            stats["n_negative"],
            stats["n_zero"],
        )
    logger.info("Wrote %s", path)
    logger.info("Wrote per-candidate files under %s", out_dir)
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
