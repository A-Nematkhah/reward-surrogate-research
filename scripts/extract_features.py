#!/usr/bin/env python3
"""Extract static and behavioral reward features for Phase 2 (no training leakage)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_builder import (
    build_feature_tables,
    ensure_probe_set,
    save_feature_tables,
)
from src.utils.config import apply_dot_overrides, load_named_config
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2" / "features",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1",
        help="Phase 1 runs root (for early behavioral features)",
    )
    parser.add_argument(
        "--regenerate-probes",
        action="store_true",
        help="Rebuild the fixed probe trajectory set even if it exists",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override under surrogate/environment, e.g. surrogate.probes.seed=7",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("extract_features")

    if not args.dataset.is_file():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    env_config = load_named_config("environment")
    surrogate_config = load_named_config("surrogate")
    if args.override:
        merged = apply_dot_overrides(
            {"environment": env_config, "surrogate": surrogate_config},
            args.override,
        )
        env_config = merged["environment"]
        surrogate_config = merged["surrogate"]

    df = pd.read_csv(args.dataset)
    logger.info("Loaded dataset: %s rows", len(df))

    probe_config = surrogate_config["probes"]
    episodes = ensure_probe_set(
        env_config,
        probe_config,
        project_root=PROJECT_ROOT,
        regenerate=args.regenerate_probes,
    )
    n_transitions = sum(len(ep.transitions) for ep in episodes)
    logger.info(
        "Probe set: %d episodes, %d transitions, policies=%s",
        len(episodes),
        n_transitions,
        sorted({ep.policy_name for ep in episodes}),
    )

    include_early = bool(surrogate_config.get("include_early_behavioral", False))
    tables = build_feature_tables(
        df,
        episodes,
        seed_aggregation=str(surrogate_config.get("seed_aggregation", "mean")),
        include_family_one_hot=bool(
            surrogate_config.get("include_family_one_hot", False)
        ),
        experiments_root=args.experiments_root,
        include_early_behavioral=include_early,
    )
    paths = save_feature_tables(tables, args.output_dir)
    for name, path in paths.items():
        logger.info("  %s -> %s", name, path)

    # Quick leakage audit printout
    for set_name in (
        "static",
        "behavioral",
        "math",
        "interaction",
        "early",
        "hybrid",
    ):
        if set_name not in tables:
            continue
        cols = list(tables[set_name].columns)
        logger.info("%s features: %d", set_name, len(cols))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
