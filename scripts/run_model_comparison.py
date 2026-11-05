#!/usr/bin/env python3
"""Run the model-comparison experiment (reads Phase 2 features; writes new dirs only)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent TabPFN from spawning browser login windows during batch runs.
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

from src.surrogate.model_comparison import load_yaml_config, run_model_comparison
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_comparison.yaml",
    )
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--feature-sets", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("run_model_comparison")
    cfg = load_yaml_config(args.config)
    if args.n_jobs is not None:
        cfg["n_jobs"] = int(args.n_jobs)
    if args.features_dir is not None:
        cfg["features_dir"] = str(args.features_dir)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)

    t0 = time.perf_counter()
    result = run_model_comparison(
        cfg,
        targets=args.targets,
        feature_sets=args.feature_sets,
        models=args.models,
        quick=bool(args.quick),
    )
    elapsed = time.perf_counter() - t0
    logger.info("Elapsed: %.1f sec", elapsed)
    logger.info("Output dir: %s", result["output_dir"])
    if result["skipped"]:
        logger.info("Skipped models: %s", result["skipped"])
    summary = result["metrics_summary"]
    if not summary.empty and "spearman" in summary.columns:
        cols = [
            c
            for c in (
                "model",
                "feature_set",
                "target",
                "spearman",
                "within_family_spearman_mean",
                "within_family_top3_hit",
                "needs_ppo_steps",
            )
            if c in summary.columns
        ]
        preview = summary[cols].sort_values(
            "within_family_spearman_mean", ascending=False
        ).head(15)
        logger.info("Top by within-family Spearman:\n%s", preview.to_string(index=False))
    logger.info(
        "Smoke/full timing note: this run took %.1fs. Scale roughly by "
        "#models × #feature_sets × #targets × n_repeats / (quick shrinkage).",
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
