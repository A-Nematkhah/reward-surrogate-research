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

from rsr.surrogate.model_comparison import load_yaml_config, run_model_comparison
from rsr.utils.config import apply_dot_overrides
from rsr.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_comparison.yaml",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="New experiment id; writes experiments/model_comparison_<id>/ "
        "unless --output-dir / --results-dir override.",
    )
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--feature-sets", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override KEY=VALUE (repeatable)",
    )
    return p.parse_args()


def _refuse_nonempty(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Refusing non-empty output dir {path} (pass --force to overwrite)"
        )


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("run_model_comparison")
    cfg = load_yaml_config(args.config)
    if args.override:
        cfg = apply_dot_overrides(cfg, list(args.override))
    if args.n_jobs is not None:
        cfg["n_jobs"] = int(args.n_jobs)
    if args.features_dir is not None:
        cfg["features_dir"] = str(args.features_dir)
    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("--run-id must be non-empty")
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    else:
        cfg["output_dir"] = f"experiments/model_comparison_{run_id}"
    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)
    else:
        cfg["results_dir"] = f"results/model_comparison_{run_id}"
    cfg["run_id"] = run_id
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    res = Path(cfg["results_dir"])
    if not res.is_absolute():
        res = PROJECT_ROOT / res
    locked = {
        (PROJECT_ROOT / "experiments/model_comparison_confirm").resolve(),
        (PROJECT_ROOT / "results/model_comparison_confirm").resolve(),
    }
    if out.resolve() in locked or res.resolve() in locked:
        raise SystemExit("Refusing to write locked model_comparison_confirm artifacts")
    _refuse_nonempty(out, force=bool(args.force))
    _refuse_nonempty(res, force=bool(args.force))

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
