#!/usr/bin/env python3
"""Run Phase 2: train/evaluate surrogate models with grouped validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.training import run_phase2_experiments
from src.utils.config import apply_dot_overrides, load_named_config
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2" / "features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. surrogate.validation.n_cv_folds=3",
    )
    return parser.parse_args()


def _resolve_output_paths(config: dict, output_dir: Path) -> dict:
    """Make configured relative output paths absolute under the project/output dir."""
    out = dict(config)
    out_cfg = dict(out.get("output", {}))
    out_cfg["results_dir"] = str(output_dir)
    out_cfg["metrics_csv"] = str(output_dir / "surrogate_metrics.csv")
    out_cfg["predictions_dir"] = str(output_dir / "predictions")
    out_cfg["figures_dir"] = str(PROJECT_ROOT / "results" / "phase2_figures")
    out["output"] = out_cfg
    return out


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("run_phase2")

    if not args.features_dir.is_dir():
        logger.error(
            "Features directory not found: %s (run scripts/extract_features.py first)",
            args.features_dir,
        )
        return 1

    config = load_named_config("surrogate")
    if args.override:
        # Support both "validation.n_cv_folds=3" and "surrogate.validation..."
        overrides = []
        for item in args.override:
            if item.startswith("surrogate."):
                overrides.append(item[len("surrogate.") :])
            else:
                overrides.append(item)
        config = apply_dot_overrides(config, overrides)

    config = _resolve_output_paths(config, args.output_dir)
    logger.info(
        "Phase 2: models=%s feature_sets=%s targets=%s",
        list(config.get("baselines", [])) + list(config.get("surrogates", [])),
        config.get("feature_sets"),
        config.get("targets"),
    )

    result = run_phase2_experiments(
        args.features_dir,
        config,
        output_dir=args.output_dir,
    )
    metrics = result["metrics"]
    oof = metrics[metrics["split"].isin(["grouped_cv_oof", "lofo_oof"])]
    oof = oof[~oof["model"].isin(["mean", "median"])]
    if not oof.empty and "spearman" in oof.columns:
        logger.info("Family-held-out OOF (non-constant models), top by Spearman:")
        cols = [
            "model",
            "feature_set",
            "target",
            "mae",
            "r2",
            "spearman",
            "fold_spearman_mean",
            "fold_spearman_std",
            "ranking_status",
        ]
        cols = [c for c in cols if c in oof.columns]
        preview = oof[cols].sort_values("spearman", ascending=False).head(15)
        logger.info("\n%s", preview.to_string(index=False))

    logger.info("Wrote %s", result["summary"]["metrics_path"])
    logger.info("Wrote %s", result["summary"]["predictions_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
