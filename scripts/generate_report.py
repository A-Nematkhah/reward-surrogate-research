#!/usr/bin/env python3
"""Generate Phase 1 / Phase 2 research reports and figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.builder import attach_run_configs, build_dataset_from_runs
from src.dataset.phase1_analysis import generate_phase1_figures, write_phase1_report
from src.dataset.validator import validate_dataset
from src.surrogate.phase2_analysis import (
    build_phase2_report,
    data_sensitivity_from_folds,
    generate_phase2_figures,
    load_phase2_artifacts,
    oof_metrics,
    write_model_comparison_table,
    write_phase2_report,
)
from src.utils.logging import get_logger, setup_logging


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which report(s) to generate",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase1",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1_figures",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1_report.md",
    )
    parser.add_argument(
        "--phase2-metrics",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2" / "surrogate_metrics.csv",
    )
    parser.add_argument(
        "--phase2-predictions",
        type=Path,
        default=(
            PROJECT_ROOT / "experiments" / "phase2" / "predictions" / "all_predictions.csv"
        ),
    )
    parser.add_argument(
        "--phase2-importances",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "phase2"
            / "predictions"
            / "tree_importances.json"
        ),
    )
    parser.add_argument(
        "--phase2-features-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2" / "features",
    )
    parser.add_argument(
        "--phase2-figures-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase2_figures",
    )
    parser.add_argument(
        "--phase2-report-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase2_report.md",
    )
    parser.add_argument(
        "--final-report-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "final_research_report.md",
    )
    return parser.parse_args()


def generate_phase1(args: argparse.Namespace) -> tuple[int, dict]:
    logger = get_logger("generate_report")
    df = attach_run_configs(
        build_dataset_from_runs(args.experiments_root),
        args.experiments_root,
    )
    if df.empty:
        logger.error("No runs found under %s", args.experiments_root)
        return 1, {}

    validation = validate_dataset(df, min_completed=1, require_metric_diversity=True)
    fig_paths = generate_phase1_figures(df, args.figures_dir)
    report_path = write_phase1_report(
        df, args.report_path, validation_report=validation
    )
    logger.info("Phase 1 figures: %d", len(fig_paths))
    logger.info("Phase 1 report: %s", report_path)
    summary_path = args.report_path.with_suffix(".validation.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    return 0, validation.get("stats", {})


def generate_phase2(args: argparse.Namespace, phase1_stats: dict) -> int:
    logger = get_logger("generate_report")
    if not args.phase2_metrics.is_file() or not args.phase2_predictions.is_file():
        logger.error(
            "Phase 2 artifacts missing. Run scripts/run_phase2.py first. "
            "metrics=%s predictions=%s",
            args.phase2_metrics,
            args.phase2_predictions,
        )
        return 1

    data = load_phase2_artifacts(
        args.phase2_metrics,
        args.phase2_predictions,
        args.phase2_importances,
    )
    feature_names_by_set = {}
    for name in ("static", "behavioral", "hybrid"):
        path = args.phase2_features_dir / f"features_{name}.csv"
        if path.is_file():
            feature_names_by_set[name] = list(pd.read_csv(path, nrows=0).columns)

    fig_paths = generate_phase2_figures(
        data["metrics"],
        data["predictions"],
        data["importances"],
        feature_names_by_set=feature_names_by_set,
        output_dir=args.phase2_figures_dir,
    )
    sensitivity = data_sensitivity_from_folds(data["metrics"])
    if not sensitivity.empty:
        sensitivity.to_csv(
            args.phase2_figures_dir.parent / "phase2_fold_sensitivity.csv",
            index=False,
        )

    report = build_phase2_report(
        data["metrics"],
        data["predictions"],
        phase1_stats=phase1_stats or None,
        sensitivity=sensitivity,
    )
    write_phase2_report(report, args.phase2_report_path)
    write_model_comparison_table(
        oof_metrics(data["metrics"]),
        PROJECT_ROOT / "results" / "phase2_model_comparison.csv",
    )
    logger.info("Phase 2 figures: %d", len(fig_paths))
    logger.info("Phase 2 report: %s", args.phase2_report_path)
    return 0


def write_final_report(
    args: argparse.Namespace,
    phase1_stats: dict,
) -> Path:
    """Concatenate Phase 1 + Phase 2 into one research summary with Phase 3 verdict."""
    parts: list[str] = []
    parts.append("# Final Research Report: Reward Performance Prediction")
    parts.append("")
    parts.append(
        "Standalone study on whether reward-function features can predict "
        "downstream PPO navigation performance **without** training each candidate."
    )
    parts.append("")
    parts.append("## Scope")
    parts.append("")
    parts.append("- Phase 1: 2D navigation env, reward candidates, PPO labels, dataset.")
    parts.append("- Phase 2: static/behavioral features, classical surrogates, grouped CV.")
    parts.append("- Out of scope here: CrowdNav, LLM reward code, Active Learning implementation.")
    parts.append("")

    if phase1_stats:
        parts.append("## Phase 1 headline")
        parts.append("")
        parts.append(
            f"- Completed labeled runs: {phase1_stats.get('n_completed')}; "
            f"families={phase1_stats.get('n_families')}; "
            f"success-rate span={phase1_stats.get('success_rate_span')}."
        )
        parts.append(
            f"- Seed variability (mean success spread)="
            f"{phase1_stats.get('mean_seed_success_spread')}."
        )
        parts.append(
            f"- Full narrative: `{_rel(args.report_path)}`."
        )
        parts.append("")

    if args.phase2_report_path.is_file():
        parts.append("## Phase 2 headline")
        parts.append("")
        # Pull recommendation section from phase2 report
        text = args.phase2_report_path.read_text(encoding="utf-8")
        parts.append(f"- Full narrative: `{_rel(args.phase2_report_path)}`.")
        parts.append(f"- Figures: `{_rel(args.phase2_figures_dir)}`.")
        parts.append("")
        # Extract recommendation paragraph
        if "## Recommendation: Phase 3 Active Learning?" in text:
            rec = text.split("## Recommendation: Phase 3 Active Learning?", 1)[1].strip()
            parts.append("## Phase 3 recommendation")
            parts.append("")
            parts.append(rec.split("\n\n")[0])
            parts.append("")
    else:
        parts.append("## Phase 2")
        parts.append("")
        parts.append("- Phase 2 report not found; run with `--phase 2` or `--phase all`.")
        parts.append("")

    parts.append("## Artifacts")
    parts.append("")
    parts.append("| Artifact | Path |")
    parts.append("|---|---|")
    parts.append(
        f"| Phase 1 dataset | `experiments/phase1/dataset/reward_performance_dataset.csv` |"
    )
    parts.append(f"| Phase 1 report | `{_rel(args.report_path)}` |")
    parts.append(
        f"| Phase 2 metrics | `experiments/phase2/surrogate_metrics.csv` |"
    )
    parts.append(
        f"| Phase 2 report | `{_rel(args.phase2_report_path)}` |"
    )
    parts.append(
        f"| Model comparison table | `results/phase2_model_comparison.csv` |"
    )
    parts.append("")
    parts.append(
        "Negative and mixed results are retained deliberately. "
        "Do not treat this toy study as evidence for CrowdNav-scale generalization."
    )
    parts.append("")

    path = args.final_report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("generate_report")

    phase1_stats: dict = {}
    if args.phase in ("1", "all"):
        code, phase1_stats = generate_phase1(args)
        if code != 0:
            return code
        # Prefer validation JSON if regenerating phase2 only later
    else:
        val_path = args.report_path.with_suffix(".validation.json")
        if val_path.is_file():
            with val_path.open("r", encoding="utf-8") as handle:
                phase1_stats = json.load(handle).get("stats", {})

    if args.phase in ("2", "all"):
        code = generate_phase2(args, phase1_stats)
        if code != 0:
            return code

    if args.phase == "all":
        final_path = write_final_report(args, phase1_stats)
        logger.info("Final research report: %s", final_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
