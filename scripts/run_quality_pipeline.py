#!/usr/bin/env python3
"""
Surrogate quality pipeline (one command):

1) flag PPO outlier seeds
2) refresh multi-checkpoint early features (parallel workers)
3) rebuild dataset (drop outliers by default)
4) extract features
5) confirmatory model comparison (ExtraTrees)
6) write model-comparison report

The default ``--env 2d`` does not run full Phase-1 PPO.
``--env crowdnav`` trains Phase 1 into experiments/crowdnav first
(ORCA humans) and writes results/crowdnav. It does not touch the 2D runs.

From repo root:

    python scripts/run_quality_pipeline.py
    python scripts/run_quality_pipeline.py --env crowdnav
    python scripts/run_quality_pipeline.py --env crowdnav --max-jobs 1 --min-completed 1
    python scripts/run_quality_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Parallel processes. CrowdNav phase 1 keeps n_envs=4. "
            "Early refresh forces n_envs=1. Default 4."
        ),
    )
    p.add_argument(
        "--min-completed",
        type=int,
        default=50,
        help="Min completed rows for dataset validation",
    )
    p.add_argument(
        "--skip-flag",
        action="store_true",
        help="Skip outlier flagging (reuse existing outlier_seeds.csv if present)",
    )
    p.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip early-feature refresh",
    )
    p.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Stop after feature extract (no model comparison / report)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Pass --force to refresh_early_features.py",
    )
    p.add_argument(
        "--refresh-limit",
        type=int,
        default=None,
        help="Optional cap on early-refresh runs (smoke / debug)",
    )
    p.add_argument(
        "--feature-sets",
        nargs="+",
        default=["early+math+behavioral"],
        help="Feature sets for confirmatory comparison",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["extra_trees"],
        help="Models for confirmatory comparison",
    )
    p.add_argument(
        "--env",
        choices=("2d", "crowdnav"),
        default="2d",
        help="2d keeps the locked nav dataset; crowdnav trains a separate root",
    )
    p.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="CrowdNav only: cap Phase 1 (candidate, seed) jobs",
    )
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help="CrowdNav only: skip PPO and reuse experiments/crowdnav runs",
    )
    p.add_argument(
        "--humans",
        type=int,
        default=5,
        help="CrowdNav human count (ORCA)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps only; do not execute",
    )
    return p.parse_args()


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("\n===", " ".join(cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    args = parse_args()
    py = sys.executable
    t0 = time.perf_counter()

    crowdnav = args.env == "crowdnav"
    if crowdnav:
        root = "experiments/crowdnav"
        runs = f"{root}/runs"
        dataset = f"{root}/dataset/reward_performance_dataset.csv"
        outliers = f"{root}/outlier_seeds.csv"
        features = f"{root}/features"
        comparison = f"{root}/model_comparison"
        report = "results/crowdnav"
        env_overrides = [
            "--override",
            "environment.env_id=CrowdNavPlusPlus-v0",
            "--override",
            f"environment.crowdnav.human_num={int(args.humans)}",
        ]
    else:
        root = "experiments/phase1"
        runs = f"{root}/runs"
        dataset = f"{root}/dataset/reward_performance_dataset.csv"
        outliers = f"{root}/outlier_seeds.csv"
        features = "experiments/phase2/features"
        comparison = "experiments/model_comparison_confirm"
        report = "results/model_comparison_confirm"
        env_overrides = []

    print("Quality pipeline starting.", flush=True)
    print(f"  env={args.env}", flush=True)
    print(f"  workers={args.workers}", flush=True)
    print(f"  models={args.models}", flush=True)
    print(f"  feature_sets={args.feature_sets}", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    if crowdnav and not args.skip_phase1:
        candidates = "experiments/phase1/candidates.json"
        if not (PROJECT_ROOT / candidates).is_file():
            run(
                [
                    py,
                    "scripts/generate_candidates.py",
                    "--out-json",
                    candidates,
                    "--out-dir",
                    f"{root}/candidates",
                ],
                dry_run=args.dry_run,
            )
        phase1 = [
            py,
            "scripts/run_phase1.py",
            "--experiments-root",
            root,
            "--candidates-json",
            candidates,
            *env_overrides,
            "--workers",
            str(int(args.workers)),
        ]
        if args.max_jobs is not None:
            phase1.extend(["--max-jobs", str(int(args.max_jobs))])
        run(phase1, dry_run=args.dry_run)

    if not args.skip_flag:
        if crowdnav:
            run(
                [
                    py,
                    "scripts/build_dataset.py",
                    "--experiments-root",
                    root,
                    "--min-completed",
                    str(int(args.min_completed)),
                    "--no-drop-outliers",
                ],
                dry_run=args.dry_run,
            )
            run(
                [
                    py,
                    "scripts/flag_ppo_outliers.py",
                    "--dataset",
                    dataset,
                    "--output",
                    outliers,
                    "--runs-root",
                    runs,
                ],
                dry_run=args.dry_run,
            )
        else:
            run([py, "scripts/flag_ppo_outliers.py"], dry_run=args.dry_run)

    refresh_crowdnav = crowdnav and args.force_refresh
    if (not crowdnav and not args.skip_refresh) or refresh_crowdnav:
        refresh_cmd = [
            py,
            "scripts/refresh_early_features.py",
            "--workers",
            str(int(args.workers)),
        ]
        if crowdnav:
            refresh_cmd.extend(["--runs-root", runs, *env_overrides])
        if args.force_refresh:
            refresh_cmd.append("--force")
        if args.refresh_limit is not None:
            refresh_cmd.extend(["--limit", str(int(args.refresh_limit))])
        run(refresh_cmd, dry_run=args.dry_run)

    build_cmd = [
        py,
        "scripts/build_dataset.py",
        "--min-completed",
        str(int(args.min_completed)),
    ]
    if crowdnav:
        build_cmd.extend(
            [
                "--experiments-root",
                root,
                "--drop-outliers-csv",
                outliers,
            ]
        )
    run(build_cmd, dry_run=args.dry_run)

    extract_cmd = [py, "scripts/extract_features.py"]
    if crowdnav:
        extract_cmd.extend(
            [
                "--dataset",
                dataset,
                "--output-dir",
                features,
                "--experiments-root",
                root,
                "--override",
                "environment.env_id=CrowdNavPlusPlus-v0",
                "--override",
                f"environment.crowdnav.human_num={int(args.humans)}",
                "--override",
                "surrogate.probes.probe_set_path=experiments/crowdnav/probe_trajectories.npz",
                "--override",
                "surrogate.probes.probe_config_path=experiments/crowdnav/probe_config.json",
            ]
        )
    run(extract_cmd, dry_run=args.dry_run)

    if not args.skip_comparison:
        cmp_cmd = [
            py,
            "scripts/run_model_comparison.py",
            "--targets",
            "success_rate",
            "--feature-sets",
            *list(args.feature_sets),
            "--models",
            *list(args.models),
        ]
        if crowdnav:
            cmp_cmd.extend(
                [
                    "--features-dir",
                    features,
                    "--output-dir",
                    comparison,
                    "--results-dir",
                    report,
                ]
            )
        run(cmp_cmd, dry_run=args.dry_run)

        report_cmd = [py, "scripts/report_model_comparison.py"]
        if crowdnav:
            report_cmd.extend(
                ["--exp-dir", comparison, "--results-dir", report]
            )
        run(report_cmd, dry_run=args.dry_run)

    elapsed = time.perf_counter() - t0
    print(f"\n=== quality pipeline finished OK ({elapsed / 3600:.2f} h) ===", flush=True)
    print(f"Outputs: {comparison}", flush=True)
    print(f"Report:  {report}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
