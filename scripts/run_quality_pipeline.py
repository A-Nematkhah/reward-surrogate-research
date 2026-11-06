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
``--env crowdnav`` trains into ``experiments/crowdnav_<run_id>/`` and writes
``results/crowdnav_<run_id>/`` (see ``configs/crowdnav_run.yaml``). Bump
``run_id`` or pass ``--run-id`` so campaigns never overwrite each other.

From repo root:

    python scripts/run_quality_pipeline.py
    python scripts/run_quality_pipeline.py --env crowdnav
    python scripts/run_quality_pipeline.py --env crowdnav --run-id v3_trial
    python scripts/run_quality_pipeline.py --env crowdnav --max-jobs 1 --min-completed 1
    python scripts/run_quality_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


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
        help="2d keeps the locked nav dataset; crowdnav uses a versioned root",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "CrowdNav only: experiment id → experiments/crowdnav_<id>/ and "
            "results/crowdnav_<id>/. Defaults to configs/crowdnav_run.yaml."
        ),
    )
    p.add_argument(
        "--force-new-run",
        action="store_true",
        help=(
            "CrowdNav only: if the run root already has PPO runs, append a UTC "
            "stamp to run_id instead of mixing campaigns."
        ),
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
        help="CrowdNav only: skip PPO and reuse the versioned experiment root",
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


def _load_crowdnav_run_cfg() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "crowdnav_run.yaml"
    if not path.is_file():
        return {"run_id": "v2_150k", "force_new_run": False, "description": ""}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid crowdnav_run config: {path}")
    return data


def _validate_run_id(run_id: str) -> str:
    rid = str(run_id).strip()
    if not _RUN_ID_RE.match(rid):
        raise SystemExit(
            f"Invalid run_id '{run_id}'. Use letters/digits/_/- "
            f"(max 64 chars, must start alphanumeric)."
        )
    return rid


def _has_completed_runs(runs_dir: Path) -> bool:
    if not runs_dir.is_dir():
        return False
    for status in runs_dir.glob("*/seed_*/status.json"):
        try:
            payload = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("training_status") == "completed":
            return True
    return False


def resolve_crowdnav_roots(
    *,
    run_id: str | None,
    force_new_run: bool,
    dry_run: bool,
) -> tuple[str, str, str, dict[str, Any]]:
    """Return (run_id, experiments_root, results_root, run_cfg)."""
    cfg = _load_crowdnav_run_cfg()
    rid = _validate_run_id(run_id or cfg.get("run_id") or "v2_150k")
    force = bool(force_new_run or cfg.get("force_new_run"))

    root = PROJECT_ROOT / f"experiments/crowdnav_{rid}"
    runs = root / "runs"
    if _has_completed_runs(runs) and force:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid = _validate_run_id(f"{rid}_{stamp}")
        root = PROJECT_ROOT / f"experiments/crowdnav_{rid}"
        print(
            f"Existing CrowdNav runs detected; switching to fresh run_id={rid}",
            flush=True,
        )
    elif _has_completed_runs(runs) and not dry_run:
        # Resume into the same root is allowed (Phase 1 skip-complete).
        # Mixing a *new* candidate pool into an old tree is the danger — warn.
        cand = root / "candidates.json"
        if cand.is_file():
            print(
                f"Resuming CrowdNav root experiments/crowdnav_{rid}/ "
                f"(has completed runs). Bump run_id for a clean campaign.",
                flush=True,
            )

    exp_root = f"experiments/crowdnav_{rid}"
    res_root = f"results/crowdnav_{rid}"
    return rid, exp_root, res_root, cfg


def write_run_manifest(
    exp_root: str,
    *,
    run_id: str,
    cfg: dict[str, Any],
    dry_run: bool,
) -> None:
    root = PROJECT_ROOT / exp_root
    if dry_run:
        print(f"(dry-run) would write {root / 'run_manifest.json'}", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "description": cfg.get("description"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments_root": exp_root,
        "results_root": f"results/crowdnav_{run_id}",
        "ppo": {
            "total_timesteps": 150000,
            "early_checkpoints": [5000, 12000, 25000],
        },
        "reward_config": "reward_generation_crowdnav",
    }
    path = root / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def main() -> int:
    args = parse_args()
    py = sys.executable
    t0 = time.perf_counter()

    crowdnav = args.env == "crowdnav"
    run_id = None
    if crowdnav:
        run_id, root, report, run_cfg = resolve_crowdnav_roots(
            run_id=args.run_id,
            force_new_run=bool(args.force_new_run),
            dry_run=bool(args.dry_run),
        )
        runs = f"{root}/runs"
        dataset = f"{root}/dataset/reward_performance_dataset.csv"
        outliers = f"{root}/outlier_seeds.csv"
        features = f"{root}/features"
        comparison = f"{root}/model_comparison"
        probe_npz = f"{root}/probe_trajectories.npz"
        probe_cfg = f"{root}/probe_config.json"
        # PPO budget + early checkpoints from configs/ppo_crowdnav.yaml
        env_overrides = [
            "--override",
            "environment.env_id=CrowdNavPlusPlus-v0",
            "--override",
            f"environment.crowdnav.human_num={int(args.humans)}",
            "--override",
            "environment.crowdnav.robot_visible=false",
            "--override",
            "ppo.total_timesteps=150000",
            "--override",
            "ppo.early_behavioral.timesteps=12000",
            "--override",
            "ppo.early_behavioral.checkpoints=[5000,12000,25000]",
            "--override",
            "ppo.save_freq_timesteps=50000",
        ]
        write_run_manifest(root, run_id=run_id, cfg=run_cfg, dry_run=args.dry_run)
    else:
        root = "experiments/phase1"
        runs = f"{root}/runs"
        dataset = f"{root}/dataset/reward_performance_dataset.csv"
        outliers = f"{root}/outlier_seeds.csv"
        features = "experiments/phase2/features"
        comparison = "experiments/model_comparison_confirm"
        report = "results/model_comparison_confirm"
        env_overrides = []
        probe_npz = ""
        probe_cfg = ""

    print("Quality pipeline starting.", flush=True)
    print(f"  env={args.env}", flush=True)
    if crowdnav:
        print(f"  run_id={run_id}", flush=True)
        print(f"  experiments_root={root}", flush=True)
        print(f"  results_root={report}", flush=True)
    print(f"  workers={args.workers}", flush=True)
    print(f"  models={args.models}", flush=True)
    print(f"  feature_sets={args.feature_sets}", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    if crowdnav and not args.skip_phase1:
        candidates = f"{root}/candidates.json"
        # Always regenerate from the CrowdNav-aware YAML into this run root.
        run(
            [
                py,
                "scripts/generate_candidates.py",
                "--config-name",
                "reward_generation_crowdnav",
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
                    "--candidates-json",
                    f"{root}/candidates.json",
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
                "--candidates-json",
                f"{root}/candidates.json",
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
                "environment.crowdnav.robot_visible=false",
                "--override",
                f"surrogate.probes.probe_set_path={probe_npz}",
                "--override",
                f"surrogate.probes.probe_config_path={probe_cfg}",
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
