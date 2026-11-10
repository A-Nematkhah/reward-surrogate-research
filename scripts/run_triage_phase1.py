#!/usr/bin/env python3
"""Phase 1: pool-level triage metrics on locked confirm OOF (no PPO)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.evaluation.pool_simulator import load_oof_candidates, simulate_pools


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/model_comparison_confirm/oof_predictions.csv",
    )
    p.add_argument("--model", default="extra_trees")
    p.add_argument("--feature-set", default="early+math+behavioral")
    p.add_argument("--target", default="success_rate")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--n-pools", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--feature-steps", type=int, default=12000)
    p.add_argument("--full-run-steps", type=int, default=50000)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "triage_phase1",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out_dir
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty {out} (pass --force)")
    out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(args.oof)
    cand = load_oof_candidates(
        oof,
        model=args.model,
        feature_set=args.feature_set,
        target=args.target,
        repeat=args.repeat,
    )
    per, summary = simulate_pools(
        cand,
        pool_sizes=(8, 16, 32),
        n_groups_list=(1, 2, 3),
        ks=(3, 4, 5, 6),
        n_pools=args.n_pools,
        selectors=("surrogate", "random", "oracle"),
        seed=args.seed,
        feature_steps_per_candidate=args.feature_steps,
        full_run_steps=args.full_run_steps,
        n_boot=args.n_boot,
    )
    per_path = out / "per_pool_metrics.csv"
    sum_path = out / "summary_by_config.csv"
    per.to_csv(per_path, index=False)
    summary.to_csv(sum_path, index=False)

    # Primary table required by Phase 1: N=16, k in {3..6}, vs random
    primary = summary[
        (summary["N"] == 16) & (summary["selector"].isin(["surrogate", "random", "oracle"]))
    ].copy()
    primary_path = out / "primary_N16.csv"
    primary.to_csv(primary_path, index=False)

    # Filename kept for compatibility; header clarifies N=16 expansion.
    single = summary[summary["n_groups_requested"] == 1].copy()
    single.to_csv(out / "summary_single_group.csv", index=False)

    meta = {
        "model": args.model,
        "feature_set": args.feature_set,
        "target": args.target,
        "repeat": args.repeat,
        "n_pools": args.n_pools,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "feature_steps_per_candidate": args.feature_steps,
        "full_run_steps": args.full_run_steps,
        "n_candidates": int(len(cand)),
        "n_families": int(cand["family_id"].nunique()),
        "caveat": (
            "Multi-group pools mix OOF scores from different LOFO fold models. "
            "For N=16 with n_groups_requested=1 on 2D, pools are "
            "2-family pools (n_groups=1 expanded to 2 families). "
            "File summary_single_group.csv keeps its name for compatibility."
        ),
        "paths": {
            "per_pool": str(per_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "summary": str(sum_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "primary_N16": str(primary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    focus = summary[
        (summary["N"] == 16)
        & (summary["k"] == 5)
        & (summary["n_groups_requested"] == 1)
        & (summary["selector"] == "surrogate")
    ]
    print(focus.to_string(index=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
