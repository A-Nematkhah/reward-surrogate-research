#!/usr/bin/env python3
"""Family-paired contrast: ranking model vs ExtraTrees on identifiable families."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.model_comparison import identifiable_families_from_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "benchmark_phase3_ranking"
        / "oof_predictions.csv",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "phase1"
        / "dataset"
        / "reward_performance_dataset.csv",
    )
    p.add_argument("--baseline", default="extra_trees")
    p.add_argument("--challenger", default="pairwise_logistic")
    p.add_argument("--feature-set", default="early+math+behavioral")
    p.add_argument("--target", default="success_rate")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "benchmark_phase3_ranking",
    )
    return p.parse_args()


def _family_spearman(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for fam, g in df.groupby("family_id"):
        if len(g) < 3 or g["y_true"].nunique() < 2 or g["y_pred"].nunique() < 2:
            out[str(fam)] = float("nan")
            continue
        sp = spearmanr(g["y_true"], g["y_pred"]).correlation
        out[str(fam)] = float(sp) if sp is not None and np.isfinite(sp) else float("nan")
    return out


def main() -> int:
    args = parse_args()
    oof = pd.read_csv(args.oof)
    m = oof[
        (oof["feature_set"] == args.feature_set)
        & (oof["target"] == args.target)
        & (oof["model"].isin([args.baseline, args.challenger]))
    ].copy()
    if m.empty:
        print(f"No OOF rows in {args.oof}", file=sys.stderr)
        return 1

    # Identifiable families from seed-mean labels
    ds = pd.read_csv(args.dataset)
    cand = (
        ds.groupby(["family_id", "candidate_id"], as_index=False)[args.target]
        .mean()
        .rename(columns={args.target: "y"})
    )
    y = cand["y"].to_numpy()
    g = cand["family_id"].to_numpy()
    # SE map
    se = (
        ds.groupby(["family_id", "candidate_id"])[args.target]
        .agg(std="std", n="count")
        .reset_index()
    )
    se["se"] = se["std"] / np.sqrt(se["n"].clip(lower=1))
    se_map = se.groupby("family_id")["se"].median().to_dict()
    allow = identifiable_families_from_labels(
        y, g, min_range=0.05, se_multiplier=2.0, family_median_se=se_map, min_size=4
    )

    rows = []
    for rep, sub in m.groupby("repeat"):
        base = _family_spearman(sub[sub["model"] == args.baseline])
        chal = _family_spearman(sub[sub["model"] == args.challenger])
        fams = sorted(set(base) | set(chal))
        for fam in fams:
            rows.append(
                {
                    "repeat": int(rep),
                    "family_id": fam,
                    "identifiable": fam in allow,
                    "spearman_baseline": base.get(fam, np.nan),
                    "spearman_challenger": chal.get(fam, np.nan),
                    "delta_challenger_minus_baseline": (
                        chal.get(fam, np.nan) - base.get(fam, np.nan)
                        if np.isfinite(chal.get(fam, np.nan))
                        and np.isfinite(base.get(fam, np.nan))
                        else np.nan
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "family_paired_deltas.csv"
    detail.to_csv(detail_path, index=False)

    id_detail = detail[detail["identifiable"]].copy()
    # Average over repeats per family then bootstrap families
    per_fam = (
        id_detail.groupby("family_id", as_index=False)["delta_challenger_minus_baseline"]
        .mean()
        .dropna()
    )
    deltas = per_fam["delta_challenger_minus_baseline"].to_numpy()
    rng = np.random.default_rng(args.seed)
    if len(deltas) == 0:
        summary = {"error": "no identifiable family deltas"}
    else:
        boots = []
        for _ in range(args.n_boot):
            samp = rng.choice(deltas, size=len(deltas), replace=True)
            boots.append(float(np.mean(samp)))
        boots_arr = np.asarray(boots)
        summary = {
            "baseline": args.baseline,
            "challenger": args.challenger,
            "feature_set": args.feature_set,
            "n_identifiable_families": int(len(deltas)),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "boot_mean": float(np.mean(boots_arr)),
            "boot_ci_lo": float(np.quantile(boots_arr, 0.05)),
            "boot_ci_hi": float(np.quantile(boots_arr, 0.95)),
            "ci_excludes_zero": bool(
                np.quantile(boots_arr, 0.05) > 0 or np.quantile(boots_arr, 0.95) < 0
            ),
            "decision": (
                "reject_H0_challenger_better"
                if np.quantile(boots_arr, 0.05) > 0
                else (
                    "reject_H0_baseline_better"
                    if np.quantile(boots_arr, 0.95) < 0
                    else "fail_to_reject_keep_extra_trees"
                )
            ),
        }

    summary_path = args.out_dir / "ranking_contrast_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
