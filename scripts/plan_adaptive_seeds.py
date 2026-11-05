#!/usr/bin/env python3
"""Phase 4: propose extra PPO seeds for close-call / weak-ranking families.

Does **not** launch training. Human must authorize any PPO budget.

Priority:
1. Identifiable families with weak confirm OOF Spearman
2. Within-family candidate pairs with |Δ mean SR| < 2·pooled_SE
3. Skip non-identifiable / flat families (e.g. distance_dense)

Example::

    python scripts/plan_adaptive_seeds.py --max-extra-runs 36
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "phase1"
        / "dataset"
        / "reward_performance_dataset.csv",
    )
    p.add_argument(
        "--identifiability",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "diagnostics_within_family"
        / "identifiability_table.csv",
    )
    p.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "model_comparison_confirm"
        / "oof_predictions.csv",
    )
    p.add_argument("--model", default="extra_trees")
    p.add_argument("--feature-set", default="early+math+behavioral")
    p.add_argument("--target", default="success_rate")
    p.add_argument("--se-mult", type=float, default=2.0, help="Close if |Δ| < se_mult * SE")
    p.add_argument("--extra-seeds-per-cand", type=int, default=3)
    p.add_argument("--max-extra-runs", type=int, default=36)
    p.add_argument(
        "--weak-spearman-threshold",
        type=float,
        default=0.55,
        help="Families below this confirm OOF Spearman get priority",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "adaptive_seed_plan",
    )
    return p.parse_args()


def _candidate_stats(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for (fam, cid), g in df.groupby(["family_id", "candidate_id"]):
        seeds = sorted({int(x) for x in g["training_seed"].dropna().tolist()})
        y = g[target].to_numpy(dtype=float)
        rows.append(
            {
                "family_id": fam,
                "candidate_id": cid,
                "mean_y": float(np.nanmean(y)),
                "std_y": float(np.nanstd(y, ddof=1)) if len(y) > 1 else 0.0,
                "n_seeds": int(len(y)),
                "existing_seeds": seeds,
            }
        )
    out = pd.DataFrame(rows)
    out["se_y"] = out["std_y"] / np.sqrt(out["n_seeds"].clip(lower=1))
    return out


def _confirm_family_spearman(oof_path: Path, model: str, feature_set: str, target: str) -> dict[str, float]:
    if not oof_path.is_file():
        return {}
    oof = pd.read_csv(oof_path)
    m = oof[
        (oof["model"] == model)
        & (oof["feature_set"] == feature_set)
        & (oof["target"] == target)
    ]
    from scipy.stats import spearmanr

    rows = []
    for (rep, fam), g in m.groupby(["repeat", "family_id"]):
        if len(g) < 3 or g["y_true"].nunique() < 2 or g["y_pred"].nunique() < 2:
            sp = np.nan
        else:
            r = spearmanr(g["y_true"], g["y_pred"]).correlation
            sp = float(r) if r is not None and np.isfinite(r) else np.nan
        rows.append({"family_id": fam, "spearman": sp})
    if not rows:
        return {}
    tab = pd.DataFrame(rows).groupby("family_id")["spearman"].mean()
    return {str(k): float(v) for k, v in tab.items()}


def main() -> int:
    args = parse_args()
    ds = pd.read_csv(args.dataset)
    if "training_status" in ds.columns:
        ds = ds[ds["training_status"] == "completed"].copy()

    cand = _candidate_stats(ds, args.target)
    id_tab = pd.read_csv(args.identifiability) if args.identifiability.is_file() else None
    id_map = {}
    if id_tab is not None:
        id_map = {
            str(r.family_id): bool(r.identifiable)
            for _, r in id_tab.iterrows()
        }

    fam_sp = _confirm_family_spearman(
        args.oof, args.model, args.feature_set, args.target
    )

    # Close pairs
    pair_rows = []
    for fam, g in cand.groupby("family_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                d = abs(float(a.mean_y - b.mean_y))
                se = float(np.sqrt(float(a.se_y or 0) ** 2 + float(b.se_y or 0) ** 2))
                close = d < args.se_mult * se if se > 0 else d < 1e-6
                pair_rows.append(
                    {
                        "family_id": fam,
                        "candidate_a": a.candidate_id,
                        "candidate_b": b.candidate_id,
                        "mean_a": a.mean_y,
                        "mean_b": b.mean_y,
                        "abs_diff": d,
                        "pooled_se": se,
                        "close": close,
                    }
                )
    pairs = pd.DataFrame(pair_rows)

    # Score candidates for extra seeds
    close_counts = {}
    for _, r in pairs[pairs["close"]].iterrows():
        for cid in (r.candidate_a, r.candidate_b):
            close_counts[cid] = close_counts.get(cid, 0) + 1

    plan_rows = []
    for _, r in cand.iterrows():
        fam = str(r.family_id)
        identifiable = id_map.get(fam, True)
        sp = fam_sp.get(fam, np.nan)
        weak = bool(np.isfinite(sp) and sp < args.weak_spearman_threshold)
        n_close = int(close_counts.get(r.candidate_id, 0))
        # Priority score (higher = more urgent)
        score = 0.0
        reasons = []
        if not identifiable:
            # Skip flat families by default
            plan_rows.append(
                {
                    "candidate_id": r.candidate_id,
                    "family_id": fam,
                    "identifiable": False,
                    "confirm_oof_spearman": sp,
                    "mean_sr": r.mean_y,
                    "n_seeds": int(r.n_seeds),
                    "n_close_pairs": n_close,
                    "priority_score": -1.0,
                    "recommended_extra_seeds": 0,
                    "reason": "skip_non_identifiable_family",
                    "existing_seeds": r.existing_seeds,
                }
            )
            continue
        if weak:
            score += 3.0
            reasons.append("weak_family_spearman")
        if n_close > 0:
            score += 1.0 + 0.25 * n_close
            reasons.append(f"close_pairs={n_close}")
        if float(r.std_y or 0) > 0.05:
            score += 0.5
            reasons.append("high_seed_std")
        extra = args.extra_seeds_per_cand if score > 0 else 0
        plan_rows.append(
            {
                "candidate_id": r.candidate_id,
                "family_id": fam,
                "identifiable": True,
                "confirm_oof_spearman": sp,
                "mean_sr": r.mean_y,
                "n_seeds": int(r.n_seeds),
                "seed_std": float(r.std_y) if pd.notna(r.std_y) else np.nan,
                "n_close_pairs": n_close,
                "priority_score": score,
                "recommended_extra_seeds": int(extra),
                "reason": ",".join(reasons) if reasons else "none",
                "existing_seeds": r.existing_seeds,
            }
        )

    plan = pd.DataFrame(plan_rows).sort_values(
        ["priority_score", "n_close_pairs", "family_id", "candidate_id"],
        ascending=[False, False, True, True],
    )

    # Budget allocation: take top candidates until max_extra_runs
    selected = []
    budget = 0
    for _, r in plan.iterrows():
        if r.recommended_extra_seeds <= 0:
            continue
        need = int(r.recommended_extra_seeds)
        if budget + need > args.max_extra_runs:
            remain = args.max_extra_runs - budget
            if remain <= 0:
                break
            need = remain
        # Propose new seed ids not in existing
        existing = set(r.existing_seeds) if isinstance(r.existing_seeds, list) else set()
        new_seeds = []
        s = 0
        while len(new_seeds) < need:
            if s not in existing:
                new_seeds.append(s)
            s += 1
            if s > 100:
                break
        selected.append(
            {
                "candidate_id": r.candidate_id,
                "family_id": r.family_id,
                "priority_score": r.priority_score,
                "reason": r.reason,
                "confirm_oof_spearman": r.confirm_oof_spearman,
                "n_close_pairs": r.n_close_pairs,
                "mean_sr": r.mean_sr,
                "n_seeds_now": r.n_seeds,
                "proposed_new_seeds": new_seeds,
                "n_extra_runs": len(new_seeds),
            }
        )
        budget += len(new_seeds)
        if budget >= args.max_extra_runs:
            break

    selected_df = pd.DataFrame(selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "candidate_priority.csv"
    sel_path = args.out_dir / "selected_extra_seeds.csv"
    pairs_path = args.out_dir / "close_pairs.csv"
    plan.to_csv(plan_path, index=False)
    selected_df.to_csv(sel_path, index=False)
    pairs[pairs["close"]].to_csv(pairs_path, index=False)

    n_close_pairs = int(pairs["close"].sum())
    n_weak_fam = int(
        sum(
            1
            for fam, sp in fam_sp.items()
            if id_map.get(fam, True) and np.isfinite(sp) and sp < args.weak_spearman_threshold
        )
    )
    summary = {
        "max_extra_runs": args.max_extra_runs,
        "extra_seeds_per_cand": args.extra_seeds_per_cand,
        "se_mult": args.se_mult,
        "n_close_pairs": n_close_pairs,
        "n_weak_identifiable_families": n_weak_fam,
        "n_candidates_selected": int(len(selected_df)),
        "n_extra_runs_planned": int(selected_df["n_extra_runs"].sum()) if len(selected_df) else 0,
        "selected_families": sorted(selected_df["family_id"].unique().tolist())
        if len(selected_df)
        else [],
        "launches_ppo": False,
        "authorization_required": True,
        "note": "Do not overwrite experiments/phase1; use a new run root if executing.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Adaptive seed plan (Phase 4) — no PPO launched",
        "",
        f"- Close pairs (|Δ| < {args.se_mult}·SE): **{n_close_pairs}**",
        f"- Weak identifiable families (Spearman < {args.weak_spearman_threshold}): **{n_weak_fam}**",
        f"- Budget cap: **{args.max_extra_runs}** extra seed-runs",
        f"- Selected candidates: **{summary['n_candidates_selected']}** → **{summary['n_extra_runs_planned']}** runs",
        f"- Families touched: `{', '.join(summary['selected_families']) or 'none'}`",
        "",
        "## Authorization gate",
        "",
        "This plan does **not** start PPO. To execute later:",
        "1. Review `selected_extra_seeds.csv`",
        "2. Use a **new** experiment root (do not overwrite phase1 archive)",
        "3. Cap wall-clock to the planned run count",
        "",
        "## Top selected",
        "",
        "| candidate | family | extra seeds | score | reason |",
        "|-----------|--------|------------:|------:|--------|",
    ]
    for _, r in selected_df.head(25).iterrows():
        lines.append(
            f"| {r.candidate_id} | {r.family_id} | {int(r.n_extra_runs)} | "
            f"{r.priority_score:.2f} | {r.reason} |"
        )
    report = args.out_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {plan_path}")
    print(f"Wrote {sel_path}")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
