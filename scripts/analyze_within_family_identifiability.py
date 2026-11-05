#!/usr/bin/env python3
"""Phase-1 diagnostic: within-family ranking identifiability + OOF Spearman.

Builds a per-family table from the Phase-1 seed-level dataset and optional
confirm OOF predictions. No PPO. Does not modify confirm artifacts.

Example::

    python scripts/analyze_within_family_identifiability.py
"""

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
    p.add_argument(
        "--se-multiplier",
        type=float,
        default=2.0,
        help="Identifiable if family range >= this * median within-cand SE",
    )
    p.add_argument(
        "--min-range",
        type=float,
        default=0.05,
        help="Absolute minimum SR range to count as identifiable",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "diagnostics_within_family",
    )
    return p.parse_args()


def _candidate_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    g = (
        df.groupby(["family_id", "candidate_id"], as_index=False)
        .agg(
            mean_y=(target, "mean"),
            std_y=(target, "std"),
            n_seeds=(target, "count"),
        )
        .sort_values(["family_id", "candidate_id"])
    )
    g["se_y"] = g["std_y"] / np.sqrt(g["n_seeds"].clip(lower=1))
    return g


def _pair_noise_stats(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in cand.groupby("family_id"):
        vals = g["mean_y"].to_numpy()
        ses = g["se_y"].fillna(0.0).to_numpy()
        n_pairs = 0
        lt_se = 0
        lt_2se = 0
        diffs = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                d = abs(float(vals[i] - vals[j]))
                se = float(np.sqrt(ses[i] ** 2 + ses[j] ** 2))
                n_pairs += 1
                diffs.append(d)
                if d < se:
                    lt_se += 1
                if d < 2.0 * se:
                    lt_2se += 1
        rows.append(
            {
                "family_id": fam,
                "n_pairs": n_pairs,
                "frac_pairs_lt_se": (lt_se / n_pairs) if n_pairs else np.nan,
                "frac_pairs_lt_2se": (lt_2se / n_pairs) if n_pairs else np.nan,
                "median_abs_pair_diff": float(np.median(diffs)) if diffs else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _oof_family_spearman(
    oof: pd.DataFrame,
    *,
    model: str,
    feature_set: str,
    target: str,
) -> pd.DataFrame:
    m = oof[
        (oof["model"] == model)
        & (oof["feature_set"] == feature_set)
        & (oof["target"] == target)
    ].copy()
    if m.empty:
        return pd.DataFrame(
            columns=["family_id", "oof_spearman_mean", "oof_spearman_std", "oof_top3_hit_mean"]
        )

    rows = []
    for (rep, fam), g in m.groupby(["repeat", "family_id"]):
        g = g.sort_values("candidate_id")
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)
        if len(g) < 3 or np.unique(y).size < 2 or np.unique(p).size < 2:
            sp = np.nan
        else:
            sp = float(spearmanr(y, p).correlation)
        # top-3 hit: true best in predicted top-3
        true_best = int(np.nanargmax(y))
        pred_top = set(np.argsort(-p)[: min(3, len(p))].tolist())
        hit = float(true_best in pred_top)
        rows.append(
            {
                "repeat": rep,
                "family_id": fam,
                "spearman": sp,
                "top3_hit": hit,
                "n": len(g),
            }
        )
    detail = pd.DataFrame(rows)
    out = (
        detail.groupby("family_id", as_index=False)
        .agg(
            oof_spearman_mean=("spearman", "mean"),
            oof_spearman_std=("spearman", "std"),
            oof_top3_hit_mean=("top3_hit", "mean"),
        )
    )
    return out


def build_table(
    dataset: pd.DataFrame,
    oof: pd.DataFrame | None,
    *,
    target: str,
    model: str,
    feature_set: str,
    se_multiplier: float,
    min_range: float,
) -> pd.DataFrame:
    cand = _candidate_table(dataset, target)
    fam = cand.groupby("family_id").agg(
        n_candidates=("candidate_id", "count"),
        mean_sr=("mean_y", "mean"),
        std_sr=("mean_y", "std"),
        min_sr=("mean_y", "min"),
        max_sr=("mean_y", "max"),
        median_seed_std=("std_y", "median"),
        median_seed_se=("se_y", "median"),
    )
    fam["range_sr"] = fam["max_sr"] - fam["min_sr"]
    pairs = _pair_noise_stats(cand)
    fam = fam.reset_index().merge(pairs, on="family_id", how="left")

    # Identifiability gate: absolute range AND relative to typical SE
    global_med_se = float(cand["se_y"].median())
    fam["identifiable"] = (fam["range_sr"] >= min_range) & (
        fam["range_sr"] >= se_multiplier * fam["median_seed_se"].fillna(global_med_se)
    )
    fam["identifiability_note"] = np.where(
        fam["identifiable"],
        "ok",
        np.where(
            fam["range_sr"] < min_range,
            "flat_range",
            "range_vs_se",
        ),
    )

    if oof is not None and not oof.empty:
        oof_stats = _oof_family_spearman(
            oof, model=model, feature_set=feature_set, target=target
        )
        fam = fam.merge(oof_stats, on="family_id", how="left")
    else:
        fam["oof_spearman_mean"] = np.nan
        fam["oof_spearman_std"] = np.nan
        fam["oof_top3_hit_mean"] = np.nan

    fam["global_median_seed_se"] = global_med_se
    fam["se_multiplier"] = se_multiplier
    fam["min_range_threshold"] = min_range
    return fam.sort_values("oof_spearman_mean", ascending=True, na_position="last")


def write_report(table: pd.DataFrame, out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "identifiability_table.csv"
    table.to_csv(csv_path, index=False)

    id_mask = table["identifiable"].astype(bool)
    n_id = int(id_mask.sum())
    n_all = int(len(table))

    def _mean(col: str, mask: pd.Series | None = None) -> float:
        s = table[col] if mask is None else table.loc[mask, col]
        return float(s.mean()) if len(s) and s.notna().any() else float("nan")

    lines = [
        "# Within-family identifiability diagnostic",
        "",
        f"- Generated by `scripts/analyze_within_family_identifiability.py`",
        f"- Model/features/target: `{meta['model']}` / `{meta['feature_set']}` / `{meta['target']}`",
        f"- Gate: `range >= max({meta['min_range']}, {meta['se_multiplier']} * median_seed_SE)`",
        f"- Families identifiable: **{n_id} / {n_all}**",
        "",
        "## Macro metrics",
        "",
        f"| Set | Mean OOF Spearman | Mean Top-3 hit |",
        f"|-----|------------------:|---------------:|",
        f"| All families | {_mean('oof_spearman_mean'):.4f} | {_mean('oof_top3_hit_mean'):.4f} |",
        f"| Identifiable only | {_mean('oof_spearman_mean', id_mask):.4f} | {_mean('oof_top3_hit_mean', id_mask):.4f} |",
        f"| Non-identifiable | {_mean('oof_spearman_mean', ~id_mask):.4f} | {_mean('oof_top3_hit_mean', ~id_mask):.4f} |",
        "",
        "## Per-family",
        "",
        "| family | n | range | med SE | identifiable | note | OOF Spearman | Top-3 |",
        "|--------|--:|------:|-------:|:------------:|------|-------------:|------:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['family_id']} | {int(r['n_candidates'])} | {r['range_sr']:.4f} | "
            f"{r['median_seed_se']:.4f} | {bool(r['identifiable'])} | {r['identifiability_note']} | "
            f"{r['oof_spearman_mean']:.3f} | {r['oof_top3_hit_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Policy implication",
            "",
            "- **Primary** within-family claim: mean Spearman / Top-3 on identifiable families.",
            "- **Secondary**: all-family macro (includes flat/noisy families for transparency).",
            "- See `research_audit/METRIC_POLICY.md`.",
            "",
        ]
    )
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        **meta,
        "n_families": n_all,
        "n_identifiable": n_id,
        "mean_oof_spearman_all": _mean("oof_spearman_mean"),
        "mean_oof_spearman_identifiable": _mean("oof_spearman_mean", id_mask),
        "mean_top3_all": _mean("oof_top3_hit_mean"),
        "mean_top3_identifiable": _mean("oof_top3_hit_mean", id_mask),
        "table_csv": str(csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "report_md": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    ds = pd.read_csv(args.dataset)
    if args.target not in ds.columns:
        print(f"Missing target column {args.target}", file=sys.stderr)
        return 1

    oof = None
    if args.oof.is_file():
        oof = pd.read_csv(args.oof)
    else:
        print(f"Warning: OOF missing ({args.oof}); Spearman columns will be NaN")

    table = build_table(
        ds,
        oof,
        target=args.target,
        model=args.model,
        feature_set=args.feature_set,
        se_multiplier=args.se_multiplier,
        min_range=args.min_range,
    )
    meta = {
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "oof": str(args.oof).replace("\\", "/") if args.oof else None,
        "model": args.model,
        "feature_set": args.feature_set,
        "target": args.target,
        "se_multiplier": args.se_multiplier,
        "min_range": args.min_range,
    }
    report = write_report(table, args.out_dir, meta)
    print(f"Wrote {args.out_dir / 'identifiability_table.csv'}")
    print(f"Wrote {report}")
    print(f"Wrote {args.out_dir / 'summary.json'}")
    print(
        f"identifiable={int(table['identifiable'].sum())}/{len(table)} "
        f"spearman_all={table['oof_spearman_mean'].mean():.4f} "
        f"spearman_id={table.loc[table['identifiable'], 'oof_spearman_mean'].mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
