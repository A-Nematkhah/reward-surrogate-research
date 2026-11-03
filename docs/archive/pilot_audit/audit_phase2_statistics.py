#!/usr/bin/env python3
"""Phase 2 statistical audit: recompute pooled vs fold Spearman from OOF preds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase2_audit"


def safe_spearman(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 2:
        return float("nan"), "too_few"
    if np.unique(p).size == 1 or np.unique(y).size == 1:
        return float("nan"), "constant"
    sp = spearmanr(y, p).correlation
    if sp is None or not np.isfinite(sp):
        return float("nan"), "undefined"
    return float(sp), "ok"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(ROOT / "experiments/phase2/predictions/all_predictions.csv")
    metrics = pd.read_csv(ROOT / "experiments/phase2/surrogate_metrics.csv")
    sens = pd.read_csv(ROOT / "results/phase2_fold_sensitivity.csv")
    meta = pd.read_csv(ROOT / "experiments/phase2/features/features_meta.csv")
    static_cols = list(
        pd.read_csv(
            ROOT / "experiments/phase2/features/features_static.csv", nrows=0
        ).columns
    )
    targets = pd.read_csv(ROOT / "experiments/phase2/features/features_targets.csv")

    g = preds[preds["split"] == "grouped_cv"].copy()
    rows = []
    for (model, fs, target), sub in g.groupby(["model", "feature_set", "target"]):
        pooled_sp, pooled_status = safe_spearman(sub["y_true"], sub["y_pred"])
        fold_sps = []
        fold_ns = []
        fold_statuses = []
        fold_pred_nunique = []
        fold_families = []
        for fold, fsub in sub.groupby("fold"):
            sp, st = safe_spearman(fsub["y_true"], fsub["y_pred"])
            fold_sps.append(sp)
            fold_ns.append(int(len(fsub)))
            fold_statuses.append(st)
            fold_pred_nunique.append(int(fsub["y_pred"].nunique(dropna=True)))
            fold_families.append(sorted(fsub["family_id"].unique().tolist()))
        fold_sps_arr = np.asarray(fold_sps, dtype=float)
        finite = fold_sps_arr[np.isfinite(fold_sps_arr)]
        rep = metrics[
            (metrics["split"] == "grouped_cv_oof")
            & (metrics["model"] == model)
            & (metrics["feature_set"] == fs)
            & (metrics["target"] == target)
        ]
        rep_sp = float(rep.iloc[0]["spearman"]) if len(rep) else float("nan")
        rep_mae = float(rep.iloc[0]["mae"]) if len(rep) else float("nan")
        rows.append(
            {
                "model": model,
                "feature_set": fs,
                "target": target,
                "n_oof": int(len(sub)),
                "n_candidates": int(sub["candidate_id"].nunique()),
                "pooled_spearman_recomputed": pooled_sp,
                "pooled_status": pooled_status,
                "reported_oof_spearman": rep_sp,
                "pooled_minus_reported": (
                    pooled_sp - rep_sp
                    if np.isfinite(pooled_sp) and np.isfinite(rep_sp)
                    else float("nan")
                ),
                "fold_spearman_mean": float(np.nanmean(fold_sps_arr))
                if finite.size
                else float("nan"),
                "fold_spearman_std": float(np.nanstd(fold_sps_arr, ddof=1))
                if finite.size >= 2
                else float("nan"),
                "n_folds_defined_spearman": int(finite.size),
                "fold_ns": json.dumps(fold_ns),
                "fold_statuses": json.dumps(fold_statuses),
                "fold_pred_nunique": json.dumps(fold_pred_nunique),
                "fold_families": json.dumps(fold_families),
                "fold_spearmans": json.dumps(
                    [None if not np.isfinite(x) else float(x) for x in fold_sps_arr]
                ),
                "reported_mae": rep_mae,
                "mae_recomputed": float(
                    mean_absolute_error(sub["y_true"], sub["y_pred"])
                ),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "recomputed_spearman_audit.csv", index=False)

    print("=== POOLED vs REPORTED ===")
    diffs = audit["pooled_minus_reported"].abs()
    print("max abs diff:", float(diffs.max(skipna=True)))
    print("mismatches > 1e-8:", int((diffs > 1e-8).sum()))

    print("\n=== MEAN/MEDIAN baseline success_rate/static ===")
    baseline_notes = []
    for model in ["mean", "median"]:
        sub = g[
            (g["model"] == model)
            & (g["feature_set"] == "static")
            & (g["target"] == "success_rate")
        ]
        pooled = safe_spearman(sub["y_true"], sub["y_pred"])
        print(model, "pooled", pooled)
        fold_stats = []
        for fold, fsub in sub.groupby("fold"):
            sp = safe_spearman(fsub["y_true"], fsub["y_pred"])
            print(
                " fold",
                fold,
                "n",
                len(fsub),
                "families",
                sorted(fsub["family_id"].unique()),
                "y_pred_nunique",
                fsub["y_pred"].nunique(),
                "pred",
                float(fsub["y_pred"].iloc[0]),
                "spearman",
                sp,
            )
            fold_stats.append(
                {
                    "fold": int(fold),
                    "pred": float(fsub["y_pred"].iloc[0]),
                    "y_mean": float(fsub["y_true"].mean()),
                }
            )
        fs = pd.DataFrame(fold_stats)
        fold_corr = safe_spearman(fs["y_mean"], fs["pred"])
        print(" fold-level corr(pred, heldout_y_mean)", fold_corr)
        baseline_notes.append(
            {
                "model": model,
                "pooled_spearman": pooled[0],
                "pooled_status": pooled[1],
                "fold_corr_pred_vs_heldout_mean": fold_corr[0],
                "fold_table": fold_stats,
            }
        )

    with (OUT / "baseline_mean_median_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_notes, handle, indent=2)

    print("\n=== Family separation GroupKFold ===")
    families = meta["family_id"].to_numpy()
    gkf = GroupKFold(n_splits=4)
    xdummy = np.zeros((len(meta), 1))
    fold_info = []
    for i, (tr, te) in enumerate(gkf.split(xdummy, groups=families)):
        tr_f = set(families[tr].tolist())
        te_f = set(families[te].tolist())
        info = {
            "fold": i,
            "train_families": sorted(tr_f),
            "test_families": sorted(te_f),
            "overlap": sorted(tr_f & te_f),
            "n_test": int(len(te)),
        }
        fold_info.append(info)
        print(info)

    with (OUT / "grouped_cv_family_splits.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_info, handle, indent=2)

    print("\n=== Family one-hot in static ===")
    print([c for c in static_cols if c.startswith("family_")])

    print("\n=== mean_goal_time NaNs ===")
    nan_mask = targets["mean_goal_time"].isna()
    print("count", int(nan_mask.sum()))
    print(meta.loc[nan_mask, ["candidate_id", "family_id"]].to_string(index=False))

    # Compare sensitivity CSV nanmean behavior vs recomputed
    merged = sens.merge(
        audit[
            [
                "model",
                "feature_set",
                "target",
                "fold_spearman_mean",
                "fold_spearman_std",
                "n_folds_defined_spearman",
            ]
        ],
        on=["model", "feature_set", "target"],
        how="left",
    )
    merged["mean_diff"] = merged["spearman_mean"] - merged["fold_spearman_mean"]
    merged.to_csv(OUT / "sensitivity_vs_recomputed.csv", index=False)
    print("\n=== sensitivity vs recomputed fold means ===")
    print("abs mean_diff max:", float(merged["mean_diff"].abs().max(skipna=True)))
    print("sens NaN spearman_mean rows:", int(sens["spearman_mean"].isna().sum()))

    nonbase = audit[~audit["model"].isin(["mean", "median"])].copy()
    print("\n=== BEST BY POOLED SPEARMAN ===")
    best_rows = []
    for t, sub in nonbase.groupby("target"):
        b = sub.sort_values("pooled_spearman_recomputed", ascending=False).iloc[0]
        print(
            f"{t}: {b['model']}/{b['feature_set']} pooled={b['pooled_spearman_recomputed']:.3f} "
            f"fold_mean={b['fold_spearman_mean']:.3f} fold_std={b['fold_spearman_std']:.3f} "
            f"n_defined={b['n_folds_defined_spearman']} folds={b['fold_spearmans']}"
        )
        best_rows.append(b.to_dict())

    print("\n=== BEST BY FOLD-MEAN SPEARMAN ===")
    for t, sub in nonbase.groupby("target"):
        b = sub.sort_values("fold_spearman_mean", ascending=False).iloc[0]
        print(
            f"{t}: {b['model']}/{b['feature_set']} fold_mean={b['fold_spearman_mean']:.3f} "
            f"fold_std={b['fold_spearman_std']:.3f} pooled={b['pooled_spearman_recomputed']:.3f} "
            f"folds={b['fold_spearmans']}"
        )

    print("\n=== FEATURE SET pooled Spearman summary (excl mean/median) ===")
    fs = (
        nonbase.groupby(["feature_set", "target"])["pooled_spearman_recomputed"]
        .agg(["mean", "max", "median"])
        .reset_index()
    )
    print(fs.to_string(index=False))
    fs.to_csv(OUT / "feature_set_pooled_summary.csv", index=False)

    print("\n=== UNSTABLE fold_std>0.5 with >=3 defined folds ===")
    unstable = nonbase[
        (nonbase["fold_spearman_std"] > 0.5)
        & (nonbase["n_folds_defined_spearman"] >= 3)
    ].sort_values("fold_spearman_std", ascending=False)
    print(
        unstable[
            [
                "model",
                "feature_set",
                "target",
                "pooled_spearman_recomputed",
                "fold_spearman_mean",
                "fold_spearman_std",
                "fold_spearmans",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    unstable.to_csv(OUT / "unstable_configs.csv", index=False)

    # duplicates
    print("\n=== duplicate OOF candidates ===")
    dup_found = False
    for (model, fs_name, target), sub in g.groupby(
        ["model", "feature_set", "target"]
    ):
        if sub["candidate_id"].duplicated().any():
            print("DUPLICATES", model, fs_name, target)
            dup_found = True
    if not dup_found:
        print("none")

    # random vs grouped
    rand = metrics[metrics["split"] == "random_holdout"]
    oof = metrics[metrics["split"] == "grouped_cv_oof"]
    cmp = oof.merge(
        rand, on=["model", "feature_set", "target"], suffixes=("_grouped", "_random")
    )
    cmp = cmp[~cmp["model"].isin(["mean", "median"])]
    print("\n=== random vs grouped mean spearman ===")
    print(
        cmp.groupby("target")[["spearman_grouped", "spearman_random"]].mean().to_string()
    )
    cmp.groupby("target")[["spearman_grouped", "spearman_random"]].mean().to_csv(
        OUT / "random_vs_grouped_spearman.csv"
    )

    # MAE baseline comparison table for success_rate
    print("\n=== MAE comparison success_rate (pooled recomputed) ===")
    sr = audit[audit["target"] == "success_rate"][
        ["model", "feature_set", "mae_recomputed", "pooled_spearman_recomputed", "fold_spearman_mean", "fold_spearman_std", "n_folds_defined_spearman"]
    ].sort_values("mae_recomputed")
    print(sr.to_string(index=False))
    sr.to_csv(OUT / "success_rate_mae_ranking.csv", index=False)

    primary = audit[
        [
            "model",
            "feature_set",
            "target",
            "n_oof",
            "pooled_spearman_recomputed",
            "fold_spearman_mean",
            "fold_spearman_std",
            "n_folds_defined_spearman",
            "mae_recomputed",
            "reported_oof_spearman",
            "reported_mae",
            "fold_spearmans",
            "fold_statuses",
            "fold_ns",
        ]
    ].copy()
    primary.to_csv(OUT / "corrected_primary_metrics.csv", index=False)

    # Highlight reported-best configs reliability
    reported_best = [
        ("success_rate", "gaussian_process", "behavioral"),
        ("collision_rate", "random_forest", "static"),
        ("timeout_rate", "ridge", "static"),
        ("mean_goal_time", "random_forest", "behavioral"),
        ("mean_path_length", "extra_trees", "static"),
    ]
    print("\n=== RELIABILITY OF REPORT-CLAIMED BEST CONFIGS ===")
    reliability = []
    for target, model, fs_name in reported_best:
        row = audit[
            (audit["target"] == target)
            & (audit["model"] == model)
            & (audit["feature_set"] == fs_name)
        ].iloc[0]
        print(
            target,
            f"{model}/{fs_name}",
            "pooled",
            row["pooled_spearman_recomputed"],
            "fold_mean",
            row["fold_spearman_mean"],
            "fold_std",
            row["fold_spearman_std"],
            "defined",
            row["n_folds_defined_spearman"],
            "folds",
            row["fold_spearmans"],
        )
        reliability.append(row.to_dict())
    pd.DataFrame(reliability).to_csv(OUT / "reported_best_reliability.csv", index=False)

    print("\nWrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
