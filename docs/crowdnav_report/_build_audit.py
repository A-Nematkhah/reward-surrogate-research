#!/usr/bin/env python3
"""Read-only CrowdNav audit → docs/crowdnav_report/ artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#8E0202",
    "secondary": "#AC4B52",
    "muted": "#D1C7C7",
    "gray": "#838282",
    "ink": "#1A1A1A",
    "white": "#FFFFFF",
}
FAMILY_COLORS = [
    "#8E0202",
    "#AC4B52",
    "#838282",
    "#1A1A1A",
    "#C47A7E",
    "#5A5A5A",
    "#A33A3A",
    "#6E4A4A",
    "#9A8A8A",
    "#4A3030",
    "#B85C5C",
    "#707070",
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "figure.facecolor": PALETTE["white"],
        "axes.facecolor": PALETTE["white"],
    }
)

NUMBERS: list[dict] = []


def add_number(**kwargs) -> None:
    NUMBERS.append(kwargs)


def savefig(name: str) -> Path:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def load_run(name: str) -> dict:
    exp = ROOT / "experiments" / name
    res = ROOT / "results" / name
    ds = pd.read_csv(exp / "dataset" / "reward_performance_dataset.csv")
    meta = pd.read_csv(exp / "features" / "features_meta.csv")
    tgt = pd.read_csv(exp / "features" / "features_targets.csv")
    lb = pd.read_csv(res / "leaderboard.csv") if (res / "leaderboard.csv").exists() else None
    ms = pd.read_csv(exp / "model_comparison" / "metrics_summary.csv")
    fold = pd.read_csv(exp / "model_comparison" / "metrics_by_fold.csv")
    oof = pd.read_csv(exp / "model_comparison" / "oof_predictions.csv")
    topk = pd.read_csv(exp / "model_comparison" / "topk_metrics.csv")
    phase1 = json.loads((exp / "phase1_summary.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((exp / "phase1_manifest.json").read_text(encoding="utf-8-sig"))
    outliers = json.loads((exp / "outlier_seeds.json").read_text(encoding="utf-8-sig"))
    cand_summary = json.loads((exp / "candidates_summary.json").read_text(encoding="utf-8-sig"))
    cand_json = json.loads((exp / "candidates.json").read_text(encoding="utf-8-sig"))
    return {
        "name": name,
        "exp": exp,
        "res": res,
        "ds": ds,
        "meta": meta,
        "tgt": tgt,
        "lb": lb,
        "ms": ms,
        "fold": fold,
        "oof": oof,
        "topk": topk,
        "phase1": phase1,
        "manifest": manifest,
        "outliers": outliers,
        "cand_summary": cand_summary,
        "cand_json": cand_json,
    }


def seed_agg(ds: pd.DataFrame) -> pd.DataFrame:
    return (
        ds.groupby(["candidate_id", "family_id"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            collision_rate=("collision_rate", "mean"),
            timeout_rate=("timeout_rate", "mean"),
            mean_goal_time=("mean_goal_time", "mean"),
            mean_path_length=("mean_path_length", "mean"),
            mean_evaluation_return=("mean_evaluation_return", "mean"),
            n_seeds=("success_rate", "size"),
            train_duration_sec=("train_duration_sec", "mean"),
            total_timesteps=("total_timesteps", "first"),
            n_eval_episodes=("n_eval_episodes", "first"),
            env_id=("env_id", "first"),
        )
    )


def primary_row(ms: pd.DataFrame) -> pd.Series:
    sub = ms[
        (ms["model"] == "extra_trees")
        & (ms["feature_set"] == "early+math+behavioral")
        & (ms["target"] == "success_rate")
    ]
    if "repeat" in sub.columns:
        # metrics_summary often already aggregated; if repeats present take mean later
        pass
    return sub.iloc[0]


def fold_family_table(fold: pd.DataFrame) -> pd.DataFrame:
    sub = fold[
        (fold["model"] == "extra_trees")
        & (fold["feature_set"] == "early+math+behavioral")
        & (fold["target"] == "success_rate")
    ].copy()
    # column name variants
    spear_col = None
    for c in sub.columns:
        if c in {"within_family_spearman", "spearman"} and "family" in c:
            spear_col = c
    if spear_col is None:
        # fold file uses spearman as fold-level within? check
        if "within_family_spearman" in sub.columns:
            spear_col = "within_family_spearman"
        elif "spearman" in sub.columns:
            spear_col = "spearman"
    top3_col = "within_family_top3_hit" if "within_family_top3_hit" in sub.columns else None
    g = sub.groupby("family_id").agg(
        within_spearman_mean=(spear_col, "mean"),
        within_spearman_std=(spear_col, "std"),
        **({ "top3_hit_mean": (top3_col, "mean") } if top3_col else {}),
        n_repeats=(spear_col, "size"),
    )
    return g.reset_index()


def label_stats(agg: pd.DataFrame, run_tag: str, source: str) -> dict:
    y = agg["success_rate"]
    stats_d = {
        "n_candidates": int(len(agg)),
        "n_families": int(agg["family_id"].nunique()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std(ddof=1)),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean_seed_std": float(agg["success_std"].mean()),
        "median_seed_std": float(agg["success_std"].median()),
        "frac_seed_std_gt_0_05": float((agg["success_std"] > 0.05).mean()),
        "frac_seed_std_gt_0_10": float((agg["success_std"] > 0.10).mean()),
        "frac_sr_lt_0_05": float((y < 0.05).mean()),
        "collision_mean": float(agg["collision_rate"].mean()),
        "timeout_mean": float(agg["timeout_rate"].mean()),
    }
    for k, v in stats_d.items():
        add_number(
            metric=f"label_{k}",
            run=run_tag,
            model="",
            features="",
            value=v,
            source=source,
        )
    return stats_d


def main() -> None:
    r50 = load_run("crowdnav_v1_50k")
    r150 = load_run("crowdnav_v4_full_150k")

    a50 = seed_agg(r50["ds"])
    a150 = seed_agg(r150["ds"])
    a50.to_csv(OUT / "labels_50k_by_candidate.csv", index=False)
    a150.to_csv(OUT / "labels_150k_by_candidate.csv", index=False)

    s50 = label_stats(
        a50,
        "50k",
        "experiments/crowdnav_v1_50k/dataset/reward_performance_dataset.csv (seed-mean)",
    )
    s150 = label_stats(
        a150,
        "150k",
        "experiments/crowdnav_v4_full_150k/dataset/reward_performance_dataset.csv (seed-mean)",
    )

    # family tables
    fam50 = (
        a50.groupby("family_id")["success_rate"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(6)
        .reset_index()
    )
    fam150 = (
        a150.groupby("family_id")["success_rate"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(6)
        .reset_index()
    )
    fam50.to_csv(OUT / "labels_50k_by_family.csv", index=False)
    fam150.to_csv(OUT / "labels_150k_by_family.csv", index=False)

    # pool integrity
    ds_fams_50 = sorted(a50["family_id"].unique())
    ds_fams_150 = sorted(a150["family_id"].unique())
    cj_fams_50 = sorted(r50["cand_summary"]["by_family"].keys())
    cj_fams_150 = sorted(r150["cand_summary"]["by_family"].keys())
    integrity = {
        "v1_dataset_families": ds_fams_50,
        "v1_candidates_json_families": cj_fams_50,
        "v1_candidates_json_matches_dataset": ds_fams_50 == cj_fams_50,
        "v4_dataset_families": ds_fams_150,
        "v4_candidates_json_families": cj_fams_150,
        "v4_candidates_json_matches_dataset": ds_fams_150 == cj_fams_150,
        "shared_candidate_ids": sorted(set(a50.candidate_id) & set(a150.candidate_id)),
        "n_shared_candidate_ids": len(set(a50.candidate_id) & set(a150.candidate_id)),
        "shared_family_names": sorted(set(ds_fams_50) & set(ds_fams_150)),
        "v1_phase1_n_jobs": r50["phase1"]["n_jobs"],
        "v1_phase1_n_failed": r50["phase1"]["n_failed"],
        "v1_phase1_duration_sec_logged": r50["phase1"]["duration_sec"],
        "v1_manifest_timesteps": r50["manifest"]["ppo_total_timesteps"],
        "v1_manifest_n_candidates": r50["manifest"]["n_candidates"],
        "v1_dataset_n_rows": len(r50["ds"]),
        "v1_sum_train_duration_sec": float(r50["ds"]["train_duration_sec"].sum()),
        "v1_sum_eval_duration_sec": float(r50["ds"]["eval_duration_sec"].sum()),
        "v4_phase1_n_jobs": r150["phase1"]["n_jobs"],
        "v4_phase1_n_failed": r150["phase1"]["n_failed"],
        "v4_phase1_duration_sec": r150["phase1"]["duration_sec"],
        "v4_manifest_timesteps": r150["manifest"]["ppo_total_timesteps"],
        "v4_sum_train_duration_sec": float(r150["ds"]["train_duration_sec"].sum()),
        "v4_sum_eval_duration_sec": float(r150["ds"]["eval_duration_sec"].sum()),
        "v1_outliers": r50["outliers"],
        "v4_outliers": r150["outliers"],
        "v1_n_eval_episodes": sorted(r50["ds"]["n_eval_episodes"].dropna().unique().tolist()),
        "v4_n_eval_episodes": sorted(r150["ds"]["n_eval_episodes"].dropna().unique().tolist()),
        "v1_env_id": sorted(r50["ds"]["env_id"].dropna().unique().tolist()),
        "v4_env_id": sorted(r150["ds"]["env_id"].dropna().unique().tolist()),
        "v1_device": sorted(r50["ds"]["device"].dropna().unique().tolist()),
        "v4_device": sorted(r150["ds"]["device"].dropna().unique().tolist()),
        "v1_early_feature_prefixes": sorted(
            {
                c.split("_")[1]
                for c in pd.read_csv(r50["exp"] / "features" / "features_early.csv", nrows=0).columns
                if c.startswith("early_t")
            }
        ),
        "v4_early_feature_prefixes": sorted(
            {
                c.split("_")[1]
                for c in pd.read_csv(r150["exp"] / "features" / "features_early.csv", nrows=0).columns
                if c.startswith("early_t")
            }
        ),
    }
    (OUT / "discovery_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )

    for k in [
        "v1_phase1_duration_sec_logged",
        "v1_sum_train_duration_sec",
        "v4_phase1_duration_sec",
        "v4_sum_train_duration_sec",
        "n_shared_candidate_ids",
    ]:
        add_number(
            metric=k,
            run="meta",
            model="",
            features="",
            value=integrity[k],
            source="docs/crowdnav_report/discovery_integrity.json",
        )

    # surrogate primary metrics
    p50 = primary_row(r50["ms"] if r50["lb"] is None else r50["lb"])
    p150 = primary_row(r150["lb"])
    metric_cols = [
        "spearman",
        "within_family_spearman_mean",
        "within_family_spearman_std",
        "within_family_top3_hit",
        "within_pairwise_accuracy",
        "mae",
        "r2",
        "top10_overlap",
        "top10_regret",
        "boot_pooled_spearman_lo",
        "boot_pooled_spearman_hi",
        "boot_within_spearman_lo",
        "boot_within_spearman_hi",
        "needs_ppo_steps",
        "n",
    ]
    for run_tag, row, src in [
        ("50k", p50, "results/crowdnav_v1_50k/leaderboard.csv"),
        ("150k", p150, "results/crowdnav_v4_full_150k/leaderboard.csv"),
    ]:
        for col in metric_cols:
            if col in row.index and pd.notna(row[col]):
                add_number(
                    metric=col,
                    run=run_tag,
                    model="extra_trees",
                    features="early+math+behavioral",
                    value=float(row[col]),
                    source=f"{src}:{col}",
                )

    # per-family fold
    # detect spearman column in fold files
    for run_tag, fold, path in [
        ("50k", r50["fold"], "experiments/crowdnav_v1_50k/model_comparison/metrics_by_fold.csv"),
        (
            "150k",
            r150["fold"],
            "experiments/crowdnav_v4_full_150k/model_comparison/metrics_by_fold.csv",
        ),
    ]:
        sub = fold[
            (fold.model == "extra_trees")
            & (fold.feature_set == "early+math+behavioral")
            & (fold.target == "success_rate")
        ]
        fam_col = "test_family" if "test_family" in sub.columns else "family_id"
        spear_col = (
            "within_family_spearman_mean"
            if "within_family_spearman_mean" in sub.columns
            else "spearman"
        )
        g = sub.groupby(fam_col).agg(
            within_mean=(spear_col, "mean"),
            within_std=(spear_col, "std"),
        )
        g.to_csv(OUT / f"within_by_family_{run_tag}.csv")
        for fam, rec in g.iterrows():
            add_number(
                metric="within_family_spearman_mean",
                run=run_tag,
                model="extra_trees",
                features=f"family:{fam}",
                value=float(rec["within_mean"]),
                source=f"{path}:{spear_col} mean over repeats",
            )

    # Cross-run on shared IDs only (NOT same rewards if families differ)
    shared = sorted(set(a50.candidate_id) & set(a150.candidate_id))
    cross = None
    if shared:
        m = a50[a50.candidate_id.isin(shared)][
            ["candidate_id", "family_id", "success_rate"]
        ].merge(
            a150[a150.candidate_id.isin(shared)][
                ["candidate_id", "family_id", "success_rate"]
            ],
            on="candidate_id",
            suffixes=("_50k", "_150k"),
        )
        cross = m
        m.to_csv(OUT / "cross_shared_candidate_ids.csv", index=False)
        rho, p = stats.spearmanr(m["success_rate_50k"], m["success_rate_150k"])
        tau, p_k = stats.kendalltau(m["success_rate_50k"], m["success_rate_150k"])
        add_number(
            metric="cross_spearman_shared_ids",
            run="50k_vs_150k",
            model="",
            features="",
            value=float(rho),
            source="labels_50k_by_candidate.csv vs labels_150k_by_candidate.csv shared IDs",
        )
        add_number(
            metric="cross_kendall_shared_ids",
            run="50k_vs_150k",
            model="",
            features="",
            value=float(tau),
            source="labels_50k_by_candidate.csv vs labels_150k_by_candidate.csv shared IDs",
        )
        add_number(
            metric="cross_spearman_pvalue",
            run="50k_vs_150k",
            model="",
            features="",
            value=float(p),
            source="scipy.stats.spearmanr",
        )
        # within-family on shared family names only
        common_fam = sorted(set(m.family_id_50k) & set(m.family_id_150k))
        # actually compare only rows where family_id matches
        same_fam = m[m.family_id_50k == m.family_id_150k]
        within_rhos = []
        for fam, g in same_fam.groupby("family_id_50k"):
            if len(g) >= 4:
                rr, _ = stats.spearmanr(g["success_rate_50k"], g["success_rate_150k"])
                within_rhos.append((fam, float(rr), len(g)))
        pd.DataFrame(within_rhos, columns=["family_id", "spearman", "n"]).to_csv(
            OUT / "cross_within_family_shared.csv", index=False
        )
        if within_rhos:
            add_number(
                metric="cross_within_spearman_mean_shared_fams",
                run="50k_vs_150k",
                model="",
                features="",
                value=float(np.mean([x[1] for x in within_rhos])),
                source="cross_within_family_shared.csv",
            )

        # top-k overlap / regret picking by 50k for 150k labels among shared
        for k in [3, 5, 10]:
            if len(m) < k:
                continue
            top50 = set(m.nlargest(k, "success_rate_50k").candidate_id)
            top150 = set(m.nlargest(k, "success_rate_150k").candidate_id)
            overlap = len(top50 & top150) / k
            oracle = float(m.nlargest(k, "success_rate_150k")["success_rate_150k"].max())
            picked = float(m[m.candidate_id.isin(top50)]["success_rate_150k"].max())
            regret = oracle - picked
            add_number(
                metric=f"cross_topk_overlap_k{k}",
                run="50k_vs_150k",
                model="",
                features="",
                value=overlap,
                source="cross_shared_candidate_ids.csv",
            )
            add_number(
                metric=f"cross_topk_regret_k{k}",
                run="50k_vs_150k",
                model="",
                features="",
                value=regret,
                source="cross_shared_candidate_ids.csv",
            )

    # selection sim v4 if present
    sel = ROOT / "results/crowdnav_v4_full_150k/selection_sim/regret_vs_budget_mean_over_repeats.csv"
    if sel.exists():
        sdf = pd.read_csv(sel)
        sdf.to_csv(OUT / "selection_sim_150k_mean.csv", index=False)
        for _, row in sdf.iterrows():
            add_number(
                metric=f"selection_sur_best_k{int(row['k'])}",
                run="150k",
                model="extra_trees",
                features="early+math+behavioral",
                value=float(row["sur"]),
                source=str(sel.relative_to(ROOT)).replace("\\", "/"),
            )

    # 2D confirm reference numbers from docs/report if available
    confirm = ROOT / "experiments/model_comparison_confirm/metrics_summary.csv"
    if confirm.exists():
        cdf = pd.read_csv(confirm)
        crow = cdf[
            (cdf.model == "extra_trees")
            & (cdf.feature_set == "early+math+behavioral")
            & (cdf.target == "success_rate")
        ]
        if len(crow):
            for col in [
                "spearman",
                "within_family_spearman_mean",
                "within_family_top3_hit",
                "mae",
            ]:
                if col in crow.columns:
                    add_number(
                        metric=col,
                        run="2d_confirm",
                        model="extra_trees",
                        features="early+math+behavioral",
                        value=float(crow.iloc[0][col]),
                        source="experiments/model_comparison_confirm/metrics_summary.csv",
                    )

    # ---------- FIGURES ----------
    # 1) per-family mean SR both runs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, famdf, title in [
        (axes[0], fam50.sort_values("mean"), "50k PPO — per-family mean success_rate"),
        (axes[1], fam150.sort_values("mean"), "150k PPO — per-family mean success_rate"),
    ]:
        colors = [FAMILY_COLORS[i % len(FAMILY_COLORS)] for i in range(len(famdf))]
        ax.barh(famdf["family_id"], famdf["mean"], xerr=famdf["std"].fillna(0), color=colors, ecolor=PALETTE["gray"])
        ax.set_xlabel("Mean success_rate (± std across candidates)")
        ax.set_title(title)
        ax.set_xlim(0, max(0.85, float(famdf["mean"].max()) + 0.1))
        ax.axvline(famdf["mean"].mean(), color=PALETTE["primary"], ls="--", lw=1, label="grand mean")
        ax.legend(loc="lower right")
    savefig("01_per_family_mean_sr.png")

    # 2) label histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].hist(a50["success_rate"], bins=12, color=PALETTE["primary"], edgecolor="white")
    axes[0].set_title(f"50k labels (n={len(a50)})\nmean={s50['mean']:.3f}")
    axes[1].hist(a150["success_rate"], bins=12, color=PALETTE["secondary"], edgecolor="white")
    axes[1].set_title(f"150k labels (n={len(a150)})\nmean={s150['mean']:.3f}")
    for ax in axes:
        ax.set_xlabel("success_rate (seed-mean)")
        ax.set_ylabel("Candidates")
    savefig("02_label_histograms.png")

    # 3) 50k vs 150k scatter if shared
    if cross is not None and len(cross):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        fams = sorted(cross["family_id_50k"].unique())
        for i, fam in enumerate(fams):
            g = cross[cross["family_id_50k"] == fam]
            ax.scatter(
                g["success_rate_50k"],
                g["success_rate_150k"],
                s=55,
                color=FAMILY_COLORS[i % len(FAMILY_COLORS)],
                label=fam,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
            )
        rho = float(stats.spearmanr(cross.success_rate_50k, cross.success_rate_150k).statistic)
        lims = [0, max(cross.success_rate_50k.max(), cross.success_rate_150k.max()) * 1.05]
        ax.plot(lims, lims, color=PALETTE["gray"], ls="--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("success_rate @ 50k (seed-mean)")
        ax.set_ylabel("success_rate @ 150k (seed-mean)")
        ax.set_title(
            f"Shared candidate_id only (n={len(cross)})\nSpearman={rho:.3f} — NOT same reward pool"
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        savefig("03_50k_vs_150k_scatter_shared_ids.png")

    # 4) pooled vs within bar
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["pooled Spearman", "within Spearman", "Top-3 hit"]
    v50 = [float(p50["spearman"]), float(p50["within_family_spearman_mean"]), float(p50["within_family_top3_hit"])]
    v150 = [float(p150["spearman"]), float(p150["within_family_spearman_mean"]), float(p150["within_family_top3_hit"])]
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w / 2, v50, w, label="50k (v1)", color=PALETTE["gray"])
    ax.bar(x + w / 2, v150, w, label="150k (v4)", color=PALETTE["primary"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ExtraTrees / early+math+behavioral / LOFO")
    ax.legend()
    for i, (a, b) in enumerate(zip(v50, v150)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
    savefig("04_pooled_vs_within_50k_150k.png")

    # 5) OOF pred vs true
    for run_tag, oof, fname in [
        ("50k", r50["oof"], "05_oof_pred_vs_true_50k.png"),
        ("150k", r150["oof"], "05_oof_pred_vs_true_150k.png"),
    ]:
        sub = oof[
            (oof.model == "extra_trees")
            & (oof.feature_set == "early+math+behavioral")
            & (oof.target == "success_rate")
            & (oof.repeat == 0)
        ]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        fams = sorted(sub.family_id.unique())
        for i, fam in enumerate(fams):
            g = sub[sub.family_id == fam]
            ax.scatter(
                g.y_true,
                g.y_pred,
                s=45,
                color=FAMILY_COLORS[i % len(FAMILY_COLORS)],
                label=fam,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
            )
        rho = float(stats.spearmanr(sub.y_true, sub.y_pred).statistic)
        lim0 = min(sub.y_true.min(), sub.y_pred.min())
        lim1 = max(sub.y_true.max(), sub.y_pred.max())
        ax.plot([lim0, lim1], [lim0, lim1], color=PALETTE["gray"], ls="--", lw=1)
        ax.set_xlabel("True success_rate")
        ax.set_ylabel("OOF predicted success_rate")
        ax.set_title(f"{run_tag}: ExtraTrees OOF (repeat 0)\nSpearman={rho:.3f}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
        savefig(fname)

    # 6) selection curve 150k
    if sel.exists():
        sdf = pd.read_csv(sel)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(sdf["k"], sdf["oracle"], color=PALETTE["ink"], lw=2, label="Oracle best")
        ax.plot(sdf["k"], sdf["sur"], color=PALETTE["primary"], lw=2.5, label="Surrogate best")
        ax.plot(sdf["k"], sdf["rand"], color=PALETTE["gray"], lw=2, ls="--", label="Random best (mean)")
        ax.set_xlabel("Budget k (candidates fully evaluated)")
        ax.set_ylabel("Best success_rate found")
        ax.set_title("150k selection simulation (mean over LOFO repeats)")
        ax.legend()
        savefig("06_selection_sim_150k.png")

    # 7) cost vs accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    # wall hours: v4 logged; v1 use sum_train/workers estimate noted in text — plot CPU-hours and within spearman
    cpu50 = integrity["v1_sum_train_duration_sec"] / 3600
    cpu150 = integrity["v4_sum_train_duration_sec"] / 3600
    ax.scatter([cpu50], [float(p50["within_family_spearman_mean"])], s=160, color=PALETTE["gray"], label="50k")
    ax.scatter([cpu150], [float(p150["within_family_spearman_mean"])], s=160, color=PALETTE["primary"], label="150k")
    ax.annotate("50k", (cpu50, float(p50["within_family_spearman_mean"])), textcoords="offset points", xytext=(8, 8))
    ax.annotate("150k", (cpu150, float(p150["within_family_spearman_mean"])), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Sum of train_duration_sec across dataset rows (CPU-hours)")
    ax.set_ylabel("Within-family Spearman (ExtraTrees)")
    ax.set_title("Cost vs within-family ranking quality")
    ax.set_ylim(0, 1)
    ax.legend()
    savefig("07_cost_vs_within_spearman.png")

    # 8) seed noise comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(a50["success_std"].dropna(), bins=12, alpha=0.65, color=PALETTE["gray"], label="50k seed-std")
    ax.hist(a150["success_std"].dropna(), bins=12, alpha=0.65, color=PALETTE["primary"], label="150k seed-std")
    ax.set_xlabel("Std of success_rate across 3 seeds")
    ax.set_ylabel("Candidates")
    ax.set_title("Seed noise comparison")
    ax.legend()
    savefig("08_seed_noise_hist.png")

    pd.DataFrame(NUMBERS).to_csv(OUT / "numbers.csv", index=False)

    # dump key summary json for markdown writer
    summary = {
        "label_50k": s50,
        "label_150k": s150,
        "primary_50k": {c: (float(p50[c]) if c in p50.index and pd.notna(p50[c]) else None) for c in metric_cols},
        "primary_150k": {
            c: (float(p150[c]) if c in p150.index and pd.notna(p150[c]) else None) for c in metric_cols
        },
        "integrity": integrity,
        "family_50k": fam50.to_dict(orient="records"),
        "family_150k": fam150.to_dict(orient="records"),
        "cross_n_shared": len(shared),
    }
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("Wrote figures + numbers + summary to", OUT)


if __name__ == "__main__":
    main()
