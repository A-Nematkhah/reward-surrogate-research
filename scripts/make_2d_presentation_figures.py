"""Generate presentation figures for the 2D experiments report into docs/figures_2d/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures_2d"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def fig_env_sketch() -> None:
    from src.environment.navigation_2d import RewardNavigation2D
    from src.environment.visualization import render_env

    env = RewardNavigation2D()
    env.reset(seed=7)
    fig = render_env(env, save_path=None, show=False)
    ax = fig.axes[0]
    ax.set_title("RewardNavigation2D-v0: robot, goal, circular obstacles")
    _save(fig, "01_env_sketch.png")


def fig_family_success() -> None:
    ds = pd.read_csv(ROOT / "experiments/phase1/dataset/reward_performance_dataset.csv")
    # one row per candidate: mean over seeds
    by_cand = (
        ds.groupby(["family_id", "candidate_id"], as_index=False)["success_rate"]
        .mean()
    )
    fam = (
        by_cand.groupby("family_id")["success_rate"]
        .agg(mean="mean", count="size")
        .sort_values("mean", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c45c26" if v < 0.2 else "#3b6ea5" if v < 0.5 else "#2f6f4e" for v in fam["mean"]]
    ax.barh(fam.index.astype(str), fam["mean"].values, color=colors)
    ax.set_xlabel("Mean success rate")
    ax.set_title("Final dataset: success rate by reward family (n=108 candidates)")
    for y, (mean, n) in enumerate(zip(fam["mean"], fam["count"])):
        ax.text(mean + 0.01, y, f"{mean:.2f} (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, 0.85)
    _save(fig, "02_family_success_rate.png")


def fig_global_vs_within() -> None:
    # numbers from report sections 3 and 5
    rows = [
        ("Pilot ExtraTrees\nbehavioral", 0.142, 0.087),
        ("Pilot GP\nbehavioral", 0.307, 0.392),
        ("Scale ExtraTrees\nearly only", 0.651, 0.239),
        ("Scale ExtraTrees\nbehavioral_early", 0.835, 0.611),
        ("Confirm ExtraTrees\nearly+math+beh.", 0.935, 0.648),
    ]
    labels = [r[0] for r in rows]
    global_s = [r[1] for r in rows]
    within_s = [r[2] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, global_s, w, label="Spearman (pooled / global)", color="#3b6ea5")
    ax.bar(x + w / 2, within_s, w, label="Spearman within-family (mean)", color="#c45c26")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="within-family threshold ~0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Global vs within-family ranking (do not confuse the two)")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "03_global_vs_within_family.png")


def fig_feature_progress() -> None:
    # ExtraTrees path from report §4–5
    steps = [
        ("static", 0.446, 0.119),
        ("math", 0.579, 0.196),
        ("early", 0.651, 0.239),
        ("behavioral", 0.822, 0.560),
        ("behavioral_early", 0.835, 0.611),
        ("early+math+beh.\n(confirm)", 0.935, 0.648),
    ]
    labels = [s[0] for s in steps]
    within = [s[2] for s in steps]
    pooled = [s[1] for s in steps]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, pooled, "o--", color="#94a3b8", label="Spearman global", linewidth=1.8)
    ax.plot(x, within, "o-", color="#c45c26", label="Spearman within-family", linewidth=2.2)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman")
    ax.set_ylim(0, 1.05)
    ax.set_title("ExtraTrees: feature progress (phase2 → confirm)")
    ax.legend(frameon=False)
    for i, v in enumerate(within):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    _save(fig, "04_feature_progress_extratrees.png")


def fig_confirm_comparison() -> None:
    m = pd.read_csv(ROOT / "experiments/model_comparison_confirm/metrics_summary.csv")
    m = m[m["target"] == "success_rate"]
    agg = (
        m.groupby(["model", "feature_set"], as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            within=("within_family_spearman_mean", "mean"),
            top3=("within_family_top3_hit", "mean"),
            mae=("mae", "mean"),
            pairwise=("within_pairwise_accuracy", "mean"),
        )
    )
    order = [
        ("extra_trees", "early+math+behavioral"),
        ("extra_trees", "early+math"),
        ("catboost", "early+math+behavioral"),
        ("catboost", "early+math"),
    ]
    rows = []
    for model, fs in order:
        r = agg[(agg.model == model) & (agg.feature_set == fs)].iloc[0]
        rows.append(r)
    labels = [f"{r.model}\n{r.feature_set}" for r in rows]
    x = np.arange(len(rows))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - w, [r.within for r in rows], w, label="Within-family Spearman", color="#c45c26")
    ax.bar(x, [r.top3 for r in rows], w, label="Top-3 hit (within family)", color="#2f6f4e")
    ax.bar(x + w, [r.spearman for r in rows], w, label="Spearman global", color="#3b6ea5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Confirm (3 repeats, mean): ExtraTrees vs CatBoost")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "05_confirm_extratrees_vs_catboost.png")


def fig_dataset_size_curve() -> None:
    c = pd.read_csv(ROOT / "results/dataset_size_curve.csv")
    # report labels 108; CSV has 99 as full train when holding out families — rename for display
    n = c["train_n"].tolist()
    labels = [("108*" if v == 99 else str(v)) for v in n]
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.errorbar(
        range(len(c)),
        c["within_spearman_mean"],
        yerr=c["within_spearman_std"],
        fmt="o-",
        color="#c45c26",
        label="Within-family Spearman",
        capsize=3,
    )
    ax1.errorbar(
        range(len(c)),
        c["top3_mean"],
        yerr=c["top3_std"],
        fmt="s--",
        color="#2f6f4e",
        label="Top-3 hit",
        capsize=3,
    )
    ax1.set_xticks(range(len(c)))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Training rewards (held-out family always full)")
    ax1.set_ylabel("Ranking score")
    ax1.set_ylim(0.2, 1.0)
    ax2 = ax1.twinx()
    ax2.plot(range(len(c)), c["mae_mean"], "^-.", color="#3b6ea5", label="MAE")
    ax2.set_ylabel("MAE")
    ax2.set_ylim(0.05, 0.20)
    ax1.set_title("Dataset size curve (ExtraTrees, final features)")
    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)
    ax1.annotate(
        "*CSV train_n=99 ≈ full 108 with LOFO train size",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color="gray",
    )
    _save(fig, "06_dataset_size_curve.png")


def fig_pred_vs_actual() -> None:
    pred = pd.read_csv(ROOT / "experiments/model_comparison_confirm/oof_predictions.csv")
    sub = pred[
        (pred.model == "extra_trees")
        & (pred.feature_set == "early+math+behavioral")
        & (pred.target == "success_rate")
        & (pred.repeat == 0)
    ].copy()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    families = sorted(sub.family_id.unique())
    cmap = plt.get_cmap("tab20")
    for i, fam in enumerate(families):
        s = sub[sub.family_id == fam]
        ax.scatter(s.y_true, s.y_pred, s=36, alpha=0.85, color=cmap(i % 20), label=fam)
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual success rate")
    ax.set_ylabel("Predicted success rate")
    ax.set_title("ExtraTrees early+math+behavioral (OOF, repeat 0)")
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
    _save(fig, "07_pred_vs_actual.png")


def fig_timeline() -> None:
    stages = [
        "Pilot\nGP beh.",
        "Scale\nstatic",
        "Scale\nearly",
        "Scale\nbeh_early",
        "Confirm\nET full",
    ]
    within = [0.39, 0.28, 0.24, 0.61, 0.65]
    mae = [0.18, 0.22, 0.16, 0.13, 0.067]
    x = np.arange(len(stages))
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(x, within, "o-", color="#c45c26", linewidth=2.2, label="Within-family Spearman")
    ax1.set_ylabel("Within-family Spearman")
    ax1.set_ylim(0, 0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax2 = ax1.twinx()
    ax2.plot(x, mae, "s--", color="#3b6ea5", label="MAE")
    ax2.set_ylabel("MAE")
    ax2.set_ylim(0, 0.28)
    ax1.set_title("Numeric path across stages (report §7)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)
    _save(fig, "08_stage_timeline.png")


def fig_heatmap_phase2() -> None:
    m = pd.read_csv(ROOT / "experiments/phase2/surrogate_metrics.csv")
    oof = m[(m.split == "lofo_oof") & (m.target == "success_rate")].copy()
    # prefer fold_spearman_mean (within-family mean under LOFO)
    pivot = oof.pivot_table(
        index="model",
        columns="feature_set",
        values="fold_spearman_mean",
        aggfunc="first",
    )
    # drop baselines / rank model clutter a bit but keep main models
    keep_models = [
        "extra_trees",
        "random_forest",
        "ensemble_trees",
        "hist_gradient_boosting",
        "gaussian_process",
        "ridge",
        "rank_extra_trees",
    ]
    keep_fs = ["static", "math", "early", "behavioral", "behavioral_early", "hybrid"]
    pivot = pivot.reindex(index=[x for x in keep_models if x in pivot.index], columns=keep_fs)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.2, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Phase2 LOFO: within-family Spearman (fold mean)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, "09_phase2_within_family_heatmap.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_env_sketch()
    fig_family_success()
    fig_global_vs_within()
    fig_feature_progress()
    fig_confirm_comparison()
    fig_dataset_size_curve()
    fig_pred_vs_actual()
    fig_timeline()
    fig_heatmap_phase2()
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
