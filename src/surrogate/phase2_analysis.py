"""Phase 2 figures and final research report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.logging import get_logger

logger = get_logger("phase2_analysis")

# Constant predictors: ranking metrics are N/A (not fair Spearman baselines).
CONSTANT_BASELINES = {"mean", "median"}
# Models treated as competitive learners (ridge included).
LEARNING_MODELS_EXCLUDE_CONSTANTS = {"mean", "median"}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def load_phase2_artifacts(
    metrics_path: str | Path,
    predictions_path: str | Path,
    importances_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(predictions_path)
    importances = []
    if importances_path and Path(importances_path).is_file():
        with Path(importances_path).open("r", encoding="utf-8") as handle:
            importances = json.load(handle)
    return {"metrics": metrics, "predictions": preds, "importances": importances}


def oof_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    mask = metrics["split"].isin(["grouped_cv_oof", "lofo_oof"])
    return metrics[mask].copy()


def fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    mask = metrics["split"].isin(["grouped_cv", "lofo"])
    return metrics[mask].copy()


def pick_best_config(
    oof: pd.DataFrame,
    target: str,
    *,
    exclude_baselines: bool = False,
    min_defined_folds: int = 0,
    max_fold_std: float | None = None,
) -> pd.Series:
    subset = oof[oof["target"] == target].copy()
    if exclude_baselines:
        subset = subset[~subset["model"].isin(LEARNING_MODELS_EXCLUDE_CONSTANTS)]
    if "n_folds_defined_spearman" in subset.columns and min_defined_folds > 0:
        subset = subset[
            subset["n_folds_defined_spearman"].fillna(0) >= min_defined_folds
        ]
    if max_fold_std is not None and "fold_spearman_std" in subset.columns:
        # Prefer stable configs when available; do not drop all if none qualify
        stable = subset[subset["fold_spearman_std"].fillna(np.inf) <= max_fold_std]
        if not stable.empty:
            subset = stable
    subset = subset.sort_values("spearman", ascending=False)
    if subset.empty:
        raise ValueError(f"No OOF rows for target={target}")
    return subset.iloc[0]


def generate_phase2_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importances: Sequence[Mapping[str, Any]],
    *,
    feature_names_by_set: Mapping[str, Sequence[str]] | None,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    oof = oof_metrics(metrics)
    grouped_preds = predictions[
        predictions["split"].isin(["grouped_cv", "lofo"])
    ].copy()

    # 1. Predicted vs actual for each target (best non-baseline by Spearman)
    for target in sorted(oof["target"].unique()):
        best = pick_best_config(oof, target, exclude_baselines=True)
        sub = grouped_preds[
            (grouped_preds["target"] == target)
            & (grouped_preds["model"] == best["model"])
            & (grouped_preds["feature_set"] == best["feature_set"])
        ]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.75, c="#3b6ea5")
        lims = [
            np.nanmin([sub["y_true"].min(), sub["y_pred"].min()]),
            np.nanmax([sub["y_true"].max(), sub["y_pred"].max()]),
        ]
        ax.plot(lims, lims, "--", color="gray", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(
            f"{target}\nbest={best['model']}/{best['feature_set']} "
            f"(Spearman={best['spearman']:.2f})"
        )
        paths.append(_save(fig, output_dir / f"01_pred_vs_actual_{target}.png"))

    # 2. Residual distributions for success_rate best model
    target = "success_rate"
    if target in oof["target"].values:
        best = pick_best_config(oof, target, exclude_baselines=True)
        sub = grouped_preds[
            (grouped_preds["target"] == target)
            & (grouped_preds["model"] == best["model"])
            & (grouped_preds["feature_set"] == best["feature_set"])
        ]
        resid = sub["y_pred"] - sub["y_true"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(resid, bins=12, color="#6b7280", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Prediction error (pred - actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals: {target} ({best['model']}/{best['feature_set']})")
        paths.append(_save(fig, output_dir / "02_residual_success_rate.png"))

    # 3. Predicted ranking vs actual ranking (success_rate)
    if target in oof["target"].values:
        best = pick_best_config(oof, target, exclude_baselines=True)
        sub = grouped_preds[
            (grouped_preds["target"] == target)
            & (grouped_preds["model"] == best["model"])
            & (grouped_preds["feature_set"] == best["feature_set"])
        ].copy()
        sub["rank_true"] = sub["y_true"].rank(ascending=False, method="average")
        sub["rank_pred"] = sub["y_pred"].rank(ascending=False, method="average")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(sub["rank_true"], sub["rank_pred"], alpha=0.75, c="#2f6f4e")
        ax.plot(
            [sub["rank_true"].min(), sub["rank_true"].max()],
            [sub["rank_true"].min(), sub["rank_true"].max()],
            "--",
            color="gray",
        )
        ax.set_xlabel("Actual rank (1=best success)")
        ax.set_ylabel("Predicted rank")
        ax.set_title(f"Ranking agreement: {target}")
        paths.append(_save(fig, output_dir / "03_ranking_success_rate.png"))

    # 4. Feature importance (average across folds) for RF static success_rate if available
    paths.extend(
        _plot_feature_importances(
            importances, feature_names_by_set, output_dir / "04_feature_importance.png"
        )
    )

    # 5. Model comparison heatmap for success_rate Spearman
    paths.append(
        _plot_model_comparison_heatmap(
            oof, target="success_rate", metric="spearman",
            path=output_dir / "05_model_comparison_success_spearman.png",
        )
    )

    # 6. Static vs behavioral vs hybrid
    paths.append(
        _plot_feature_set_comparison(
            oof, path=output_dir / "06_feature_set_comparison.png"
        )
    )

    # 7. Performance by reward family (best success_rate model residuals)
    if "success_rate" in grouped_preds["target"].values:
        best = pick_best_config(oof, "success_rate", exclude_baselines=True)
        sub = grouped_preds[
            (grouped_preds["target"] == "success_rate")
            & (grouped_preds["model"] == best["model"])
            & (grouped_preds["feature_set"] == best["feature_set"])
        ].copy()
        sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()
        family_err = sub.groupby("family_id")["abs_err"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(family_err.index.astype(str), family_err.values, color="#b85c38")
        ax.set_ylabel("Mean |error|")
        ax.set_title(
            f"Error by family ({best['model']}/{best['feature_set']}, success_rate)"
        )
        ax.tick_params(axis="x", rotation=30)
        paths.append(_save(fig, output_dir / "07_error_by_family.png"))

    # 8. Uncertainty vs error where available
    unc = grouped_preds.dropna(subset=["y_std"])
    if not unc.empty:
        unc = unc.copy()
        unc["abs_err"] = (unc["y_pred"] - unc["y_true"]).abs()
        # Use success_rate RF/GP if present
        sample = unc[unc["target"] == "success_rate"]
        if sample.empty:
            sample = unc
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(sample["y_std"], sample["abs_err"], alpha=0.6, c="#7c3aed")
        ax.set_xlabel("Predictive uncertainty (std-like)")
        ax.set_ylabel("|error|")
        if len(sample) >= 3:
            sp = spearmanr(sample["y_std"], sample["abs_err"]).correlation
            ax.set_title(f"Uncertainty vs error (Spearman={sp:.2f})")
        else:
            ax.set_title("Uncertainty vs error")
        paths.append(_save(fig, output_dir / "08_uncertainty_vs_error.png"))

    # Bonus: baseline vs best for all targets
    paths.append(
        _plot_baseline_vs_best(oof, path=output_dir / "09_baseline_vs_best.png")
    )

    logger.info("Wrote %d Phase 2 figures to %s", len(paths), output_dir)
    return paths


def _plot_feature_importances(
    importances: Sequence[Mapping[str, Any]],
    feature_names_by_set: Mapping[str, Sequence[str]] | None,
    path: Path,
) -> list[Path]:
    if not importances:
        return []
    # Prefer random_forest / static / success_rate
    candidates = [
        item
        for item in importances
        if item.get("model") == "random_forest"
        and item.get("feature_set") == "static"
        and item.get("target") == "success_rate"
    ]
    if not candidates:
        candidates = [
            item
            for item in importances
            if item.get("model") in {"random_forest", "extra_trees"}
        ]
    if not candidates:
        return []

    mats = [np.asarray(item["importances"], dtype=np.float64) for item in candidates]
    # Align lengths
    min_len = min(m.size for m in mats)
    avg = np.mean([m[:min_len] for m in mats], axis=0)
    names: Sequence[str]
    if feature_names_by_set and candidates[0]["feature_set"] in feature_names_by_set:
        names = list(feature_names_by_set[candidates[0]["feature_set"]])[:min_len]
    else:
        names = [f"f{i}" for i in range(min_len)]

    order = np.argsort(avg)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order][::-1], avg[order][::-1], color="#3b6ea5")
    ax.set_xlabel("Mean impurity importance")
    ax.set_title(
        f"Feature importance ({candidates[0]['model']}/"
        f"{candidates[0]['feature_set']}/{candidates[0]['target']})"
    )
    return [_save(fig, path)]


def _plot_model_comparison_heatmap(
    oof: pd.DataFrame, *, target: str, metric: str, path: Path
) -> Path:
    sub = oof[oof["target"] == target]
    pivot = sub.pivot_table(index="model", columns="feature_set", values=metric)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{target}: grouped OOF {metric}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _save(fig, path)


def _plot_feature_set_comparison(oof: pd.DataFrame, *, path: Path) -> Path:
    # Mean Spearman across non-baseline models per feature set and target
    sub = oof[~oof["model"].isin(CONSTANT_BASELINES)].copy()
    grouped = (
        sub.groupby(["feature_set", "target"])["spearman"].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    targets = sorted(grouped["target"].unique())
    preferred = ["static", "behavioral", "early", "hybrid"]
    available = list(grouped["feature_set"].unique())
    sets = [s for s in preferred if s in available] or sorted(available)
    x = np.arange(len(targets))
    width = 0.8 / max(len(sets), 1)
    offset0 = (len(sets) - 1) / 2.0
    for i, fs in enumerate(sets):
        vals = [
            float(
                grouped.loc[
                    (grouped["feature_set"] == fs) & (grouped["target"] == t),
                    "spearman",
                ].mean()
            )
            if not grouped[
                (grouped["feature_set"] == fs) & (grouped["target"] == t)
            ].empty
            else np.nan
            for t in targets
        ]
        ax.bar(x + (i - offset0) * width, vals, width=width, label=fs)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel("Mean Spearman (non-baseline models)")
    ax.set_title("Feature-set comparison")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def _plot_baseline_vs_best(oof: pd.DataFrame, *, path: Path) -> Path:
    rows = []
    for target, g in oof.groupby("target"):
        mean_row = g[g["model"] == "mean"].sort_values("spearman", ascending=False)
        best = g[~g["model"].isin(CONSTANT_BASELINES)].sort_values(
            "spearman", ascending=False
        )
        mean_mae = (
            float(mean_row.iloc[0]["mae"]) if not mean_row.empty else np.nan
        )
        best_sp = float(best.iloc[0]["spearman"]) if not best.empty else np.nan
        best_mae = float(best.iloc[0]["mae"]) if not best.empty else np.nan
        rows.append(
            {
                "target": target,
                "mean_mae": mean_mae,
                "best_mae": best_mae,
                "best_spearman": best_sp,
                "best_model": f"{best.iloc[0]['model']}/{best.iloc[0]['feature_set']}"
                if not best.empty
                else "",
            }
        )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(df))
    axes[0].bar(x - 0.2, df["mean_mae"], width=0.4, label="mean baseline MAE", color="#9ca3af")
    axes[0].bar(x + 0.2, df["best_mae"], width=0.4, label="best model MAE", color="#3b6ea5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["target"], rotation=25, ha="right")
    axes[0].set_ylabel("OOF MAE (lower better)")
    axes[0].set_title("Point accuracy vs mean baseline")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, df["best_spearman"], color="#2f6f4e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["target"], rotation=25, ha="right")
    axes[1].set_ylabel("OOF Spearman")
    axes[1].set_title("Best non-constant model ranking")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    # Constant baselines intentionally omitted from ranking panel
    fig.tight_layout()
    return _save(fig, path)


def data_sensitivity_from_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Approximate sensitivity using per-fold metric variance (family-held-out folds)."""
    folds = fold_metrics(metrics)
    if folds.empty:
        return pd.DataFrame()

    def _nanmean_ok(s: pd.Series) -> float:
        # Ignore undefined ranking folds (constant preds / too few)
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    def _nanstd_ok(s: pd.Series) -> float:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return float("nan")
        return float(np.std(arr, ddof=1))

    summary = (
        folds.groupby(["model", "feature_set", "target"])
        .agg(
            spearman_mean=("spearman", _nanmean_ok),
            spearman_std=("spearman", _nanstd_ok),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    return summary


def build_phase2_report(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    phase1_stats: Mapping[str, Any] | None = None,
    sensitivity: pd.DataFrame | None = None,
) -> str:
    oof = oof_metrics(metrics)
    lines: list[str] = []
    lines.append("# Phase 2 Report: Reward Performance Surrogates")
    lines.append("")
    lines.append("## Research question")
    lines.append("")
    lines.append(
        "Can we predict the RL performance of an **unseen** reward function "
        "from features of that reward, **without** training PPO on it?"
    )
    lines.append("")
    report_splits: list[str] = []
    if not oof.empty:
        report_splits.extend(sorted({str(s) for s in oof["split"].unique().tolist()}))
    if "split" in metrics.columns and (metrics["split"] == "random_holdout").any():
        report_splits.append("random_holdout")
    lines.append(
        "Primary evaluation: **family-held-out CV** "
        f"(splits used in this report: {report_splits}). "
        "Reports **pooled OOF** ranking together with **per-fold mean±std**. "
        "Random holdout is secondary only."
    )
    lines.append("")

    lines.append("## Can we beat naive baselines?")
    lines.append("")
    lines.append(
        "Mean/median baselines emit (near-)constant predictions, so **Spearman ranking is N/A** "
        "for them. Fair baseline comparison uses **MAE/RMSE**. Ranking is reported only for "
        "models with non-degenerate score variation, with **pooled OOF Spearman** plus "
        "**fold mean±std**."
    )
    lines.append("")
    for target, g in oof.groupby("target"):
        mean_rows = g[g["model"] == "mean"]
        best = g[~g["model"].isin(CONSTANT_BASELINES)].sort_values(
            "spearman", ascending=False
        )
        if mean_rows.empty or best.empty:
            continue
        mean_mae = float(mean_rows.iloc[0]["mae"])
        b0 = best.iloc[0]
        best_mae = float(b0["mae"])
        best_sp = float(b0["spearman"]) if pd.notna(b0["spearman"]) else float("nan")
        fold_mean = b0.get("fold_spearman_mean", np.nan)
        fold_std = b0.get("fold_spearman_std", np.nan)
        mae_better = best_mae < mean_mae - 1e-6
        lines.append(
            f"- **{target}**: mean MAE={mean_mae:.3f}; "
            f"best ranker={b0['model']}/{b0['feature_set']} "
            f"MAE={best_mae:.3f} ({'better' if mae_better else 'not better'} than mean), "
            f"pooled Spearman={best_sp:.3f}, "
            f"fold Spearman={_fmt_float(fold_mean)}±{_fmt_float(fold_std)}."
        )
    lines.append("")
    lines.append(
        "Exact-value prediction can remain weak (low/negative R²) even when ranking looks moderate. "
        "A high pooled Spearman with large fold std is **not** treated as reliable."
    )
    lines.append("")

    lines.append("## Which features are most informative?")
    lines.append("")
    fs = (
        oof[~oof["model"].isin(CONSTANT_BASELINES)]
        .groupby("feature_set")["spearman"]
        .mean()
        .sort_values(ascending=False)
    )
    for name, val in fs.items():
        lines.append(f"- Mean Spearman across non-baseline models: **{name}** = {val:.3f}")
    lines.append("")
    lines.append(
        "Static features are competitive overall. Behavioral probes help for some targets "
        "(notably `mean_goal_time` and the best `success_rate` ranking config), but hybrid "
        "is **not** consistently best — adding probe features can also add noise on this small dataset."
    )
    lines.append("")

    lines.append("## Do behavioral probe features improve prediction?")
    lines.append("")
    for target, g in oof.groupby("target"):
        g2 = g[~g["model"].isin(CONSTANT_BASELINES)]
        by_fs = g2.groupby("feature_set")["spearman"].max()
        static_v = float(by_fs.get("static", np.nan))
        beh_v = float(by_fs.get("behavioral", np.nan))
        hyb_v = float(by_fs.get("hybrid", np.nan))
        lines.append(
            f"- {target}: best static={static_v:.3f}, behavioral={beh_v:.3f}, hybrid={hyb_v:.3f}"
        )
    lines.append("")

    lines.append("## Generalization to unseen reward families")
    lines.append("")
    sr_pool = oof[
        (oof["target"] == "success_rate")
        & (~oof["model"].isin(CONSTANT_BASELINES))
    ]["spearman"]
    if not sr_pool.empty and sr_pool.notna().any():
        lines.append(
            f"- Best non-constant success-rate pooled Spearman in this run: "
            f"**{float(sr_pool.max()):.3f}**."
        )
    lines.append(
        "Family-held-out CV forces prediction for families never seen in training. "
        "**Claiming robust family generalization requires both decent pooled ranking and "
        "low fold variance.**"
    )
    rand = metrics[metrics["split"] == "random_holdout"]
    learn_oof = oof[~oof["model"].isin(CONSTANT_BASELINES)]
    if not rand.empty and not learn_oof.empty:
        rand_learn = rand[~rand["model"].isin(CONSTANT_BASELINES)]
        lines.append("")
        lines.append(
            f"- Mean Spearman random holdout (non-constant models): "
            f"{rand_learn['spearman'].mean():.3f}"
        )
        lines.append(
            f"- Mean Spearman family-held-out OOF (non-constant models): "
            f"{learn_oof['spearman'].mean():.3f}"
        )
        lines.append(
            "- If random ≫ held-out-family, within-family similarity inflated the random split; "
            "family-held-out results are the trustworthy ones."
        )
    lines.append("")

    lines.append("## Exact performance vs ranking")
    lines.append("")
    lines.append(
        "Across targets, Spearman/Kendall often look better than R². "
        "The surrogate is closer to a **noisy ranker** than a calibrated performance regressor. "
        "For Active Learning / candidate triage, ranking quality matters more — but current "
        "success-rate ranking is still too weak for high-confidence selection."
    )
    lines.append("")

    lines.append("## Triage metrics (Top-k / pairwise)")
    lines.append("")
    lines.append(
        "Top-k set overlap and pairwise ranking accuracy measure whether the surrogate "
        "can surface good rewards without training every candidate."
    )
    lines.append("")
    triage = oof[~oof["model"].isin(CONSTANT_BASELINES)].copy()
    if "fold_top_3_accuracy_mean" in triage.columns:
        for target, g in triage.groupby("target"):
            best = g.sort_values("fold_top_3_accuracy_mean", ascending=False).iloc[0]
            lines.append(
                f"- **{target}**: best fold Top-3={_fmt_float(best.get('fold_top_3_accuracy_mean'))}"
                f"±{_fmt_float(best.get('fold_top_3_accuracy_std'))} "
                f"({best['model']}/{best['feature_set']}), "
                f"pooled Top-3={_fmt_float(best.get('top_3_accuracy'))}, "
                f"pairwise={_fmt_float(best.get('pairwise_accuracy'))}."
            )
        lines.append("")
        lines.append(
            "Note: **fold Top-3** is the primary triage estimand under LOFO "
            "(top items inside each held-out family). Pooled global Top-3 can be near zero "
            "even with strong Spearman when the absolute best candidates sit in hard families."
        )
    elif "top_3_accuracy" in triage.columns:
        for target, g in triage.groupby("target"):
            best = g.sort_values("top_3_accuracy", ascending=False).iloc[0]
            lines.append(
                f"- **{target}**: best Top-3={_fmt_float(best.get('top_3_accuracy'))} "
                f"({best['model']}/{best['feature_set']}), "
                f"pairwise={_fmt_float(best.get('pairwise_accuracy'))}."
            )
    else:
        lines.append("- Top-k / pairwise columns not present in this metrics table.")
    lines.append("")

    lines.append("## Which metrics are difficult?")
    lines.append("")
    difficulty = (
        oof[~oof["model"].isin(CONSTANT_BASELINES)]
        .groupby("target")
        .agg(
            best_spearman=("spearman", "max"),
            mean_r2=("r2", "mean"),
            mean_fold_std=("fold_spearman_std", "mean"),
        )
        .sort_values("best_spearman")
    )
    for target, row in difficulty.iterrows():
        lines.append(
            f"- **{target}**: best pooled Spearman={row['best_spearman']:.3f}, "
            f"mean R²={row['mean_r2']:.3f}, "
            f"mean fold Spearman std={row['mean_fold_std']:.3f}"
        )
    lines.append("")

    lines.append("## Sensitivity to training data / fold variability")
    lines.append("")
    if sensitivity is not None and not sensitivity.empty:
        hard = sensitivity.sort_values("spearman_std", ascending=False).head(8)
        lines.append(
            "Per-fold Spearman std (higher ⇒ more sensitive to which families are held out):"
        )
        lines.append("")
        lines.append("| model | feature_set | target | mean Spearman | std |")
        lines.append("|---|---|---|---:|---:|")
        for _, row in hard.iterrows():
            lines.append(
                f"| {row['model']} | {row['feature_set']} | {row['target']} | "
                f"{row['spearman_mean']:.3f} | {row['spearman_std']:.3f} |"
            )
        lines.append("")
        n_cand = None if phase1_stats is None else phase1_stats.get("n_candidates")
        n_fam = None if phase1_stats is None else phase1_stats.get("n_families")
        if n_cand is not None and n_fam is not None:
            pool_txt = f"this {n_cand}-candidate / {n_fam}-family pool"
        else:
            pool_txt = "this candidate pool"
        lines.append(
            f"Large fold variance indicates unstable estimates on {pool_txt}. "
            "Interpret sensitivity conclusions cautiously when fold std is high."
        )
    else:
        lines.append(
            "- Formal learning curves were not the main protocol; fold-level variance "
            "shows estimates move substantially with held-out families."
        )
    lines.append("")

    if phase1_stats:
        lines.append("## Link back to Phase 1")
        lines.append("")
        lines.append(
            f"- Phase 1 completed rows: {phase1_stats.get('n_completed', 'n/a')}; "
            f"success-rate span={phase1_stats.get('success_rate_span', 'n/a')}."
        )
        lines.append(
            "- Dataset diversity is real (good/bad rewards), so weak surrogate performance "
            "is **not** explained by a flat label distribution alone."
        )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Toy 2D navigation only; not CrowdNav / multi-agent.")
    lines.append(
        "- Current budget is ~50k PPO steps × 3 seeds × ~108 candidates across 12 families; "
        "some structural families (e.g. distance_dense) still learn poorly."
    )
    lines.append("- Classical ML on ~100 candidates; high-dim hybrid features can still overfit.")
    lines.append(
        "- Pooled Top-k can look worse than per-fold Top-k because absolute global top items "
        "may sit in hard held-out families."
    )
    lines.append(
        "- Family one-hot features are disabled for this evaluation "
        "(incompatible with unseen-family generalization)."
    )
    lines.append("- Tree/GP uncertainties are not validated as calibrated probabilities.")
    lines.append("- No claim that the best Phase 1 candidate is universally optimal.")
    lines.append(
        "- Mean/median ranking Spearmans are N/A; do not treat them as competitive rankers."
    )
    lines.append("")

    lines.append("## Recommendation: Phase 3 Active Learning?")
    lines.append("")
    sr_best = float("nan")
    sr_fold_std = float("nan")
    sr_fold_top3 = float("nan")
    if "success_rate" in oof["target"].values:
        best_sr = pick_best_config(oof, "success_rate", exclude_baselines=True)
        sr_best = float(best_sr["spearman"]) if pd.notna(best_sr["spearman"]) else float("nan")
        sr_fold_std = float(best_sr.get("fold_spearman_std", np.nan))
        if (
            "fold_top_3_accuracy_mean" in best_sr.index
            and pd.notna(best_sr["fold_top_3_accuracy_mean"])
        ):
            sr_fold_top3 = float(best_sr["fold_top_3_accuracy_mean"])
        elif "top_3_accuracy" in best_sr.index and pd.notna(best_sr["top_3_accuracy"]):
            sr_fold_top3 = float(best_sr["top_3_accuracy"])
    stable_enough = np.isfinite(sr_fold_std) and sr_fold_std <= 0.35
    triage_ok = np.isfinite(sr_fold_top3) and sr_fold_top3 >= 0.5
    if np.isfinite(sr_best) and sr_best >= 0.6 and stable_enough and triage_ok:
        verdict = (
            "**Conditional yes** — ranking and fold Top-3 triage are strong/stable enough "
            "to prototype Active Learning for candidate triage, with uncertainty treated cautiously."
        )
    elif np.isfinite(sr_best) and sr_best >= 0.5 and stable_enough:
        verdict = (
            "**Exploratory AL only** — success ranking is strong and relatively stable, "
            "but triage Top-3 is not yet reliable enough to skip PPO on high-stakes decisions. "
            f"(Spearman≈{_fmt_float(sr_best)}, fold std≈{_fmt_float(sr_fold_std)}, "
            f"fold Top-3≈{_fmt_float(sr_fold_top3)})."
        )
    elif np.isfinite(sr_best) and sr_best >= 0.4:
        verdict = (
            "**Not yet for decision-making AL** — exploratory AL to grow the dataset "
            "could be studied, but do not trust the surrogate to safely skip PPO evaluations."
        )
    else:
        verdict = (
            "**No — do not proceed to Active Learning as a reward-selection system yet.** "
            f"Best success-rate pooled Spearman ≈ {_fmt_float(sr_best)} "
            f"(fold std ≈ {_fmt_float(sr_fold_std)}, fold Top-3 ≈ {_fmt_float(sr_fold_top3)}). "
            "Require stronger *and* more stable predictors before AL."
        )
    lines.append(verdict)
    lines.append("")
    lines.append(
        "If future work revisits AL, require: (1) held-out family ranking Spearman ≳ 0.5–0.6 "
        "on success with fold std ≲ 0.35, (2) **fold** Top-3 accuracy ≳ 0.5, "
        "(3) uncertainty that correlates with error, (4) explicit PPO cost accounting."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt_float(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.3f}"


def write_phase2_report(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote Phase 2 report: %s", path)
    return path


def write_model_comparison_table(oof: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "feature_set",
        "target",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "kendall",
        "ranking_status",
        "fold_spearman_mean",
        "fold_spearman_std",
        "n_folds_defined_spearman",
    ]
    cols = [c for c in cols if c in oof.columns]
    oof[cols].sort_values(["target", "spearman"], ascending=[True, False]).to_csv(
        path, index=False
    )
    return path
