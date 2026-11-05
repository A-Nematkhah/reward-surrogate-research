#!/usr/bin/env python3
"""Build markdown report + figures for model comparison (new results dir only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.model_comparison import load_yaml_config, resolve_path
from src.utils.logging import get_logger, setup_logging


def _df_to_md(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional ``tabulate`` package."""
    if df.empty:
        return "_Empty table._"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append("nan" if not np.isfinite(v) else f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_comparison.yaml",
    )
    p.add_argument("--exp-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    return p.parse_args()


def _agg_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["model", "feature_set", "target"]
    num_cols = [
        c
        for c in df.columns
        if c not in keys + ["repeat", "skipped"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby(keys, as_index=False)[num_cols].mean(numeric_only=True)


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("report_model_comparison")
    cfg = load_yaml_config(args.config)
    if args.exp_dir is not None:
        cfg["output_dir"] = str(args.exp_dir)
    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)
    exp_dir = resolve_path(cfg.get("output_dir", "experiments/model_comparison"))
    out_dir = resolve_path(cfg.get("results_dir", "results/model_comparison"))
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sum_path = exp_dir / "metrics_summary.csv"
    if not sum_path.is_file():
        logger.error("Missing %s — run scripts/run_model_comparison.py first", sum_path)
        return 1
    raw = pd.read_csv(sum_path)
    summary = _agg_summary(raw)
    bias_path = exp_dir / "selection_bias_guard.csv"
    bias = pd.read_csv(bias_path) if bias_path.is_file() else pd.DataFrame()
    paired_path = exp_dir / "paired_vs_baseline.csv"
    paired = pd.read_csv(paired_path) if paired_path.is_file() else pd.DataFrame()
    run_cfg = {}
    rc_path = exp_dir / "run_config.json"
    if rc_path.is_file():
        run_cfg = json.loads(rc_path.read_text(encoding="utf-8"))

    primary = cfg.get("primary", {})
    p_target = primary.get("target", "success_rate")
    p_fs = primary.get("feature_set", "early+math")
    p_models = list(primary.get("models", []))

    primary_tbl = summary[
        (summary["target"] == p_target)
        & (summary["feature_set"] == p_fs)
        & (summary["model"].isin(p_models))
    ].copy()
    primary_tbl = primary_tbl.sort_values("within_family_spearman_mean", ascending=False)
    leaderboard_path = out_dir / "leaderboard.csv"
    primary_tbl.to_csv(leaderboard_path, index=False)

    # Figures
    success = summary[summary["target"] == "success_rate"]
    if not success.empty:
        pivot = success.pivot_table(
            index="model", columns="feature_set", values="spearman", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_title("Pooled Spearman (success_rate) — exploratory grid")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(fig_dir / "heatmap_pooled_spearman_success.png", dpi=150)
        plt.close(fig)

        # Within-family bar with CI if present
        sub = success.copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        order = sub.sort_values("within_family_spearman_mean", ascending=False).head(20)
        labels = [f"{r.model}|{r.feature_set}" for r in order.itertuples()]
        ax.barh(labels[::-1], order["within_family_spearman_mean"].to_numpy()[::-1])
        if "boot_within_spearman_lo" in order.columns:
            y = np.arange(len(order))
            lo = order["boot_within_spearman_lo"].to_numpy()[::-1]
            hi = order["boot_within_spearman_hi"].to_numpy()[::-1]
            mean = order["within_family_spearman_mean"].to_numpy()[::-1]
            ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="none", ecolor="black")
        ax.set_xlabel("Within-family Spearman (mean)")
        ax.set_title("Within-family Spearman — top 20 configs")
        fig.tight_layout()
        fig.savefig(fig_dir / "within_family_spearman_ci.png", dpi=150)
        plt.close(fig)

        # Top-10 selected vs random vs oracle
        if "top10_selected_mean" in success.columns:
            top = (
                success.sort_values("top10_selected_mean", ascending=False)
                .groupby("model", as_index=False)
                .first()
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(top))
            w = 0.25
            ax.bar(x - w, top["top10_selected_mean"], width=w, label="selected")
            ax.bar(x, top["top10_random_mean"], width=w, label="random")
            ax.bar(x + w, top["top10_oracle_mean"], width=w, label="oracle")
            ax.set_xticks(x)
            ax.set_xticklabels(top["model"], rotation=45, ha="right")
            ax.set_ylabel("Mean success_rate of top-10")
            ax.legend()
            ax.set_title("Top-10 selection quality by model")
            fig.tight_layout()
            fig.savefig(fig_dir / "top10_selected_vs_oracle.png", dpi=150)
            plt.close(fig)

        # Accuracy vs PPO cost
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            success["needs_ppo_steps"],
            success["within_family_spearman_mean"],
            alpha=0.7,
        )
        for r in success.itertuples():
            if pd.notna(r.within_family_spearman_mean) and r.within_family_spearman_mean > 0.3:
                ax.annotate(
                    f"{r.model}",
                    (r.needs_ppo_steps, r.within_family_spearman_mean),
                    fontsize=7,
                    alpha=0.7,
                )
        ax.set_xlabel("needs_ppo_steps (feature cost)")
        ax.set_ylabel("Within-family Spearman")
        ax.set_title("Accuracy vs PPO feature cost")
        fig.tight_layout()
        fig.savefig(fig_dir / "accuracy_vs_ppo_cost.png", dpi=150)
        plt.close(fig)

        # Optional plotly HTML
        try:
            import plotly.express as px

            html_dir = fig_dir
            fig1 = px.imshow(
                pivot,
                title="Pooled Spearman heatmap (success_rate)",
                aspect="auto",
            )
            fig1.write_html(str(html_dir / "heatmap_pooled_spearman_success.html"))
            fig2 = px.bar(
                order,
                x="within_family_spearman_mean",
                y=labels,
                orientation="h",
                title="Within-family Spearman top 20",
            )
            fig2.write_html(str(html_dir / "within_family_spearman_ci.html"))
            if "top10_selected_mean" in success.columns:
                fig3 = px.bar(
                    top,
                    x="model",
                    y=["top10_selected_mean", "top10_random_mean", "top10_oracle_mean"],
                    barmode="group",
                    title="Top-10 selection quality",
                )
                fig3.write_html(str(html_dir / "top10_selected_vs_oracle.html"))
        except Exception as exc:  # noqa: BLE001
            logger.info("Plotly HTML skipped: %s", exc)

    # Markdown report
    lines: list[str] = []
    lines.append("# Model comparison report")
    lines.append("")
    lines.append("## Research question")
    lines.append("")
    lines.append(
        "Which model family best predicts PPO navigation performance of an **unseen** "
        "reward function from reward features, under leave-one-family-out (LOFO)?"
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "Outer evaluation is LOFO over 12 reward families. Hyperparameters and "
        "preprocessing (imputation, feature selection, target transforms) are tuned "
        "with **inner LOFO on training families only**. Primary metrics emphasize "
        "**within-family Spearman** and top-k selection quality, not only pooled Spearman. "
        f"Repeats={cfg.get('n_repeats')}; bootstrap resamples families "
        f"(n={cfg.get('n_bootstrap')}). Phase 1/2 artifacts were not modified."
    )
    lines.append("")
    lines.append("## Pre-registered primary results")
    lines.append("")
    lines.append(
        f"Pre-registered before looking at results: target=`{p_target}`, "
        f"feature_set=`{p_fs}`, models={p_models}."
    )
    lines.append("")
    if primary_tbl.empty:
        lines.append("_No primary rows present in metrics_summary (models may have been skipped)._")
    else:
        show = [
            c
            for c in (
                "model",
                "spearman",
                "within_family_spearman_mean",
                "within_family_spearman_std",
                "within_family_top3_hit",
                "top10_selected_mean",
                "top10_regret",
                "mae",
                "needs_ppo_steps",
            )
            if c in primary_tbl.columns
        ]
        lines.append(_df_to_md(primary_tbl[show]))
    lines.append("")
    lines.append(
        "This primary table is **not** cherry-picked from the full grid. "
        "Skipped primary models (optional deps) are listed in `run_config.json`."
    )
    lines.append("")
    lines.append("## Exploratory full grid")
    lines.append("")
    lines.append(
        "The tables/figures below explore many models × feature sets. "
        "Headline claims must not be the max over this grid without correction."
    )
    lines.append("")
    if not bias.empty:
        lines.append("### Selection-bias guard")
        lines.append("")
        lines.append(
            _df_to_md(
                bias.sort_values("median_spearman_across_feature_sets", ascending=False)
            )
        )
        lines.append("")
    explor = summary[summary["target"] == "success_rate"].sort_values(
        "within_family_spearman_mean", ascending=False
    ).head(25)
    show2 = [
        c
        for c in (
            "model",
            "feature_set",
            "spearman",
            "within_family_spearman_mean",
            "fold_spearman_std",
            "within_family_top3_hit",
            "needs_ppo_steps",
        )
        if c in explor.columns
    ]
    lines.append("### Top exploratory rows by within-family Spearman (success_rate)")
    lines.append("")
    if explor.empty:
        lines.append("_Empty._")
    else:
        lines.append(_df_to_md(explor[show2]))
    lines.append("")
    lines.append("## Within-family and top-k")
    lines.append("")
    lines.append(
        "Within-family ranking is the practical use-case. See "
        "`figures/within_family_spearman_ci.png` and `figures/top10_selected_vs_oracle.png`."
    )
    lines.append("")
    lines.append("## Accuracy vs PPO cost")
    lines.append("")
    lines.append(
        "Feature sets containing `early` cost ~5000 PPO steps per candidate; "
        "static/math/behavioral cost 0. See `figures/accuracy_vs_ppo_cost.png`."
    )
    lines.append("")
    lines.append("## Neural nets vs trees (honest)")
    lines.append("")
    nn = summary[summary["model"].isin(["mlp_mse", "mlp_rank", "tabpfn"])]
    trees = summary[
        summary["model"].isin(
            ["extra_trees", "random_forest", "lightgbm", "hist_gradient_boosting"]
        )
    ]
    if nn.empty:
        lines.append("Neural / TabPFN models were skipped or produced no rows.")
    else:
        lines.append(
            f"NN/TabPFN rows: {len(nn)}; tree rows: {len(trees)}. "
            "With n≈108 and p large, small MLPs are honest baselines, not expected winners."
        )
    lines.append("")
    lines.append("## Paired comparison vs ExtraTrees")
    lines.append("")
    if paired.empty:
        lines.append("_No paired table._")
    else:
        lines.append(
            "Mean per-family Spearman difference vs ExtraTrees on the same feature set "
            "(bootstrap CI over families; **low power with 12 families**)."
        )
        lines.append("")
        lines.append(_df_to_md(paired.head(30)))
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- n=108 candidates, 12 families; within-family signal is weak historically.")
    lines.append("- Toy 2D navigation only; not CrowdNav.")
    lines.append("- Optional packages may skip models; see run_config skipped list.")
    lines.append("- Exploratory max over many configs is selection-biased.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if not primary_tbl.empty and "within_family_spearman_mean" in primary_tbl.columns:
        best = primary_tbl.iloc[0]
        lines.append(
            f"Under the **pre-registered** setting, best available row is "
            f"`{best['model']}` (within-family Spearman≈{best.get('within_family_spearman_mean', float('nan')):.3f}, "
            f"pooled≈{best.get('spearman', float('nan')):.3f}). "
            "Use this as the decision baseline; treat exploratory winners as hypotheses only."
        )
    else:
        lines.append(
            "Primary models mostly unavailable; re-run after installing extras, "
            "or fall back to ExtraTrees from the exploratory grid with explicit caveats."
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Experiments: `{exp_dir}`")
    lines.append(f"- Leaderboard: `{leaderboard_path}`")
    lines.append(f"- Figures: `{fig_dir}`")
    if run_cfg.get("skipped"):
        lines.append(f"- Skipped: `{run_cfg.get('skipped')}`")
    lines.append("")

    report_path = out_dir / "model_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
