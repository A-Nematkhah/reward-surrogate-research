"""Phase 1 plots and textual analysis from the reward–performance dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset.builder import WEIGHT_COLUMNS
from src.rewards.components import COMPONENT_NAMES
from src.utils.logging import get_logger

logger = get_logger("phase1_analysis")


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["training_status"] == "completed"].copy()


def _candidate_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across seeds per candidate."""
    completed = _completed(df)
    if completed.empty:
        return completed
    metric_cols = [
        c
        for c in completed.columns
        if c.endswith("_rate")
        or c.startswith("mean_")
        or c in ("train_duration_sec", "eval_duration_sec")
    ]
    weight_cols = [c for c in WEIGHT_COLUMNS if c in completed.columns]
    agg = (
        completed.groupby(["candidate_id", "family_id"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    # weights are constant per candidate; take first
    weights = completed.groupby("candidate_id", as_index=False)[weight_cols].first()
    return agg.merge(weights, on="candidate_id", how="left")


def generate_phase1_figures(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cand = _candidate_means(df)
    paths: list[Path] = []
    if cand.empty:
        logger.warning("No completed runs; skipping Phase 1 figures")
        return paths

    # 1. Success rate distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cand["success_rate"], bins=12, color="#3b6ea5", edgecolor="white")
    ax.set_xlabel("Success rate")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Success rate distribution across candidates")
    paths.append(_save(fig, output_dir / "01_success_rate_distribution.png"))

    # 2. Collision rate distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cand["collision_rate"], bins=12, color="#b85c38", edgecolor="white")
    ax.set_xlabel("Collision rate")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Collision rate distribution across candidates")
    paths.append(_save(fig, output_dir / "02_collision_rate_distribution.png"))

    # 3. Reward weights vs performance (dynamic grid for expanded component sets)
    weight_names = [
        name
        for name in COMPONENT_NAMES
        if f"weight_{name}" in cand.columns and cand[f"weight_{name}"].notna().any()
    ]
    n_w = max(len(weight_names), 1)
    n_cols = min(4, n_w)
    n_rows = int(np.ceil(n_w / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for i, name in enumerate(weight_names):
        col = f"weight_{name}"
        axes_arr[i].scatter(cand[col], cand["success_rate"], alpha=0.75, c="#2f6f4e")
        axes_arr[i].set_xlabel(col)
        axes_arr[i].set_ylabel("success_rate")
        axes_arr[i].set_title(name)
    for j in range(len(weight_names), len(axes_arr)):
        axes_arr[j].axis("off")
    fig.suptitle("Reward weights vs success rate")
    fig.tight_layout()
    paths.append(_save(fig, output_dir / "03_weights_vs_success.png"))

    # 4. Training time per candidate
    fig, ax = plt.subplots(figsize=(10, 4))
    order = cand.sort_values("train_duration_sec")
    ax.bar(order["candidate_id"], order["train_duration_sec"], color="#6b7280")
    ax.set_ylabel("Train duration (s)")
    ax.set_title("Training time per candidate (seed-averaged)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    paths.append(_save(fig, output_dir / "04_training_time.png"))

    # 5. Candidate performance comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    order = cand.sort_values("success_rate")
    ax.barh(order["candidate_id"], order["success_rate"], color="#3b6ea5", label="success")
    ax.barh(
        order["candidate_id"],
        order["collision_rate"],
        left=0,
        color="#b85c38",
        alpha=0.35,
        label="collision",
    )
    ax.set_xlabel("Rate")
    ax.set_title("Candidate performance comparison")
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    paths.append(_save(fig, output_dir / "05_candidate_comparison.png"))

    # 6. Correlation matrix of performance metrics
    metric_cols = [
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_episode_length",
        "mean_path_length",
        "mean_goal_time",
        "mean_min_obstacle_distance",
        "mean_evaluation_return",
    ]
    metric_cols = [c for c in metric_cols if c in cand.columns]
    corr = cand[metric_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_yticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(metric_cols, fontsize=8)
    ax.set_title("Spearman correlation of performance metrics")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    paths.append(_save(fig, output_dir / "06_metric_correlation.png"))

    # 7. Pareto front: success ↑, collision ↓, goal time ↓
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    gt = cand["mean_goal_time"].fillna(cand["mean_goal_time"].max() + 1)
    sc = ax.scatter(
        cand["collision_rate"],
        cand["success_rate"],
        c=gt,
        cmap="viridis",
        s=40,
        alpha=0.85,
    )
    pareto = _pareto_indices(
        cand["success_rate"].to_numpy(),
        -cand["collision_rate"].to_numpy(),
        -gt.to_numpy(),
    )
    ax.scatter(
        cand.iloc[pareto]["collision_rate"],
        cand.iloc[pareto]["success_rate"],
        facecolors="none",
        edgecolors="black",
        s=90,
        linewidths=1.5,
        label="Pareto set",
    )
    ax.set_xlabel("Collision rate (lower better)")
    ax.set_ylabel("Success rate (higher better)")
    ax.set_title("Pareto view: success vs collision (color=goal time)")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="mean goal time")
    fig.tight_layout()
    paths.append(_save(fig, output_dir / "07_pareto_success_collision_goaltime.png"))

    logger.info("Wrote %d Phase 1 figures to %s", len(paths), output_dir)
    return paths


def _pareto_indices(*objectives: np.ndarray) -> list[int]:
    """Indices of non-dominated points (all objectives maximized)."""
    objs = np.column_stack(objectives)
    n = objs.shape[0]
    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(objs[j] >= objs[i]) and np.any(objs[j] > objs[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def build_phase1_report(
    df: pd.DataFrame,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> str:
    """Generate a markdown Phase 1 research report (honest, uncertainty-aware)."""
    completed = _completed(df)
    failed = df[df["training_status"] == "failed"] if not df.empty else df
    cand = _candidate_means(df)

    lines: list[str] = []
    lines.append("# Phase 1 Report: Reward–Performance Dataset")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Total rows: **{len(df)}**; completed: **{len(completed)}**; "
        f"failed: **{len(failed)}**."
    )
    if completed.empty:
        lines.append("- No completed runs; cannot assess reward diversity.")
        return "\n".join(lines)

    lines.append(
        f"- Candidates with completed runs: **{completed['candidate_id'].nunique()}** "
        f"across **{completed['family_id'].nunique()}** families."
    )
    lines.append("")

    # Top / bottom performers (seed-averaged); do not claim universal best
    lines.append("## Which candidates performed well / poorly?")
    lines.append("")
    lines.append(
        "Rankings below are **specific to this environment, PPO budget, and seeds**. "
        "They are not universal claims about reward design."
    )
    lines.append("")
    top = cand.sort_values("success_rate", ascending=False).head(5)
    bottom = cand.sort_values("success_rate", ascending=True).head(5)
    lines.append("### Highest seed-averaged success rate")
    lines.append("")
    lines.append("| candidate_id | family_id | success | collision | timeout | mean_goal_time |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['candidate_id']} | {row['family_id']} | "
            f"{row['success_rate']:.3f} | {row['collision_rate']:.3f} | "
            f"{row['timeout_rate']:.3f} | {_fmt(row.get('mean_goal_time'))} |"
        )
    lines.append("")
    lines.append("### Lowest seed-averaged success rate")
    lines.append("")
    lines.append("| candidate_id | family_id | success | collision | timeout |")
    lines.append("|---|---|---:|---:|---:|")
    for _, row in bottom.iterrows():
        lines.append(
            f"| {row['candidate_id']} | {row['family_id']} | "
            f"{row['success_rate']:.3f} | {row['collision_rate']:.3f} | "
            f"{row['timeout_rate']:.3f} |"
        )
    lines.append("")

    # Behavioral diversity
    lines.append("## Did different rewards produce different behaviors?")
    lines.append("")
    sr = cand["success_rate"]
    cr = cand["collision_rate"]
    lines.append(
        f"- Success rate range: [{sr.min():.3f}, {sr.max():.3f}] "
        f"(span={sr.max() - sr.min():.3f}, std={sr.std(ddof=0):.3f})."
    )
    lines.append(
        f"- Collision rate range: [{cr.min():.3f}, {cr.max():.3f}] "
        f"(span={cr.max() - cr.min():.3f})."
    )
    family_means = cand.groupby("family_id")["success_rate"].agg(["mean", "std", "count"])
    lines.append("")
    lines.append("### Family-level success (seed-averaged candidates)")
    lines.append("")
    lines.append("| family_id | n | mean success | std |")
    lines.append("|---|---:|---:|---:|")
    for family_id, row in family_means.sort_values("mean", ascending=False).iterrows():
        std = row["std"] if np.isfinite(row["std"]) else 0.0
        lines.append(
            f"| {family_id} | {int(row['count'])} | {row['mean']:.3f} | {std:.3f} |"
        )
    lines.append("")

    # Seed stability
    lines.append("## Stability across training seeds")
    lines.append("")
    seed_counts = completed.groupby("candidate_id")["training_seed"].nunique()
    multi = seed_counts[seed_counts > 1]
    if len(multi) == 0:
        lines.append(
            "- Only one training seed per candidate in this dataset; "
            "seed variability cannot be estimated yet."
        )
    else:
        spreads = []
        for cid in multi.index:
            vals = completed.loc[completed["candidate_id"] == cid, "success_rate"]
            spreads.append((cid, float(vals.max() - vals.min()), float(vals.mean())))
        spreads.sort(key=lambda x: x[1], reverse=True)
        mean_spread = float(np.mean([s[1] for s in spreads]))
        lines.append(
            f"- Candidates with ≥2 seeds: **{len(spreads)}**; "
            f"mean success-rate seed spread: **{mean_spread:.3f}**."
        )
        lines.append("- Largest seed spreads (uncertainty caution):")
        lines.append("")
        lines.append("| candidate_id | seed spread | mean success |")
        lines.append("|---|---:|---:|")
        for cid, spread, mean_s in spreads[:5]:
            lines.append(f"| {cid} | {spread:.3f} | {mean_s:.3f} |")
    lines.append("")

    # Failures
    lines.append("## Training failures")
    lines.append("")
    if failed.empty:
        lines.append("- No failed runs recorded.")
    else:
        lines.append(
            f"- **{len(failed)}** failed runs are retained in the dataset "
            "(not silently discarded)."
        )
        for _, row in failed.head(10).iterrows():
            lines.append(
                f"  - `{row.get('candidate_id')}` seed={row.get('training_seed')}: "
                f"{row.get('failure_type')}: {row.get('failure_message')}"
            )
    lines.append("")

    # Surrogate readiness
    lines.append("## Is the dataset diverse enough for a surrogate?")
    lines.append("")
    span = float(sr.max() - sr.min())
    if span < 0.05:
        lines.append(
            "- **Caution:** performance levels are nearly identical. "
            "A surrogate may only learn a constant; investigate before Phase 2."
        )
    elif span < 0.2:
        lines.append(
            "- **Limited diversity:** some spread exists, but Phase 2 results "
            "should be interpreted cautiously."
        )
    else:
        lines.append(
            "- **Yes, provisionally:** multiple performance levels exist. "
            "Grouped family splits remain necessary to avoid optimistic leakage."
        )
    if validation_report:
        for warning in validation_report.get("warnings", []):
            lines.append(f"- Validation warning: {warning}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Toy 2D navigation only; no claim about CrowdNav or multi-agent settings.")
    lines.append("- Pilot PPO budgets may under-train some reward designs.")
    lines.append("- Evaluation return is descriptive and must not replace task metrics.")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "nan"
    return f"{float(value):.2f}"


def write_phase1_report(
    df: pd.DataFrame,
    path: str | Path,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_phase1_report(df, validation_report=validation_report)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote Phase 1 report: %s", path)
    return path
