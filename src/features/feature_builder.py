"""Assemble static / behavioral / hybrid feature tables for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from src.features.behavioral_features import extract_behavioral_features
from src.features.early_behavioral import load_aggregated_early_features
from src.features.interaction_features import extract_interaction_features
from src.features.math_features import extract_math_features
from src.features.probe_trajectories import (
    ProbeEpisode,
    generate_probe_set,
    load_probe_set,
    save_probe_set,
)
from src.features.static_features import static_feature_matrix_from_dataframe
from src.rewards.components import COMPONENT_NAMES
from src.rewards.reward_function import RewardSpec
from src.utils.logging import get_logger

logger = get_logger("feature_builder")

FeatureSetName = Literal[
    "static", "behavioral", "probe", "math", "interaction", "early", "hybrid"
]

TARGET_COLUMNS = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_goal_time",
    "mean_path_length",
]


def aggregate_phase1_dataset(
    df: pd.DataFrame,
    *,
    how: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate multi-seed Phase 1 rows to one row per candidate.

    Weights/family are constant per candidate; metrics are aggregated.
    """
    completed = df[df["training_status"] == "completed"].copy()
    if completed.empty:
        raise ValueError("No completed rows to aggregate")

    for name in COMPONENT_NAMES:
        col = f"weight_{name}"
        if col not in completed.columns:
            completed[col] = 0.0

    weight_cols = [f"weight_{name}" for name in COMPONENT_NAMES]
    metric_cols = [c for c in TARGET_COLUMNS if c in completed.columns]
    extra_metrics = [
        c
        for c in completed.columns
        if (c.startswith("mean_") or c.endswith("_rate")) and c not in metric_cols
    ]
    metric_cols = sorted(set(metric_cols + extra_metrics))

    if how not in {"mean", "median", "per_seed"}:
        raise ValueError(f"Unknown aggregation: {how}")
    if how == "per_seed":
        return completed.reset_index(drop=True)

    agg_fn = "mean" if how == "mean" else "median"
    grouped = (
        completed.groupby(["candidate_id", "family_id"], as_index=False)[metric_cols]
        .agg(agg_fn)
    )
    weights = completed.groupby("candidate_id", as_index=False)[weight_cols].first()
    out = grouped.merge(weights, on="candidate_id", how="left")
    return out


def specs_from_dataframe(df: pd.DataFrame) -> list[RewardSpec]:
    specs: list[RewardSpec] = []
    for _, row in df.iterrows():
        weights = {
            name: float(row[f"weight_{name}"]) if f"weight_{name}" in row.index else 0.0
            for name in COMPONENT_NAMES
        }
        specs.append(
            RewardSpec(
                candidate_id=str(row["candidate_id"]),
                family_id=str(row["family_id"]),
                weights=weights,
                scales={name: 1.0 for name in COMPONENT_NAMES},
            )
        )
    return specs


def ensure_probe_set(
    env_config: Mapping[str, Any],
    probe_config: Mapping[str, Any],
    *,
    project_root: str | Path,
    regenerate: bool = False,
) -> list[ProbeEpisode]:
    root = Path(project_root)
    npz_path = root / probe_config["probe_set_path"]
    config_path = root / probe_config["probe_config_path"]

    if npz_path.is_file() and not regenerate:
        logger.info("Loading existing probe set from %s", npz_path)
        return load_probe_set(npz_path)

    episodes = generate_probe_set(env_config, probe_config)
    save_probe_set(
        episodes,
        npz_path=npz_path,
        config_path=config_path,
        probe_config=probe_config,
    )
    return episodes


def build_feature_tables(
    dataset_df: pd.DataFrame,
    episodes: Sequence[ProbeEpisode],
    *,
    seed_aggregation: str = "mean",
    include_family_one_hot: bool = False,
    experiments_root: str | Path | None = None,
    include_early_behavioral: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Return feature tables including ablation sets:

    meta, static, behavioral (alias probe), math, interaction, early, hybrid, targets.
    """
    agg = aggregate_phase1_dataset(dataset_df, how=seed_aggregation)
    meta = agg[["candidate_id", "family_id"]].copy()
    targets = agg[[c for c in TARGET_COLUMNS if c in agg.columns]].copy()

    static_df, _ = static_feature_matrix_from_dataframe(
        agg, include_family_one_hot=include_family_one_hot
    )
    specs = specs_from_dataframe(agg)
    behavioral_df = pd.DataFrame(
        [extract_behavioral_features(spec, episodes) for spec in specs]
    )
    math_df = pd.DataFrame([extract_math_features(spec, episodes) for spec in specs])
    interaction_df = pd.DataFrame(
        [extract_interaction_features(spec, episodes) for spec in specs]
    )

    early_df = pd.DataFrame(index=range(len(meta)))
    if include_early_behavioral:
        if experiments_root is None:
            raise ValueError(
                "experiments_root is required when include_early_behavioral=True"
            )
        early_how = "median" if seed_aggregation == "median" else "mean"
        early_df = load_aggregated_early_features(
            experiments_root,
            meta["candidate_id"].tolist(),
            how=early_how,
        )

    frames = [
        static_df,
        behavioral_df,
        math_df,
        interaction_df,
        early_df,
        targets,
        meta,
    ]
    for frame in frames:
        frame.reset_index(drop=True, inplace=True)

    hybrid_parts = [static_df, behavioral_df, math_df, interaction_df]
    if include_early_behavioral and early_df.shape[1] > 0:
        hybrid_parts.append(early_df)
    hybrid_df = pd.concat(hybrid_parts, axis=1)

    tables = {
        "meta": meta,
        "static": static_df,
        "behavioral": behavioral_df,
        "probe": behavioral_df.copy(),
        "math": math_df,
        "interaction": interaction_df,
        "early": early_df,
        "hybrid": hybrid_df,
        "targets": targets,
    }
    for name, frame in tables.items():
        if name in {"meta", "targets"}:
            continue
        if frame.shape[1] > 0:
            _assert_no_target_leakage(frame, name)
    return tables


def save_feature_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        csv_path = output_dir / f"features_{name}.csv"
        frame.to_csv(csv_path, index=False)
        paths[name] = csv_path

    def _ncols(key: str) -> int:
        frame = tables.get(key)
        return int(frame.shape[1]) if frame is not None else 0

    manifest = {
        "n_candidates": int(len(tables["meta"])),
        "n_static_features": _ncols("static"),
        "n_behavioral_features": _ncols("behavioral"),
        "n_math_features": _ncols("math"),
        "n_interaction_features": _ncols("interaction"),
        "n_early_features": _ncols("early"),
        "n_hybrid_features": _ncols("hybrid"),
        "target_columns": list(tables["targets"].columns),
        "candidate_ids": tables["meta"]["candidate_id"].tolist(),
        "leakage_checks": "passed",
    }
    manifest_path = output_dir / "features_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    paths["manifest"] = manifest_path
    logger.info(
        "Saved features: static=%d probe=%d math=%d ix=%d early=%d hybrid=%d -> %s",
        manifest["n_static_features"],
        manifest["n_behavioral_features"],
        manifest["n_math_features"],
        manifest["n_interaction_features"],
        manifest["n_early_features"],
        manifest["n_hybrid_features"],
        output_dir,
    )
    return paths


def _assert_no_target_leakage(features: pd.DataFrame, label: str) -> None:
    forbidden = set(TARGET_COLUMNS) | {
        "mean_evaluation_return",
        "mean_episode_length",
        "mean_min_obstacle_distance",
        "out_of_bounds_rate",
        "train_duration_sec",
        "eval_duration_sec",
    }
    overlap = forbidden.intersection(features.columns)
    if overlap:
        raise RuntimeError(f"{label} features leak targets/metrics: {sorted(overlap)}")
