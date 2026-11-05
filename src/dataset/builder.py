"""Build the Phase 1 reward–performance dataset from completed runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.rewards.components import COMPONENT_NAMES
from src.utils.config import relpath_to_project
from src.utils.logging import get_logger

logger = get_logger("dataset.builder")

WEIGHT_COLUMNS = [f"weight_{name}" for name in COMPONENT_NAMES]
METRIC_COLUMNS = [
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "mean_episode_length",
    "mean_path_length",
    "mean_goal_time",
    "mean_min_obstacle_distance",
    "mean_evaluation_return",
]


def discover_status_files(experiments_root: str | Path) -> list[Path]:
    root = Path(experiments_root) / "runs"
    if not root.is_dir():
        return []
    return sorted(root.glob("*/seed_*/status.json"))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def status_to_record(
    status: Mapping[str, Any],
    *,
    status_path: Path | None = None,
) -> dict[str, Any]:
    """Flatten a run status.json into one dataset row."""
    weights = status.get("reward_weights") or {}
    metrics = status.get("metrics") or {}
    failure = status.get("failure")

    record: dict[str, Any] = {
        "candidate_id": status.get("candidate_id"),
        "family_id": status.get("family_id"),
        "training_seed": status.get("training_seed"),
        "training_status": status.get("training_status"),
        "total_timesteps": status.get("total_timesteps"),
        "train_duration_sec": status.get("train_duration_sec"),
        "eval_duration_sec": status.get("eval_duration_sec"),
        "device": status.get("device"),
        "model_path": status.get("model_path"),
        "failure_type": None if failure is None else failure.get("type"),
        "failure_message": None if failure is None else failure.get("message"),
        "status_path": (
            relpath_to_project(status_path) if status_path is not None else None
        ),
    }
    for name in COMPONENT_NAMES:
        record[f"weight_{name}"] = weights.get(name)
    for metric in METRIC_COLUMNS:
        record[metric] = metrics.get(metric)
    return record


def build_dataset_from_runs(
    experiments_root: str | Path,
    *,
    require_completed: bool = False,
    candidate_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Collect all run status files into a DataFrame.

    By default includes failed runs (with null metrics) so failures are visible.
    Set ``require_completed=True`` to keep only successful trainings.
    If ``candidate_ids`` is set, keep only those candidate directories.
    """
    allowed = None if candidate_ids is None else {str(c) for c in candidate_ids}
    paths = discover_status_files(experiments_root)
    records: list[dict[str, Any]] = []
    for path in paths:
        cand_from_path = path.parent.parent.name
        if allowed is not None and cand_from_path not in allowed:
            continue
        try:
            status = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read %s: %s", path, exc)
            records.append(
                {
                    "candidate_id": cand_from_path,
                    "family_id": None,
                    "training_seed": path.parent.name.replace("seed_", ""),
                    "training_status": "corrupt_status",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "status_path": relpath_to_project(path),
                }
            )
            continue
        record = status_to_record(status, status_path=path)
        if allowed is not None and str(record.get("candidate_id")) not in allowed:
            continue
        if require_completed and record.get("training_status") != "completed":
            continue
        records.append(record)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        sort_cols = [c for c in ("candidate_id", "training_seed") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def attach_run_configs(
    df: pd.DataFrame,
    experiments_root: str | Path,
) -> pd.DataFrame:
    """Add environment/ppo fingerprint columns from run_config.json when present."""
    if df.empty or "status_path" not in df.columns:
        return df

    env_ids: list[Optional[str]] = []
    ppo_lrs: list[Optional[float]] = []
    n_eval_eps: list[Optional[int]] = []
    for status_path in df["status_path"]:
        if not status_path:
            env_ids.append(None)
            ppo_lrs.append(None)
            n_eval_eps.append(None)
            continue
        cfg_path = Path(status_path).with_name("run_config.json")
        if not cfg_path.is_file():
            env_ids.append(None)
            ppo_lrs.append(None)
            n_eval_eps.append(None)
            continue
        cfg = _load_json(cfg_path)
        env_ids.append(cfg.get("environment", {}).get("env_id"))
        ppo_lrs.append(cfg.get("ppo", {}).get("learning_rate"))
        n_eval_eps.append(cfg.get("evaluation", {}).get("n_eval_episodes"))

    out = df.copy()
    out["env_id"] = env_ids
    out["ppo_learning_rate"] = ppo_lrs
    out["n_eval_episodes"] = n_eval_eps
    return out


def save_dataset(
    df: pd.DataFrame,
    *,
    csv_path: str | Path,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    outputs = {"csv": csv_path}

    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        outputs["parquet"] = parquet_path

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            **(dict(metadata) if metadata else {}),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        outputs["metadata"] = metadata_path

    logger.info("Wrote dataset (%d rows) to %s", len(df), csv_path)
    return outputs


def build_and_save_phase1_dataset(
    experiments_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    candidate_ids: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    experiments_root = Path(experiments_root)
    output_dir = Path(output_dir) if output_dir is not None else experiments_root / "dataset"
    df = build_dataset_from_runs(
        experiments_root,
        require_completed=False,
        candidate_ids=candidate_ids,
    )
    df = attach_run_configs(df, experiments_root)
    meta: dict[str, Any] = {
        "experiments_root": str(experiments_root),
        "n_completed": int((df["training_status"] == "completed").sum())
        if "training_status" in df.columns and not df.empty
        else 0,
        "n_failed": int((df["training_status"] == "failed").sum())
        if "training_status" in df.columns and not df.empty
        else 0,
    }
    if candidate_ids is not None:
        meta["n_candidates_filter"] = int(len(set(candidate_ids)))
    paths = save_dataset(
        df,
        csv_path=output_dir / "reward_performance_dataset.csv",
        parquet_path=output_dir / "reward_performance_dataset.parquet",
        metadata_path=output_dir / "dataset_metadata.json",
        metadata=meta,
    )
    return df, paths
