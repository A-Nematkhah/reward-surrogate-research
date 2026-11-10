#!/usr/bin/env python3
"""Export seed-aggregated datasets and feature tables into tracked data/ trees.

Never exports model checkpoints. Refuses to overwrite a non-empty output dir
without --force. Asserts total exported size under 20 MB (else prints LFS note).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_BYTES = 20 * 1024 * 1024

# (env_key, run_id, dataset_csv, features_dir)
SOURCES: list[tuple[str, str, Path | None, Path | None]] = [
    (
        "navigation_2d",
        "phase1_108",
        PROJECT_ROOT / "experiments/phase1/dataset/reward_performance_dataset.csv",
        PROJECT_ROOT / "experiments/phase2/features",
    ),
    (
        "crowdnav",
        "crowdnav_local",
        PROJECT_ROOT / "experiments/crowdnav/dataset/reward_performance_dataset.csv",
        PROJECT_ROOT / "experiments/crowdnav/features",
    ),
    (
        "crowdnav",
        "v1_50k",
        PROJECT_ROOT / "experiments/crowdnav_v1_50k/dataset/reward_performance_dataset.csv",
        PROJECT_ROOT / "experiments/crowdnav_v1_50k/features",
    ),
    (
        "crowdnav",
        "v4_full_150k",
        PROJECT_ROOT
        / "experiments/crowdnav_v4_full_150k/dataset/reward_performance_dataset.csv",
        PROJECT_ROOT / "experiments/crowdnav_v4_full_150k/features",
    ),
    (
        "crowdnav",
        "v5_invisible_24",
        PROJECT_ROOT
        / "experiments/crowdnav_v5_invisible_24/dataset/reward_performance_dataset.csv",
        PROJECT_ROOT / "experiments/crowdnav_v5_invisible_24/features",
    ),
]

METRIC_COLS = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "out_of_bounds_rate",
    "mean_episode_length",
    "mean_path_length",
    "mean_goal_time",
    "mean_min_obstacle_distance",
    "mean_evaluation_return",
    "train_duration_sec",
    "eval_duration_sec",
    "total_timesteps",
]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _aggregate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seed rows to one row per candidate_id."""
    if "candidate_id" not in df.columns:
        raise ValueError("dataset missing candidate_id")
    if "training_seed" not in df.columns:
        return df.copy()

    weight_cols = [c for c in df.columns if c.startswith("weight_")]
    first_cols = [
        c
        for c in (
            "family_id",
            "env_id",
            "training_status",
            "ppo_learning_rate",
            "n_eval_episodes",
        )
        if c in df.columns
    ] + weight_cols

    named: dict[str, tuple[str, str]] = {c: (c, "first") for c in first_cols}
    for c in METRIC_COLS:
        if c in df.columns:
            named[c] = (c, "mean")
    named["n_seeds"] = ("training_seed", "nunique")
    if "success_rate" in df.columns:
        named["success_rate_std"] = ("success_rate", "std")
    return df.groupby("candidate_id", as_index=False).agg(**named)


def _copy_features(src: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pt", ".pth", ".zip", ".ckpt"}:
            continue
        # Prefer early / math / behavioral / meta / targets / manifest tables
        name = path.name.lower()
        keep = (
            name.startswith("features_")
            or name.endswith(".json")
            or "early" in name
        )
        if not keep:
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        written.append(target)
    return written


def _dir_nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def export_one(
    env_key: str,
    run_id: str,
    dataset_csv: Path | None,
    features_dir: Path | None,
    *,
    force: bool,
) -> dict[str, Any] | None:
    out_dir = PROJECT_ROOT / "data" / env_key / run_id
    missing: list[str] = []
    if dataset_csv is None or not dataset_csv.is_file():
        missing.append(str(dataset_csv))
    if features_dir is None or not features_dir.is_dir():
        missing.append(str(features_dir))
    if missing and (dataset_csv is None or not dataset_csv.is_file()):
        return {
            "env": env_key,
            "run_id": run_id,
            "status": "skipped_missing_source",
            "missing": missing,
        }

    if _dir_nonempty(out_dir) and not force:
        raise SystemExit(
            f"Refusing non-empty output dir {out_dir} (pass --force to overwrite)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    files_meta: list[dict[str, Any]] = []
    if dataset_csv is not None and dataset_csv.is_file():
        raw = pd.read_csv(dataset_csv)
        agg = _aggregate_candidates(raw)
        dest = out_dir / "candidates_seed_agg.csv"
        agg.to_csv(dest, index=False)
        files_meta.append(
            {
                "path": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(dest),
                "rows": int(len(agg)),
                "schema": list(agg.columns),
                "source_path": str(dataset_csv.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
            }
        )

    if features_dir is not None and features_dir.is_dir():
        feat_out = out_dir / "features"
        written = _copy_features(features_dir, feat_out)
        for path in written:
            try:
                rows = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))) - 1
            except OSError:
                rows = -1
            if path.suffix.lower() == ".csv":
                try:
                    schema = list(pd.read_csv(path, nrows=0).columns)
                except Exception:  # noqa: BLE001
                    schema = []
            else:
                schema = []
            files_meta.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "sha256": _sha256(path),
                    "rows": rows,
                    "schema": schema,
                    "source_path": str(
                        (features_dir / path.name).relative_to(PROJECT_ROOT)
                    ).replace("\\", "/"),
                }
            )

    manifest = {
        "env": env_key,
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "files": files_meta,
        "notes": [
            "No model checkpoints exported.",
            "Candidate table is seed-aggregated (mean metrics; weight_* from first seed).",
        ],
    }
    man_path = out_dir / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for env_key, run_id, ds, feats in SOURCES:
        print(f"Export {env_key}/{run_id} ...")
        results.append(export_one(env_key, run_id, ds, feats, force=args.force))

    data_root = PROJECT_ROOT / "data"
    total = 0
    if data_root.is_dir():
        total = sum(p.stat().st_size for p in data_root.rglob("*") if p.is_file())
    print(f"Total data/ size: {total} bytes ({total / (1024 * 1024):.2f} MiB)")
    if total > MAX_BYTES:
        print(
            "WARNING: data/ exceeds 20 MiB. Prefer Git LFS for large CSVs "
            "(see docs/phase_reports/phase_0.md)."
        )
        return 2

    summary_path = data_root / "EXPORT_SUMMARY.json"
    data_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"git_commit": _git_commit(), "exports": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
