"""Phase 1 experiment runner: train/evaluate all candidates with resume support."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.features.early_behavioral import resolve_early_checkpoints
from src.rewards.candidate_generator import load_candidates
from src.rewards.reward_function import RewardSpec
from src.training.parallel_run_jobs import phase1_train_job
from src.training.ppo_trainer import (
    is_run_complete,
    run_dir_for,
    train_candidate_seed,
)
from src.utils.logging import get_logger
from src.utils.parallel_jobs import map_maybe_parallel

logger = get_logger("experiment_runner")


def resolve_training_seeds(ppo_config: Mapping[str, Any]) -> list[int]:
    if ppo_config.get("seeds") is not None:
        seeds = [int(s) for s in ppo_config["seeds"]]
    else:
        n = int(ppo_config.get("n_seeds", 1))
        seeds = list(range(n))
    n_seeds = int(ppo_config.get("n_seeds", len(seeds)))
    if len(seeds) < n_seeds:
        raise ValueError(
            f"ppo.seeds has {len(seeds)} values but ppo.n_seeds={n_seeds}"
        )
    return seeds[:n_seeds]


def iter_jobs(
    candidates: Sequence[RewardSpec],
    seeds: Sequence[int],
) -> list[tuple[RewardSpec, int]]:
    return [(cand, int(seed)) for cand in candidates for seed in seeds]


def _skip_complete(
    run_dir: Path,
    *,
    ppo_config: Mapping[str, Any],
    results: list[dict[str, Any]],
) -> bool:
    expected_timesteps = int(ppo_config["total_timesteps"])
    early_cfg = dict(ppo_config.get("early_behavioral") or {})
    early_enabled = bool(early_cfg.get("enabled", False))
    expected_early = resolve_early_checkpoints(early_cfg) if early_enabled else None
    if not is_run_complete(
        run_dir,
        expected_timesteps=expected_timesteps,
        require_early_features=early_enabled,
        expected_early_checkpoints=expected_early,
    ):
        return False
    with (run_dir / "status.json").open("r", encoding="utf-8") as handle:
        results.append(json.load(handle))
    return True


def run_phase1(
    *,
    candidates: Sequence[RewardSpec],
    env_config: Mapping[str, Any],
    ppo_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiments_root: str | Path,
    candidate_ids: Optional[Sequence[str]] = None,
    max_jobs: Optional[int] = None,
    resume: bool = True,
    stop_on_failure: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Train and evaluate each (candidate, seed) pair.

    Completed runs are skipped when ``resume=True``. Failed runs are recorded
    and counted; by default the runner continues so the dataset remains useful.
    Set ``stop_on_failure=True`` to abort on the first exception.
    """
    experiments_root = Path(experiments_root)
    experiments_root.mkdir(parents=True, exist_ok=True)

    if candidate_ids is not None:
        wanted = set(candidate_ids)
        candidates = [c for c in candidates if c.candidate_id in wanted]
        missing = wanted - {c.candidate_id for c in candidates}
        if missing:
            raise ValueError(f"Unknown candidate_ids: {sorted(missing)}")

    seeds = resolve_training_seeds(ppo_config)
    jobs = iter_jobs(candidates, seeds)
    if max_jobs is not None:
        jobs = jobs[: int(max_jobs)]

    manifest = {
        "n_candidates": len(candidates),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "n_jobs": len(jobs),
        "resume": resume,
        "workers": max(1, int(workers)),
        "ppo_total_timesteps": int(ppo_config["total_timesteps"]),
        "started_at": time.time(),
    }
    _write_json(experiments_root / "phase1_manifest.json", manifest)

    results: list[dict[str, Any]] = []
    n_completed = n_skipped = n_failed = 0
    t0 = time.perf_counter()
    n_workers = max(1, int(workers))

    if n_workers > 1:
        pending: list[dict[str, Any]] = []
        for spec, seed in jobs:
            run_dir = run_dir_for(experiments_root, spec.candidate_id, seed)
            if resume and _skip_complete(
                run_dir,
                ppo_config=ppo_config,
                results=results,
            ):
                n_skipped += 1
                continue
            pending.append(
                {
                    "reward_spec": spec.to_dict(),
                    "candidate_id": spec.candidate_id,
                    "family_id": spec.family_id,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "env_config": dict(env_config),
                    "ppo_config": dict(ppo_config),
                    "eval_config": dict(eval_config),
                    "resume": resume,
                }
            )
        logger.info(
            "Phase 1 parallel: %d pending, %d already complete, workers=%d",
            len(pending),
            n_skipped,
            n_workers,
        )

        def _on_result(done: int, _job: dict[str, Any], status: dict[str, Any]) -> None:
            logger.info(
                "[%d/%d] %s seed=%s status=%s",
                done,
                len(pending),
                status.get("candidate_id"),
                status.get("training_seed"),
                status.get("training_status"),
            )

        trained = map_maybe_parallel(
            phase1_train_job,
            pending,
            workers=n_workers,
            on_result=_on_result,
        )
        for status in trained:
            results.append(status)
            if status.get("training_status") == "completed":
                n_completed += 1
            elif status.get("training_status") == "failed":
                n_failed += 1
        if stop_on_failure and n_failed:
            raise RuntimeError(f"Phase 1 failed for {n_failed} jobs")
    else:
        for idx, (spec, seed) in enumerate(jobs):
            run_dir = run_dir_for(experiments_root, spec.candidate_id, seed)
            logger.info(
                "[%d/%d] %s seed=%s -> %s",
                idx + 1,
                len(jobs),
                spec.candidate_id,
                seed,
                run_dir,
            )
            if resume and _skip_complete(
                run_dir,
                ppo_config=ppo_config,
                results=results,
            ):
                n_skipped += 1
                continue

            try:
                status = train_candidate_seed(
                    spec,
                    env_config=env_config,
                    ppo_config=ppo_config,
                    eval_config=eval_config,
                    seed=seed,
                    run_dir=run_dir,
                    resume_if_complete=resume,
                )
                results.append(status)
                if status.get("training_status") == "completed":
                    n_completed += 1
                elif status.get("training_status") == "failed":
                    n_failed += 1
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                failure_record = {
                    "candidate_id": spec.candidate_id,
                    "family_id": spec.family_id,
                    "training_seed": int(seed),
                    "training_status": "failed",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
                results.append(failure_record)
                if stop_on_failure:
                    raise
                logger.error(
                    "Continuing after failure for %s seed=%s (%s)",
                    spec.candidate_id,
                    seed,
                    exc,
                )

    summary = {
        "n_jobs": len(jobs),
        "n_completed_new": n_completed,
        "n_skipped_complete": n_skipped,
        "n_failed": n_failed,
        "duration_sec": float(time.perf_counter() - t0),
        "results": results,
    }
    _write_json(experiments_root / "phase1_summary.json", summary)
    logger.info(
        "Phase 1 finished: new=%d skipped=%d failed=%d (%.1fs)",
        n_completed,
        n_skipped,
        n_failed,
        summary["duration_sec"],
    )
    return summary


def load_phase1_candidates(
    experiments_root: str | Path,
    candidates_json: str | Path | None = None,
) -> list[RewardSpec]:
    root = Path(experiments_root)
    path = (
        Path(candidates_json)
        if candidates_json is not None
        else root / "candidates.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Candidates file not found: {path}. Run scripts/generate_candidates.py first."
        )
    return load_candidates(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
