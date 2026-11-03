"""Generate a diverse pool of reward-function candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.rewards.components import COMPONENT_NAMES
from src.rewards.reward_function import RewardSpec, WeightedRewardFunction
from src.utils.reproducibility import make_rng


def _sample_weight(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    if low > high:
        raise ValueError(f"Invalid range [{low}, {high}]")
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


def generate_candidates(
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> list[RewardSpec]:
    """
    Sample reward candidates from family templates in ``config``.

    Each candidate receives:
      - unique candidate_id
      - family_id
      - weights
      - scales from config.component_scales
      - generation_seed (global) and per-candidate sample seed in metadata
    """
    generation_seed = int(seed if seed is not None else config["generation_seed"])
    rng = make_rng(generation_seed)
    scales = {
        name: float(config.get("component_scales", {}).get(name, 1.0))
        for name in COMPONENT_NAMES
    }
    families = config["families"]
    expected = int(config.get("n_candidates", 0))

    candidates: list[RewardSpec] = []
    global_index = 0

    for family_id, family_cfg in families.items():
        n = int(family_cfg["n"])
        ranges: Mapping[str, Sequence[float]] = family_cfg["weight_ranges"]
        for local_idx in range(n):
            # Derive a reproducible per-candidate seed from the global stream
            candidate_seed = int(rng.integers(0, 2**31 - 1))
            local_rng = make_rng(candidate_seed)

            weights: dict[str, float] = {}
            for name in COMPONENT_NAMES:
                if name not in ranges:
                    # Structural families activate a subset; others default to 0.
                    weights[name] = 0.0
                    continue
                lo, hi = ranges[name]
                weights[name] = _sample_weight(local_rng, float(lo), float(hi))

            candidate_id = f"{family_id}_{local_idx:02d}"
            spec = RewardSpec(
                candidate_id=candidate_id,
                family_id=str(family_id),
                weights=weights,
                scales=dict(scales),
                generation_seed=generation_seed,
                metadata={
                    "family_description": family_cfg.get("description", ""),
                    "family_index": local_idx,
                    "global_index": global_index,
                    "candidate_seed": candidate_seed,
                    "weight_ranges": {
                        k: [float(v[0]), float(v[1])] for k, v in ranges.items()
                    },
                },
            )
            candidates.append(spec)
            global_index += 1

    if expected and len(candidates) != expected:
        raise ValueError(
            f"Generated {len(candidates)} candidates but config.n_candidates={expected}. "
            "Ensure family counts sum to n_candidates."
        )

    _validate_diversity(candidates)
    return candidates


def _validate_diversity(candidates: Sequence[RewardSpec]) -> None:
    """Fail loudly if the pool is empty or all weight vectors are identical."""
    if not candidates:
        raise ValueError("Candidate pool is empty")
    weight_matrix = np.array(
        [[c.weights[name] for name in COMPONENT_NAMES] for c in candidates],
        dtype=np.float64,
    )
    # Pairwise uniqueness check (exact duplicates are suspicious for continuous sampling)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if np.allclose(weight_matrix[i], weight_matrix[j], atol=1e-12):
                raise ValueError(
                    f"Duplicate weight vectors for "
                    f"{candidates[i].candidate_id} and {candidates[j].candidate_id}"
                )

    # Require non-trivial spread on at least some components
    spreads = weight_matrix.max(axis=0) - weight_matrix.min(axis=0)
    if float(np.max(spreads)) < 1e-6:
        raise ValueError(
            "Candidate weight pool has negligible diversity across all components"
        )


def candidates_to_jsonable(candidates: Sequence[RewardSpec]) -> dict[str, Any]:
    return {
        "n_candidates": len(candidates),
        "component_names": list(COMPONENT_NAMES),
        "candidates": [c.to_dict() for c in candidates],
    }


def save_candidates(
    candidates: Sequence[RewardSpec],
    *,
    json_path: str | Path,
    candidates_dir: str | Path | None = None,
) -> Path:
    """
    Save the full pool as JSON and optionally one JSON file per candidate.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = candidates_to_jsonable(candidates)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)

    if candidates_dir is not None:
        out_dir = Path(candidates_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for spec in candidates:
            single = out_dir / f"{spec.candidate_id}.json"
            with single.open("w", encoding="utf-8") as handle:
                json.dump(spec.to_dict(), handle, indent=2)

    return path


def load_candidates(json_path: str | Path) -> list[RewardSpec]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [RewardSpec.from_dict(item) for item in payload["candidates"]]


def summarize_candidates(candidates: Sequence[RewardSpec]) -> dict[str, Any]:
    """Compact diversity summary for logging / reports."""
    by_family: dict[str, int] = {}
    for c in candidates:
        by_family[c.family_id] = by_family.get(c.family_id, 0) + 1

    weight_matrix = np.array(
        [[c.weights[name] for name in COMPONENT_NAMES] for c in candidates],
        dtype=np.float64,
    )
    stats = {}
    for i, name in enumerate(COMPONENT_NAMES):
        col = weight_matrix[:, i]
        stats[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "n_positive": int(np.sum(col > 0)),
            "n_negative": int(np.sum(col < 0)),
            "n_zero": int(np.sum(np.isclose(col, 0.0))),
        }

    return {
        "n_candidates": len(candidates),
        "n_families": len(by_family),
        "by_family": by_family,
        "weight_stats": stats,
    }


def make_reward_fn(spec: RewardSpec) -> WeightedRewardFunction:
    """Convenience factory used by training / evaluation code."""
    return WeightedRewardFunction.from_spec(spec)
