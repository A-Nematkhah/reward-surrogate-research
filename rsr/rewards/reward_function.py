"""Configurable weighted reward functions built from interpretable components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from rsr.rewards.components import (
    COMPONENT_NAMES,
    RewardComponent,
    build_default_components,
    compute_all_components,
)


@dataclass(frozen=True)
class RewardSpec:
    """Serializable reward specification for a candidate."""

    candidate_id: str
    family_id: str
    weights: dict[str, float]
    scales: dict[str, float] = field(default_factory=dict)
    generation_seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "weights": dict(self.weights),
            "scales": dict(self.scales),
            "generation_seed": int(self.generation_seed),
            "metadata": dict(self.metadata),
            "components": list(COMPONENT_NAMES),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewardSpec":
        weights = {name: float(data["weights"].get(name, 0.0)) for name in COMPONENT_NAMES}
        scales_raw = data.get("scales") or {}
        scales = {name: float(scales_raw.get(name, 1.0)) for name in COMPONENT_NAMES}
        return cls(
            candidate_id=str(data["candidate_id"]),
            family_id=str(data["family_id"]),
            weights=weights,
            scales=scales,
            generation_seed=int(data.get("generation_seed", 0)),
            metadata=dict(data.get("metadata") or {}),
        )

    def normalized_weights(self) -> dict[str, float]:
        """All registry components present; missing keys default to 0."""
        return {name: float(self.weights.get(name, 0.0)) for name in COMPONENT_NAMES}


class WeightedRewardFunction:
    """
    R = sum_i w_i * s_i * c_i(s, a, s')

    where ``c_i`` are unweighted components, ``s_i`` optional scales, ``w_i`` weights.

    Exposes ``last_components`` and ``last_weighted`` after each call so the
    environment (or evaluators) can log per-component values.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        scales: Mapping[str, float] | None = None,
        components: Mapping[str, RewardComponent] | None = None,
        candidate_id: str | None = None,
        family_id: str | None = None,
        clip_abs: float | None = None,
    ) -> None:
        self.components = dict(components or build_default_components())
        self.weights = {name: float(weights.get(name, 0.0)) for name in COMPONENT_NAMES}
        scale_src = scales or {}
        self.scales = {
            name: float(scale_src.get(name, 1.0)) for name in COMPONENT_NAMES
        }
        for name, scale in self.scales.items():
            if scale == 0 or not np.isfinite(scale):
                raise ValueError(f"Invalid scale for '{name}': {scale}")
        self.candidate_id = candidate_id
        self.family_id = family_id
        self.clip_abs = clip_abs

        self.last_components: dict[str, float] = {n: 0.0 for n in COMPONENT_NAMES}
        self.last_weighted: dict[str, float] = {n: 0.0 for n in COMPONENT_NAMES}
        self.last_total: float = 0.0

    @classmethod
    def from_spec(
        cls,
        spec: RewardSpec,
        *,
        components: Mapping[str, RewardComponent] | None = None,
        clip_abs: float | None = None,
    ) -> "WeightedRewardFunction":
        return cls(
            spec.weights,
            scales=spec.scales,
            components=components,
            candidate_id=spec.candidate_id,
            family_id=spec.family_id,
            clip_abs=clip_abs,
        )

    def compute_components(self, transition: Mapping[str, Any]) -> dict[str, float]:
        return compute_all_components(transition, self.components)

    def __call__(self, env: Any, transition: Mapping[str, Any]) -> float:
        del env  # Components are transition-only (no privileged env access needed)
        raw = self.compute_components(transition)
        weighted: dict[str, float] = {}
        total = 0.0
        for name in COMPONENT_NAMES:
            term = self.weights[name] * self.scales[name] * raw[name]
            if not np.isfinite(term):
                raise RuntimeError(
                    f"Non-finite weighted term '{name}': "
                    f"w={self.weights[name]}, s={self.scales[name]}, c={raw[name]}"
                )
            weighted[name] = float(term)
            total += term

        if self.clip_abs is not None:
            total = float(np.clip(total, -self.clip_abs, self.clip_abs))

        if not np.isfinite(total):
            raise RuntimeError(f"Non-finite total reward: {total}")

        self.last_components = raw
        self.last_weighted = weighted
        self.last_total = float(total)
        return float(total)

    def breakdown(self) -> dict[str, Any]:
        return {
            "components": dict(self.last_components),
            "weighted": dict(self.last_weighted),
            "total": self.last_total,
            "weights": dict(self.weights),
            "scales": dict(self.scales),
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
        }

    def to_spec(self, *, generation_seed: int = 0, metadata: Optional[dict] = None) -> RewardSpec:
        if self.candidate_id is None or self.family_id is None:
            raise ValueError("candidate_id and family_id are required to build a RewardSpec")
        return RewardSpec(
            candidate_id=self.candidate_id,
            family_id=self.family_id,
            weights=dict(self.weights),
            scales=dict(self.scales),
            generation_seed=generation_seed,
            metadata=metadata or {},
        )
