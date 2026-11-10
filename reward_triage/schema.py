"""Dataclasses for candidates, validation, and predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ValidationStatus = Literal[
    "ok",
    "exec_error",
    "timeout",
    "non_finite",
    "constant",
    "warn",
]


@dataclass
class CandidateRecord:
    """One reward-function candidate from a host generation loop."""

    candidate_id: str
    code: str
    code_hash: str
    host: str
    env_id: str
    lineage_id: str
    parent_id: str | None = None
    prompt_id: str | None = None
    generation: int = 0
    llm_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateRecord:
        return cls(
            candidate_id=str(data["candidate_id"]),
            code=str(data["code"]),
            code_hash=str(data["code_hash"]),
            host=str(data["host"]),
            env_id=str(data["env_id"]),
            lineage_id=str(data["lineage_id"]),
            parent_id=None if data.get("parent_id") is None else str(data["parent_id"]),
            prompt_id=None if data.get("prompt_id") is None else str(data["prompt_id"]),
            generation=int(data.get("generation", 0)),
            llm_meta=dict(data.get("llm_meta") or {}),
        )


@dataclass
class ValidationReport:
    """Tier-0 / executor outcome for one candidate."""

    status: ValidationStatus
    reasons: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "timings": dict(self.timings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidationReport:
        return cls(
            status=str(data["status"]),  # type: ignore[arg-type]
            reasons=list(data.get("reasons") or []),
            timings={str(k): float(v) for k, v in dict(data.get("timings") or {}).items()},
        )


@dataclass
class Prediction:
    """Surrogate score for a candidate (used by later phases)."""

    mean: float
    std: float
    interval: tuple[float, float]
    in_distribution: bool
    feature_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "interval": [float(self.interval[0]), float(self.interval[1])],
            "in_distribution": bool(self.in_distribution),
            "feature_hash": str(self.feature_hash),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Prediction:
        interval = data["interval"]
        return cls(
            mean=float(data["mean"]),
            std=float(data["std"]),
            interval=(float(interval[0]), float(interval[1])),
            in_distribution=bool(data["in_distribution"]),
            feature_hash=str(data["feature_hash"]),
        )
