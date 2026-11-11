"""Append-only per-environment label store (schema v2, JSONL)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from reward_triage.schema import CandidateRecord, ValidationReport

ModeName = Literal["off", "shadow", "active"]


@dataclass
class SeedLabel:
    seed: int
    success: float
    collision: float
    timeout: float
    return_mean: float
    eval_episodes: int
    train_steps: int
    wall_time_s: float
    status: str = "completed"
    failure: dict[str, Any] | None = None
    early_features: dict[str, float] | None = None
    run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeedLabel:
        early = data.get("early_features")
        return cls(
            seed=int(data["seed"]),
            success=float(data["success"]),
            collision=float(data["collision"]),
            timeout=float(data["timeout"]),
            return_mean=float(data["return_mean"]),
            eval_episodes=int(data["eval_episodes"]),
            train_steps=int(data["train_steps"]),
            wall_time_s=float(data["wall_time_s"]),
            status=str(data.get("status", "completed")),
            failure=None if data.get("failure") is None else dict(data["failure"]),
            early_features=(
                None
                if early is None
                else {str(k): float(v) for k, v in dict(early).items()}
            ),
            run_dir=None if data.get("run_dir") is None else str(data["run_dir"]),
        )


@dataclass
class SelectionLog:
    score: float | None
    probability: float
    decision: Literal["train", "skip", "audit"]
    mode: ModeName
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "probability": self.probability,
            "decision": self.decision,
            "mode": self.mode,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionLog:
        return cls(
            score=None if data.get("score") is None else float(data["score"]),
            probability=float(data["probability"]),
            decision=str(data["decision"]),  # type: ignore[arg-type]
            mode=str(data["mode"]),  # type: ignore[arg-type]
            reasons=list(data.get("reasons") or []),
        )


@dataclass
class StoreRecord:
    """One append-only row in the per-env label store."""

    candidate: CandidateRecord
    tier0: ValidationReport
    features: dict[str, float] | None
    feature_hash: str | None
    selection: SelectionLog
    labels: list[SeedLabel]
    audit: bool = False
    run_id: str = ""
    env_config_id: str = "2d"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "tier0": self.tier0.to_dict(),
            "features": self.features,
            "feature_hash": self.feature_hash,
            "selection": self.selection.to_dict(),
            "labels": [lab.to_dict() for lab in self.labels],
            "audit": bool(self.audit),
            "run_id": self.run_id,
            "env_config_id": self.env_config_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StoreRecord:
        return cls(
            candidate=CandidateRecord.from_dict(data["candidate"]),
            tier0=ValidationReport.from_dict(data["tier0"]),
            features=None if data.get("features") is None else {str(k): float(v) for k, v in dict(data["features"]).items()},
            feature_hash=None if data.get("feature_hash") is None else str(data["feature_hash"]),
            selection=SelectionLog.from_dict(data["selection"]),
            labels=[SeedLabel.from_dict(x) for x in list(data.get("labels") or [])],
            audit=bool(data.get("audit", False)),
            run_id=str(data.get("run_id", "")),
            env_config_id=str(data.get("env_config_id", "2d")),
        )


class LabelStore:
    """Append-only JSONL store; one file per environment configuration / run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, record: StoreRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def extend(self, records: Iterable[StoreRecord]) -> None:
        for rec in records:
            self.append(rec)

    def iter_records(self) -> Iterator[StoreRecord]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return iter(())
        def _gen() -> Iterator[StoreRecord]:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    yield StoreRecord.from_dict(json.loads(line))
        return _gen()

    def load_all(self) -> list[StoreRecord]:
        return list(self.iter_records())


def validate_store(records: list[StoreRecord]) -> list[str]:
    """Return list of validation errors (empty = ok)."""
    errors: list[str] = []
    ids = [r.candidate.candidate_id for r in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate candidate_id")
    for rec in records:
        cid = rec.candidate.candidate_id
        if not rec.candidate.code_hash:
            errors.append(f"{cid}: missing code_hash")
        if not rec.candidate.lineage_id:
            errors.append(f"{cid}: missing lineage_id")
        if rec.selection.decision in ("train", "audit"):
            if not rec.labels and rec.tier0.status in ("ok", "warn"):
                errors.append(f"{cid}: selected for train/audit but missing labels")
        parent = rec.candidate.parent_id
        if parent is not None and parent not in set(ids) and rec.candidate.generation > 0:
            # Parent may be from an earlier resumable file; warn only if same file missing.
            pass
        for lab in rec.labels:
            if lab.status == "completed":
                for key in ("success", "collision", "timeout"):
                    val = getattr(lab, key)
                    if not (0.0 <= float(val) <= 1.0):
                        errors.append(f"{cid}: {key} out of range: {val}")
    # Lineage consistency: generation 0 has no parent.
    for rec in records:
        if rec.candidate.generation == 0 and rec.candidate.parent_id is not None:
            errors.append(f"{rec.candidate.candidate_id}: gen0 has parent_id")
    return errors
