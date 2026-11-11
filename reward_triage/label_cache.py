"""Disk label cache keyed by code_hash (reuse PPO labels across runs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reward_triage.store import SeedLabel


class LabelCache:
    """JSONL cache: one file per code_hash under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.saved_steps = 0

    def _path(self, code_hash: str) -> Path:
        safe = "".join(c for c in code_hash if c.isalnum() or c in "-_")[:64]
        return self.root / f"{safe}.json"

    def get(self, code_hash: str) -> list[SeedLabel] | None:
        path = self._path(code_hash)
        if not path.is_file():
            self.misses += 1
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        labels = [SeedLabel.from_dict(x) for x in data.get("labels", [])]
        if not labels:
            self.misses += 1
            return None
        self.hits += 1
        self.saved_steps += int(
            sum(max(0, int(lab.train_steps)) for lab in labels)
        )
        return labels

    def put(
        self,
        code_hash: str,
        labels: list[SeedLabel],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "code_hash": code_hash,
            "labels": [lab.to_dict() for lab in labels],
            "meta": meta or {},
        }
        self._path(code_hash).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def stats(self) -> dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "saved_ppo_steps": int(self.saved_steps),
        }
