"""Adapter for CrowdNav++ reward candidates.

VERIFY: CrowdNavEnv.step info exposes goal_distance, prev_goal_distance,
min_obstacle_clearance, collision, success, reward_components, reward_weighted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reward_triage.adapters.base import RewardAdapter
from reward_triage.executor import _call_reward, _load_callable


class CrowdNavAdapter(RewardAdapter):
    """Same call contract as TwoD; uses CrowdNav info field names in transitions."""

    def __init__(self, *, allow_torch: bool = False) -> None:
        self._fn: Any = None
        self._allow_torch = allow_torch

    def load(self, code: str) -> None:
        self._fn = _load_callable(code, allow_torch=self._allow_torch)

    def call(
        self, transition: Mapping[str, Any]
    ) -> tuple[float, dict[str, float] | None]:
        if self._fn is None:
            raise RuntimeError("CrowdNavAdapter.load(code) must be called first")
        # Prefer env-provided clearance / progress signals when present.
        tr = dict(transition)
        if "clearance" not in tr and "min_obstacle_clearance" in tr:
            tr["clearance"] = tr["min_obstacle_clearance"]
        if "progress" not in tr and "prev_goal_distance" in tr and "goal_distance" in tr:
            tr["progress"] = float(tr["prev_goal_distance"]) - float(tr["goal_distance"])
        return _call_reward(self._fn, tr)
