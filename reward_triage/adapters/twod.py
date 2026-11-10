"""Adapter for native 2D navigation reward candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reward_triage.adapters.base import RewardAdapter
from reward_triage.executor import _call_reward, _load_callable


class TwoDAdapter(RewardAdapter):
    """Expects ``compute_reward(transition) -> float | (float, dict)``.

    Transition keys mirror WeightedRewardFunction inputs: prev_position, position,
    action, prev_action, prev_goal_distance, goal_distance, success, collision,
    obstacles, target, robot_radius, etc.
    """

    def __init__(self, *, allow_torch: bool = False) -> None:
        self._fn: Any = None
        self._allow_torch = allow_torch

    def load(self, code: str) -> None:
        self._fn = _load_callable(code, allow_torch=self._allow_torch)

    def call(
        self, transition: Mapping[str, Any]
    ) -> tuple[float, dict[str, float] | None]:
        if self._fn is None:
            raise RuntimeError("TwoDAdapter.load(code) must be called first")
        return _call_reward(self._fn, transition)
