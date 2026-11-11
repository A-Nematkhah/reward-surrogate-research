"""Adapt sandbox ``compute_reward(transition)`` to the 2D env reward_fn API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from reward_triage.executor import _call_reward, _load_callable


def transition_for_code(transition: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize env transition fields into sandbox-friendly plain data."""
    item: dict[str, Any] = {}
    for k, v in transition.items():
        if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "radius"):
            item[k] = [float(v.x), float(v.y), float(v.radius)]
        elif isinstance(v, list) and v and hasattr(v[0], "x"):
            item[k] = [[float(o.x), float(o.y), float(o.radius)] for o in v]
        elif hasattr(v, "tolist"):
            item[k] = v.tolist()
        elif isinstance(v, (list, tuple, dict, str, bool, int, float, type(None))):
            item[k] = v
        else:
            item[k] = v
    # Convenience aliases used by local corpus / LLM prompts.
    if "progress" not in item and "prev_goal_distance" in item and "goal_distance" in item:
        item["progress"] = float(item["prev_goal_distance"]) - float(item["goal_distance"])
    if "clearance" not in item and item.get("min_obstacle_clearance") is not None:
        item["clearance"] = float(item["min_obstacle_clearance"])
    ad = item.get("action")
    prev = item.get("prev_action")
    if item.get("action_delta") is None and ad is not None and prev is not None:
        try:
            item["action_delta"] = float(np.linalg.norm(np.asarray(ad) - np.asarray(prev)))
        except Exception:  # noqa: BLE001
            item["action_delta"] = 0.0
    return item


class CodeRewardFunction:
    """``reward_fn(env, transition) -> float`` backed by candidate source code."""

    def __init__(self, code: str, *, allow_torch: bool = False) -> None:
        self.code = code
        self._fn = _load_callable(code, allow_torch=allow_torch)
        self.last_components: dict[str, float] | None = None
        self.last_weighted: dict[str, float] | None = None

    def __call__(self, env: Any, transition: Mapping[str, Any]) -> float:  # noqa: ANN401
        del env  # unused; code sees the transition dict only
        total, comps = _call_reward(self._fn, transition_for_code(transition))
        self.last_components = comps
        self.last_weighted = None
        if not np.isfinite(total):
            raise RuntimeError(f"Non-finite code reward: {total}")
        return float(total)
