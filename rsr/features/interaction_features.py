"""Interaction features between reward components / weights."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rsr.features.probe_trajectories import ProbeEpisode
from rsr.rewards.components import COMPONENT_NAMES
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction

KEY_PAIRS = (
    ("progress", "collision"),
    ("progress", "obstacle"),
    ("goal", "collision"),
    ("goal", "obstacle"),
    ("progress", "time"),
    ("goal", "time"),
    ("smoothness", "progress"),
    ("energy", "progress"),
    ("risk", "progress"),
    ("potential", "goal"),
    ("nonlinear_progress", "collision"),
    ("distance", "goal"),
)


def extract_interaction_features(
    spec: RewardSpec,
    episodes: Sequence[ProbeEpisode] | None = None,
) -> dict[str, float]:
    """
    Weight products plus optional probe contribution interactions.

    Weight interactions are always available. Contribution interactions require
    probe episodes and are skipped when ``episodes`` is None/empty.
    """
    features: dict[str, float] = {}
    weights = {name: float(spec.weights.get(name, 0.0)) for name in COMPONENT_NAMES}

    for a, b in KEY_PAIRS:
        if a not in weights or b not in weights:
            continue
        features[f"ix_w_{a}_x_{b}"] = weights[a] * weights[b]
        features[f"ix_abs_w_{a}_x_{b}"] = abs(weights[a]) * abs(weights[b])

    # Safety vs progress aggregate interactions
    safety = abs(weights.get("collision", 0.0)) + abs(weights.get("obstacle", 0.0)) + abs(
        weights.get("risk", 0.0)
    )
    progress_like = abs(weights.get("progress", 0.0)) + abs(
        weights.get("nonlinear_progress", 0.0)
    ) + abs(weights.get("potential", 0.0))
    features["ix_safety_x_progress"] = safety * progress_like
    features["ix_goal_x_distance"] = weights.get("goal", 0.0) * weights.get("distance", 0.0)

    if not episodes:
        return features

    reward_fn = WeightedRewardFunction.from_spec(spec)
    sums = {name: 0.0 for name in COMPONENT_NAMES}
    n = 0
    for episode in episodes:
        for transition in episode.transitions:
            reward_fn(None, transition)
            for name in COMPONENT_NAMES:
                sums[name] += float(reward_fn.last_components.get(name, 0.0))
            n += 1
    if n == 0:
        return features

    means = {name: sums[name] / n for name in COMPONENT_NAMES}
    for a, b in KEY_PAIRS:
        if a not in means or b not in means:
            continue
        features[f"ix_contrib_{a}_x_{b}"] = abs(means[a] * means[b])
        features[f"ix_wcontrib_{a}_x_{b}"] = abs(
            weights[a] * means[a] * weights[b] * means[b]
        )
    return features
