"""Environment adapters that normalize LLM reward code to a common call shape."""

from __future__ import annotations

from reward_triage.adapters.base import RewardAdapter
from reward_triage.adapters.crowdnav import CrowdNavAdapter
from reward_triage.adapters.eureka import EurekaStyleAdapter
from reward_triage.adapters.irsim import IRSimAdapter
from reward_triage.adapters.twod import TwoDAdapter

__all__ = [
    "CrowdNavAdapter",
    "EurekaStyleAdapter",
    "IRSimAdapter",
    "RewardAdapter",
    "TwoDAdapter",
]
