"""Abstract reward adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class RewardAdapter(ABC):
    """Load candidate source and evaluate on a transition dict."""

    @abstractmethod
    def load(self, code: str) -> None:
        """Compile / bind candidate code."""

    @abstractmethod
    def call(
        self, transition: Mapping[str, Any]
    ) -> tuple[float, dict[str, float] | None]:
        """Return (total_reward, optional component dict)."""
