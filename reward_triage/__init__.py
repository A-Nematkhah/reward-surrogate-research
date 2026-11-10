"""Host-agnostic reward triage: sandboxed black-box features and Tier 0.

Importing this package must not require torch or gymnasium.
"""

from __future__ import annotations

from reward_triage.schema import CandidateRecord, Prediction, ValidationReport

__version__ = "0.1.0"

__all__ = [
    "CandidateRecord",
    "Prediction",
    "ValidationReport",
    "__version__",
]
