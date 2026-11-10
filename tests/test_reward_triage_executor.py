"""Sandbox executor tests (multiprocessing; no torch)."""

from __future__ import annotations

from reward_triage.executor import platform_isolation_note, run_candidate_on_probes
from reward_triage.probes import make_synthetic_probe_batch

GOOD_CODE = """
def compute_reward(transition):
    return float(transition.get("progress", 0.0)), {"progress": float(transition.get("progress", 0.0))}
"""

NAN_CODE = """
def compute_reward(transition):
    return float("nan")
"""

BOOM_CODE = """
def compute_reward(transition):
    raise RuntimeError("boom")
"""

# Busy-loop timeout without importing time (not on allowlist).
BUSY_TIMEOUT_CODE = """
def compute_reward(transition):
    x = 0
    while True:
        x += 1
    return float(x)
"""

IMPORT_BLOCKED = """
import os
def compute_reward(transition):
    return float(os.getpid())
"""


def _transitions(n: int = 4):
    return make_synthetic_probe_batch("navigation_2d", n_per_policy=1, seed=1).transitions[:n]


def test_executor_ok():
    result = run_candidate_on_probes(GOOD_CODE, _transitions(), timeout_s=10.0)
    assert result.report.status == "ok"
    assert len(result.rewards) == 4
    assert result.components[0] is not None


def test_executor_exception():
    result = run_candidate_on_probes(BOOM_CODE, _transitions(), timeout_s=10.0)
    assert result.report.status == "exec_error"
    assert any("boom" in r for r in result.report.reasons)


def test_executor_timeout():
    result = run_candidate_on_probes(BUSY_TIMEOUT_CODE, _transitions(1), timeout_s=1.0)
    assert result.report.status == "timeout"


def test_executor_blocks_disallowed_import():
    result = run_candidate_on_probes(IMPORT_BLOCKED, _transitions(1), timeout_s=10.0)
    assert result.report.status == "exec_error"


def test_platform_note_nonempty():
    note = platform_isolation_note()
    assert "multiprocessing" in note.lower() or "Windows" in note or "POSIX" in note


def test_nan_rewards_returned_for_tier0():
    result = run_candidate_on_probes(NAN_CODE, _transitions(2), timeout_s=10.0)
    assert result.report.status == "ok"
    assert len(result.rewards) == 2
