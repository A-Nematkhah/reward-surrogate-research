"""Sparse rewards must not be hard-rejected on real probes of each env."""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_triage.executor import run_candidate_on_probes
from reward_triage.probes import load_probe_npz
from reward_triage.tier0 import apply_tier0, is_hard_reject

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "data" / "probes"

SPARSE_CODES = {
    "goal_only": """
def compute_reward(transition):
    g = 1.0 if transition.get('success') else 0.0
    return g, {'goal': g}
""",
    "collision_only": """
def compute_reward(transition):
    c = 1.0 if (transition.get('collision') or transition.get('out_of_bounds')) else 0.0
    return (-c), {'collision': c}
""",
    "time_only": """
def compute_reward(transition):
    return (-1.0), {'time': 1.0}
""",
}


@pytest.mark.parametrize("env_name", ["navigation_2d", "crowdnav", "irsim"])
@pytest.mark.parametrize("sparse_name", sorted(SPARSE_CODES))
def test_sparse_rewards_not_hard_rejected(env_name: str, sparse_name: str):
    path = PROBE_DIR / f"{env_name}.npz"
    if not path.exists():
        pytest.fail(f"missing real probe set {path}; build with --from-env")
    batch = load_probe_npz(path)
    assert batch.meta.get("synthetic") is False
    code = SPARSE_CODES[sparse_name]
    result = run_candidate_on_probes(code, batch.transitions, timeout_s=30.0, in_process=True)
    report = apply_tier0(result)
    assert not is_hard_reject(report), (
        f"{env_name}/{sparse_name}: hard reject {report.status} {report.reasons}. "
        "Fix probes or Tier0 rule, not this test."
    )
    assert report.status in ("ok", "warn")
