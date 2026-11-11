"""Local hand-written reward corpus (no API key)."""

from __future__ import annotations

from pathlib import Path

from reward_triage.llm_client import LocalCorpusLLMClient, make_llm_client
from reward_triage.local_rewards import TEMPLATES, sample_code
from reward_triage.pipeline import triage_candidate
from reward_triage.probes import load_probe_npz
from reward_triage.tier0 import is_hard_reject

ROOT = Path(__file__).resolve().parents[1]


def test_make_llm_local():
    client = make_llm_client("local", "local-corpus")
    assert isinstance(client, LocalCorpusLLMClient)


def test_corpus_has_no_deliberate_weak_families():
    names = {name for name, _ in TEMPLATES}
    assert "weak_near_zero" not in names
    assert len(TEMPLATES) >= 10


def test_local_rewards_pass_tier0_on_real_probes():
    batch = load_probe_npz(ROOT / "data" / "probes" / "navigation_2d.npz")
    client = LocalCorpusLLMClient()
    n = len(TEMPLATES)
    responses = client.complete(
        "Task: navigate\nParent code (none for generation 0):\n# none\n\nParent training stats:\n{}",
        n=n,
    )
    assert len(responses) == n
    for resp in responses:
        report, feats, _, _ = triage_candidate(resp.text, batch.transitions[:64], in_process=True)
        assert not is_hard_reject(report), (report.status, report.reasons, resp.meta)
        assert feats is not None


def test_mutate_path_uses_parent():
    parent = sample_code("Parent code (none for generation 0):\n# none\n\nParent training stats:\n{}", index=0)[0]
    prompt = f"Parent code:\n{parent}\n\nParent training stats:\n{{\"mean_success\": 0.1}}"
    code, meta = sample_code(prompt, index=1)
    assert meta["family"] == "mutate"
    assert "def compute_reward" in code
