"""End-to-end Tier 0 + black-box feature extraction for one candidate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reward_triage.executor import ExecResult, run_candidate_on_probes
from reward_triage.features import extract_blackbox_features, feature_hash
from reward_triage.schema import ValidationReport
from reward_triage.tier0 import apply_tier0, is_hard_reject, should_extract_features


def triage_candidate(
    code: str,
    transitions: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = 10.0,
    allow_torch: bool = False,
    feature_groups: Sequence[str] | None = None,
) -> tuple[ValidationReport, dict[str, float] | None, str | None, ExecResult]:
    """Run sandbox + Tier 0; return report, features (or None), feature_hash, raw exec."""
    exec_result = run_candidate_on_probes(
        code,
        transitions,
        timeout_s=timeout_s,
        allow_torch=allow_torch,
    )
    report = apply_tier0(exec_result)
    feats: dict[str, float] | None = None
    fhash: str | None = None
    if should_extract_features(report) and not is_hard_reject(report):
        feats = extract_blackbox_features(
            exec_result.rewards,
            list(transitions),
            exec_result.components,
            code=code,
            groups=feature_groups,
        )
        fhash = feature_hash(feats)
    elif should_extract_features(report):
        # Hard reject (non_finite/constant) still may want diagnostic features.
        feats = extract_blackbox_features(
            exec_result.rewards,
            list(transitions),
            exec_result.components,
            code=code,
            groups=feature_groups,
        )
        fhash = feature_hash(feats)
    return report, feats, fhash, exec_result
