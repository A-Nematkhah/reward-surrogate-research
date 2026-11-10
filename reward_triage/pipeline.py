"""End-to-end Tier 0 + feature extraction for one candidate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reward_triage.executor import (
    ExecResult,
    _call_reward,
    _load_callable,
    run_candidate_on_probes,
)
from reward_triage.features import extract_blackbox_features, feature_hash
from reward_triage.mb_features import group_transitions_into_episodes
from reward_triage.schema import ValidationReport
from reward_triage.tier0 import apply_tier0, is_hard_reject, should_extract_features


def triage_candidate(
    code: str,
    transitions: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = 10.0,
    allow_torch: bool = False,
    feature_groups: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
    episodes: Sequence[tuple[str, Sequence[Mapping[str, Any]]]] | None = None,
    in_process: bool = True,
) -> tuple[ValidationReport, dict[str, float] | None, str | None, ExecResult]:
    """Run sandbox + Tier 0; default features are Gate-1 ``mb`` group."""
    groups = tuple(feature_groups) if feature_groups is not None else ("mb",)
    exec_result = run_candidate_on_probes(
        code,
        transitions,
        timeout_s=timeout_s,
        allow_torch=allow_torch,
        in_process=in_process,
    )
    report = apply_tier0(exec_result)
    feats: dict[str, float] | None = None
    fhash: str | None = None

    score_fn = None
    if "mb" in groups and should_extract_features(report):
        # Same sandbox loader as the child (in-process for FD scoring).
        try:
            fn = _load_callable(code, allow_torch=allow_torch)

            def score_fn(transition: Mapping[str, Any]):  # noqa: ANN202
                return _call_reward(fn, transition)
        except Exception:  # noqa: BLE001
            score_fn = None

    if should_extract_features(report):
        try:
            feats = extract_blackbox_features(
                exec_result.rewards,
                list(transitions),
                exec_result.components,
                code=code,
                groups=groups,
                score_fn=score_fn,
                weights=weights,
                scales=scales,
                episodes=episodes
                if episodes is not None
                else group_transitions_into_episodes(list(transitions)),
            )
            fhash = feature_hash(feats)
        except Exception as exc:  # noqa: BLE001
            if not is_hard_reject(report):
                report = ValidationReport(
                    status="warn",
                    reasons=list(report.reasons) + [f"feature_extract_error: {exc}"],
                    timings=dict(report.timings),
                )
    return report, feats, fhash, exec_result
