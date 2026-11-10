"""Tier 0 hard filters for reward candidates.

Hard reject only for: exec_error, timeout, non_finite, constant.
Everything else yields warnings and features (never a reject) to avoid false rejects.
"""

from __future__ import annotations

import numpy as np

from reward_triage.executor import ExecResult
from reward_triage.schema import ValidationReport

CONSTANT_STD_EPS = 1e-8


def apply_tier0(exec_result: ExecResult) -> ValidationReport:
    """Update / finalize ValidationReport after sandboxed execution."""
    report = exec_result.report
    if report.status in ("exec_error", "timeout"):
        return report

    rewards = np.asarray(exec_result.rewards, dtype=np.float64)
    reasons = list(report.reasons)
    timings = dict(report.timings)

    if rewards.size == 0:
        return ValidationReport(
            status="exec_error",
            reasons=reasons + ["empty reward vector"],
            timings=timings,
        )

    if not np.all(np.isfinite(rewards)):
        return ValidationReport(
            status="non_finite",
            reasons=reasons + ["non-finite reward values on probes"],
            timings=timings,
        )

    if float(np.std(rewards)) < CONSTANT_STD_EPS:
        return ValidationReport(
            status="constant",
            reasons=reasons + [f"reward std < {CONSTANT_STD_EPS} across probes"],
            timings=timings,
        )

    # Soft warnings only — never hard-reject for these.
    if float(np.max(np.abs(rewards))) > 1e6:
        reasons.append("large_reward_magnitude")
    if float(np.mean(rewards == 0.0)) > 0.95:
        reasons.append("mostly_zero_rewards")

    status = "warn" if reasons else "ok"
    return ValidationReport(status=status, reasons=reasons, timings=timings)


def is_hard_reject(report: ValidationReport) -> bool:
    return report.status in ("exec_error", "timeout", "non_finite", "constant")


def should_extract_features(report: ValidationReport) -> bool:
    """Features are extracted unless execution itself failed before rewards."""
    return report.status not in ("exec_error", "timeout")
