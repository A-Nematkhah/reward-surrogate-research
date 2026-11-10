"""Tier 0 hard filters for reward candidates.

Hard reject only for: exec_error, timeout, non_finite, and degenerate
constant-near-zero rewards. A non-zero constant (e.g. pure time penalty of -1
per step) yields a warning — episode returns still vary with horizon — so we
do not false-reject sparse but valid LLM rewards.
"""

from __future__ import annotations

import numpy as np

from reward_triage.executor import ExecResult
from reward_triage.schema import ValidationReport

CONSTANT_STD_EPS = 1e-8
CONSTANT_MEAN_EPS = 1e-8


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

    std = float(np.std(rewards))
    mean_abs = float(np.abs(np.mean(rewards)))
    if std < CONSTANT_STD_EPS and mean_abs < CONSTANT_MEAN_EPS:
        return ValidationReport(
            status="constant",
            reasons=reasons
            + [f"degenerate near-zero constant reward (std<{CONSTANT_STD_EPS})"],
            timings=timings,
        )
    if std < CONSTANT_STD_EPS:
        # Non-zero constant (e.g. pure time cost): warn, do not hard-reject.
        reasons.append("constant_nonzero_reward")

    if float(np.max(np.abs(rewards))) > 1e6:
        reasons.append("large_reward_magnitude")
    if float(np.mean(np.abs(rewards) < 1e-12)) > 0.95:
        reasons.append("mostly_zero_rewards")

    status = "warn" if reasons else "ok"
    return ValidationReport(status=status, reasons=reasons, timings=timings)


def is_hard_reject(report: ValidationReport) -> bool:
    return report.status in ("exec_error", "timeout", "non_finite", "constant")


def should_extract_features(report: ValidationReport) -> bool:
    """Features are extracted unless execution itself failed before rewards."""
    return report.status not in ("exec_error", "timeout")
