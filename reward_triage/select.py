"""Candidate selection: top-k, OOD keep, audit quota, selection probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from reward_triage.schema import Prediction

Decision = Literal["train", "skip", "audit"]


@dataclass
class SelectionItem:
    index: int
    decision: Decision
    probability: float
    score: float | None
    reasons: list[str]


def select_candidates(
    predictions: Sequence[Prediction | None],
    *,
    hard_reject: Sequence[bool],
    k_frac: float = 0.5,
    audit_frac: float = 0.15,
    rng: np.random.Generator | None = None,
) -> list[SelectionItem]:
    """Select which candidates to train fully.

    Rules:
    - hard Tier0 rejects → skip
    - OOD (in_distribution=False) always train
    - among remaining ID candidates, keep top-k by predicted mean
    - audit random sample from the filtered remainder
    """
    n = len(predictions)
    if len(hard_reject) != n:
        raise ValueError("hard_reject length mismatch")
    rng = rng or np.random.default_rng(0)

    items: list[SelectionItem] = [
        SelectionItem(index=i, decision="skip", probability=0.0, score=None, reasons=[])
        for i in range(n)
    ]

    alive: list[int] = []
    ood_keep: list[int] = []
    for i in range(n):
        if hard_reject[i]:
            items[i].reasons.append("tier0_hard_reject")
            continue
        pred = predictions[i]
        if pred is None:
            # No score yet → keep (conservative) during warm-up gaps.
            alive.append(i)
            items[i].reasons.append("missing_prediction_keep")
            continue
        items[i].score = float(pred.mean)
        if not pred.in_distribution:
            ood_keep.append(i)
            items[i].reasons.append("ood_bypass")
        else:
            alive.append(i)

    # Rank ID (and missing) candidates by score (missing → -inf last unless kept separately).
    ranked = sorted(
        alive,
        key=lambda i: (
            predictions[i].mean if predictions[i] is not None else float("-inf")
        ),
        reverse=True,
    )
    k = max(1, int(round(len(ranked) * k_frac))) if ranked else 0
    top = set(ranked[:k])
    rest = [i for i in ranked if i not in top]
    n_audit = int(round(len(rest) * audit_frac)) if rest else 0
    if n_audit > 0:
        audit_idx = list(rng.choice(rest, size=min(n_audit, len(rest)), replace=False))
    else:
        audit_idx = []
    audit = set(int(x) for x in audit_idx)

    p_top = k / max(len(ranked), 1)
    p_audit = max(audit_frac, 1e-6)

    for i in ood_keep:
        items[i].decision = "train"
        items[i].probability = 1.0
    for i in top:
        items[i].decision = "train"
        items[i].probability = p_top
        items[i].reasons.append("active_topk")
    for i in audit:
        items[i].decision = "audit"
        items[i].probability = p_audit
        items[i].reasons.append("active_audit")
    for i in rest:
        if i in audit:
            continue
        items[i].decision = "skip"
        items[i].probability = 0.0
        items[i].reasons.append("active_filtered")

    return items
