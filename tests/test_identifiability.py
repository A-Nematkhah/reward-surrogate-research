"""Smoke tests for within-family identifiability diagnostic."""

from __future__ import annotations

import pandas as pd

from scripts.analyze_within_family_identifiability import build_table


def test_flat_family_not_identifiable():
    rows = []
    for fam, vals in [("flat", [0.01, 0.011, 0.012]), ("wide", [0.1, 0.4, 0.7])]:
        for i, v in enumerate(vals):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "family_id": fam,
                        "candidate_id": f"{fam}_{i}",
                        "success_rate": v + 0.001 * seed * (0 if fam == "flat" else 1),
                    }
                )
    ds = pd.DataFrame(rows)
    table = build_table(
        ds,
        oof=None,
        target="success_rate",
        model="extra_trees",
        feature_set="early+math+behavioral",
        se_multiplier=2.0,
        min_range=0.05,
    )
    by = table.set_index("family_id")
    assert bool(by.loc["flat", "identifiable"]) is False
    assert bool(by.loc["wide", "identifiable"]) is True
