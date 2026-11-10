#!/usr/bin/env python3
"""Rebuild pooled / within-family Spearman from locked OOF predictions (<60 lines)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "experiments/model_comparison_confirm/oof_predictions.csv"
SUMMARY = ROOT / "experiments/model_comparison_confirm/metrics_summary.csv"


def _spearman(y: pd.Series, p: pd.Series) -> float:
    if y.nunique() < 2 or p.nunique() < 2:
        return float("nan")
    return float(spearmanr(y, p).correlation)


def main() -> int:
    oof = pd.read_csv(OOF)
    rows = []
    for (model, fs, target, rep), g in oof.groupby(
        ["model", "feature_set", "target", "repeat"]
    ):
        pooled = _spearman(g["y_true"], g["y_pred"])
        within = [
            _spearman(fg["y_true"], fg["y_pred"])
            for _, fg in g.groupby("family_id")
            if len(fg) >= 2
        ]
        rows.append(
            {
                "model": model,
                "feature_set": fs,
                "target": target,
                "repeat": rep,
                "spearman": pooled,
                "within_family_spearman_mean": float(pd.Series(within).mean()),
            }
        )
    recon = (
        pd.DataFrame(rows)
        .groupby(["model", "feature_set", "target"], as_index=False)[
            ["spearman", "within_family_spearman_mean"]
        ]
        .mean()
    )
    summary = pd.read_csv(SUMMARY)
    mean = summary.groupby(["model", "feature_set", "target"], as_index=False)[
        ["spearman", "within_family_spearman_mean"]
    ].mean()
    merged = recon.merge(mean, on=["model", "feature_set", "target"], suffixes=("_oof", "_csv"))
    print(merged.to_string(index=False))
    for _, r in merged.iterrows():
        assert abs(r["spearman_oof"] - r["spearman_csv"]) < 1e-3
        assert abs(r["within_family_spearman_mean_oof"] - r["within_family_spearman_mean_csv"]) < 1e-3
    et = merged[
        (merged["model"] == "extra_trees")
        & (merged["feature_set"] == "early+math+behavioral")
    ].iloc[0]
    print(
        f"ExtraTrees early+math+behavioral: pooled={et['spearman_oof']:.3f} "
        f"within={et['within_family_spearman_mean_oof']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
