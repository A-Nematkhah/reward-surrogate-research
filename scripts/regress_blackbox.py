#!/usr/bin/env python3
"""Slow regression: black-box features vs parametric math/behavioral pipeline.

Requires local datasets. Marked for manual / --slow use. No new PPO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.executor import run_candidate_on_probes
from reward_triage.features import extract_blackbox_features
from reward_triage.probes import ensure_default_synthetic_probes, load_probe_npz
from reward_triage.spec_to_code import COMPONENT_NAMES, spec_to_code
from reward_triage.tier0 import apply_tier0


def _load_specs(candidates_csv: Path) -> list[dict]:
    df = pd.read_csv(candidates_csv)
    specs = []
    for _, row in df.iterrows():
        weights = {}
        scales = {}
        for name in COMPONENT_NAMES:
            for key in (f"weight_{name}", f"w_{name}", name):
                if key in df.columns:
                    weights[name] = float(row[key])
                    break
            for key in (f"scale_{name}", f"s_{name}"):
                if key in df.columns:
                    scales[name] = float(row[key])
                    break
        specs.append(
            {
                "candidate_id": str(row.get("candidate_id", row.get("id"))),
                "family_id": str(row.get("family_id", row.get("family", "unknown"))),
                "weights": weights,
                "scales": scales,
            }
        )
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "data" / "navigation_2d" / "phase1_108" / "candidates_seed_agg.csv",
    )
    p.add_argument(
        "--math-features",
        type=Path,
        default=ROOT
        / "data"
        / "navigation_2d"
        / "phase1_108"
        / "features"
        / "features_math.csv",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase3_blackbox_regress")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-candidates", type=int, default=0, help="0 = all")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(
            "dry-run: would load specs, run sandbox features, compare Spearman to math "
            f"features, write {args.out_dir}"
        )
        return 0

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        print(f"Refusing non-empty output dir without --force: {args.out_dir}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe_paths = ensure_default_synthetic_probes()
    batch = load_probe_npz(probe_paths["navigation_2d"])

    if not args.candidates.exists():
        report = {
            "status": "skipped",
            "reason": f"missing candidates csv: {args.candidates}",
            "note": "Regression needs local datasets; unit tests cover sandbox without them.",
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    specs = _load_specs(args.candidates)
    if args.max_candidates:
        specs = specs[: args.max_candidates]

    rows = []
    for spec in specs:
        code = spec_to_code(spec)
        exec_result = run_candidate_on_probes(code, batch.transitions, timeout_s=10.0)
        report = apply_tier0(exec_result)
        feats = extract_blackbox_features(
            exec_result.rewards,
            batch.transitions,
            exec_result.components,
            code=code,
        )
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "family_id": spec["family_id"],
                "tier0_status": report.status,
                **feats,
            }
        )

    bb = pd.DataFrame(rows)
    bb.to_csv(args.out_dir / "blackbox_features.csv", index=False)

    corr_table = []
    documented_misses: list[str] = []
    meta_path = args.math_features.parent / "features_meta.csv"
    if args.math_features.exists() and meta_path.exists():
        math_df = pd.read_csv(args.math_features)
        meta_df = pd.read_csv(meta_path)
        id_col = "candidate_id" if "candidate_id" in meta_df.columns else None
        if id_col is None:
            documented_misses.append(f"no candidate_id in {meta_path}")
            merged = None
        else:
            math_df = pd.concat([meta_df[[id_col]], math_df], axis=1)
            merged = bb.merge(math_df, on=id_col, how="inner", suffixes=("", "_math"))
        # Align bb probe stats to math magnitude columns when present.
        pairs = [
            ("bb_stats_mean", "math_magnitude_mean"),
            ("bb_stats_std", "math_variance"),
            ("bb_stats_scale", "math_magnitude_max"),
        ]
        if merged is not None:
            for left, right in pairs:
                if left not in merged.columns or right not in merged.columns:
                    documented_misses.append(f"missing columns for pair {left} vs {right}")
                    continue
                r, _ = spearmanr(merged[left], merged[right])
                corr_table.append(
                    {
                        "bb_feature": left,
                        "ref_feature": right,
                        "spearman": float(r) if r is not None and np.isfinite(r) else float("nan"),
                        "pass_ge_0_95": bool(r is not None and np.isfinite(r) and r >= 0.95),
                    }
                )
        if corr_table and not any(c["pass_ge_0_95"] for c in corr_table):
            documented_misses.append(
                "Black-box probe stats vs parametric math features did not reach "
                "Spearman >= 0.95 on this synthetic probe set; documented. "
                "Primary LOFO parity gate needs full OOF outside CI."
            )
    else:
        documented_misses.append(f"math/meta features missing under {args.math_features.parent}")

    summary = {
        "n_candidates": len(bb),
        "tier0_ok_frac": float((bb["tier0_status"] == "ok").mean()) if len(bb) else 0.0,
        "feature_corr": corr_table,
        "documented_misses": documented_misses,
        "lofo_note": (
            "LOFO within-family Spearman delta vs parametric requires OOF labels; "
            "run with full local confirm artifacts outside CI."
        ),
    }
    (args.out_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
