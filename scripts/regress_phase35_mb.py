#!/usr/bin/env python3
"""Phase 3.5: mb features via triage_candidate must match Gate-1 math+behavioral."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.pipeline import triage_candidate  # noqa: E402
from reward_triage.spec_to_code import spec_to_code, specs_from_rows  # noqa: E402
from rsr.features.probe_trajectories import load_probe_set  # noqa: E402
from rsr.surrogate.model_comparison import _within_family_spearman  # noqa: E402


def _serialize(tr: dict) -> dict:
    out = {}
    for k, v in tr.items():
        if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "radius"):
            out[k] = [float(v.x), float(v.y), float(v.radius)]
        elif isinstance(v, list) and v and hasattr(v[0], "x"):
            out[k] = [[float(o.x), float(o.y), float(o.radius)] for o in v]
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _episodes_from_probe_set(episodes):
    flat = []
    grouped = []
    for ep in episodes:
        trs = []
        for tr in ep.transitions:
            item = dict(tr)
            item["policy"] = ep.policy_name
            item["episode_index"] = ep.episode_index
            item["progress"] = float(item["prev_goal_distance"]) - float(item["goal_distance"])
            if item.get("min_obstacle_clearance") is not None:
                item["clearance"] = item["min_obstacle_clearance"]
            ser = _serialize(item)
            trs.append(ser)
            flat.append(ser)
        grouped.append((ep.policy_name, trs))
    return flat, grouped


def _lofo_within(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    preds = np.zeros(len(y), dtype=float)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=250,
                    random_state=0,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    return float(
        _within_family_spearman(y, preds, groups, min_size=2)["within_family_spearman_mean"]
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "data" / "navigation_2d" / "phase1_108" / "candidates_seed_agg.csv",
    )
    p.add_argument(
        "--probes",
        type=Path,
        default=ROOT / "experiments" / "phase2" / "probe_trajectories.npz",
    )
    p.add_argument(
        "--math",
        type=Path,
        default=ROOT
        / "data"
        / "navigation_2d"
        / "phase1_108"
        / "features"
        / "features_math.csv",
    )
    p.add_argument(
        "--behavioral",
        type=Path,
        default=ROOT
        / "data"
        / "navigation_2d"
        / "phase1_108"
        / "features"
        / "features_behavioral.csv",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=ROOT
        / "data"
        / "navigation_2d"
        / "phase1_108"
        / "features"
        / "features_meta.csv",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase3_5_mb")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(f"dry-run: would compare triage mb vs Gate1 tables -> {args.out_dir}")
        return 0
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        print(f"Refusing non-empty {args.out_dir} without --force")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidates)
    specs = specs_from_rows(df.to_dict(orient="records"))
    if args.max_candidates:
        specs = specs[: args.max_candidates]

    meta = pd.read_csv(args.meta)
    gate = pd.concat(
        [meta, pd.read_csv(args.math), pd.read_csv(args.behavioral)], axis=1
    )
    gate = gate.set_index("candidate_id")

    probe_eps = load_probe_set(args.probes)
    flat, grouped = _episodes_from_probe_set(probe_eps)

    rows = []
    max_col_diff = 0.0
    worst_col = None
    n_fail_cols = 0
    for spec in specs:
        code = spec_to_code(spec)
        report, feats, _, _ = triage_candidate(
            code,
            flat,
            weights=spec["weights"],
            scales=spec.get("scales") or {},
            episodes=grouped,
            feature_groups=("mb",),
            in_process=True,
            timeout_s=60.0,
        )
        assert report.status in ("ok", "warn"), (spec["candidate_id"], report)
        assert feats is not None
        grow = gate.loc[spec["candidate_id"]]
        col_diffs = {}
        for col, val in feats.items():
            if col not in grow.index:
                continue
            ref = float(grow[col])
            d = abs(float(val) - ref)
            col_diffs[col] = d
            if d > max_col_diff:
                max_col_diff = d
                worst_col = col
            if d >= 1e-9:
                n_fail_cols += 1
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "family_id": spec["family_id"],
                "max_abs_diff": max(col_diffs.values()) if col_diffs else float("inf"),
                "n_compared": len(col_diffs),
                **{f"mb_{k}": v for k, v in feats.items()},
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_dir / "mb_features.csv", index=False)

    # LOFO on pipeline mb vs Gate-1 table
    y = df.set_index("candidate_id").loc[out_df["candidate_id"], "success_rate"].to_numpy(
        dtype=float
    )
    groups = out_df["family_id"].to_numpy()
    mb_cols = [c for c in out_df.columns if c.startswith("mb_")]
    X_pipe = out_df[mb_cols].to_numpy(dtype=float)
    gate_aligned = gate.loc[out_df["candidate_id"]]
    # Intersection of columns
    feat_names = [c[len("mb_") :] for c in mb_cols]
    common = [c for c in feat_names if c in gate_aligned.columns]
    X_gate = gate_aligned[common].to_numpy(dtype=float)
    X_pipe_common = out_df[[f"mb_{c}" for c in common]].to_numpy(dtype=float)

    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        within_pipe = float("nan")
        within_gate = float("nan")
        lofo_vs_table = float("nan")
        lofo_vs_reported = float("nan")
        lofo_pass = False
    else:
        within_pipe = _lofo_within(X_pipe_common, y, groups)
        within_gate = _lofo_within(X_gate, y, groups)
        # Gate1 reported ~0.57 (ExtraTrees math+behavioral mean over repeats)
        lofo_vs_table = abs(within_pipe - within_gate)
        lofo_vs_reported = abs(within_pipe - 0.57)
        # Prefer matching the Gate-1 feature-table LOFO; also accept ~0.57 ± 0.01
        lofo_pass = (lofo_vs_table <= 0.01) or (lofo_vs_reported <= 0.01)
    gate1_reported = 0.57

    feature_pass = n_fail_cols == 0 and max_col_diff < 1e-9

    summary = {
        "status": "pass" if feature_pass and lofo_pass else "fail",
        "n_candidates": len(out_df),
        "feature_match": {
            "pass": feature_pass,
            "max_abs_diff": max_col_diff,
            "worst_column": worst_col,
            "n_column_failures": n_fail_cols,
            "n_common_columns": len(common),
        },
        "lofo_within_family": {
            "pass": lofo_pass,
            "pipeline": within_pipe,
            "gate1_table": within_gate,
            "gate1_reported_approx": gate1_reported,
            "abs_delta_vs_table": lofo_vs_table,
            "abs_delta_vs_reported": lofo_vs_reported,
            "tolerance": 0.01,
        },
    }
    (args.out_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
