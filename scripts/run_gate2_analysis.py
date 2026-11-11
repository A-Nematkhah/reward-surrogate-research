"""Gate 2 confirmatory analysis on Phase-6 2D live stores (lineage groups).

Pre-registered: docs/prereg/gate2.md
Outputs: experiments/gate2_2d/, results/gate2_2d/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.adapt import load_prior_xy  # noqa: E402
from reward_triage.feature_matrix import vectorize_record  # noqa: E402
from reward_triage.model import SurrogateModel  # noqa: E402
from reward_triage.store import LabelStore, StoreRecord  # noqa: E402
from rsr.evaluation.cost import triage_cost  # noqa: E402
from rsr.evaluation.pool_simulator import (  # noqa: E402
    evaluate_pool,
    simulate_pools,
)
from rsr.evaluation.triage_metrics import recall_at_k, regret_at_k  # noqa: E402

RUN_IDS = ("phase6_2d_r1b", "phase6_2d_r2", "phase6_2d_r3")
N_POOL = 16
K_THRESH = 5
FEATURE_STEPS = 12_000
FULL_STEPS = 50_000
M_MIN_HELD_OUT = 20


def _mean_success(rec: StoreRecord) -> float | None:
    done = [lab for lab in rec.labels if lab.status == "completed"]
    if not done:
        return None
    return float(sum(lab.success for lab in done) / len(done))


def load_labeled_table(
    run_ids: tuple[str, ...],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[StoreRecord]]:
    rows: list[dict] = []
    kept: list[StoreRecord] = []
    for run_id in run_ids:
        path = ROOT / "experiments" / run_id / "label_store.jsonl"
        for rec in LabelStore(path).load_all():
            y = _mean_success(rec)
            if y is None:
                continue
            x = vectorize_record(rec, feature_cols)
            rows.append(
                {
                    "candidate_id": rec.candidate.candidate_id,
                    "lineage_id": rec.candidate.lineage_id,
                    "run_id": run_id,
                    "generation": int(rec.candidate.generation),
                    "y_true": y,
                    "decision": rec.selection.decision,
                    "live_score": rec.selection.score,
                    **{f"f{i}": float(x[i]) for i in range(len(feature_cols))},
                }
            )
            kept.append(rec)
    df = pd.DataFrame(rows)
    return df, kept


def lolo_oof_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    use_prior: bool = True,
) -> pd.DataFrame:
    """Leave-one-lineage-out ExtraTrees preds; optional phase1_108 prior blend."""
    X = df[[f"f{i}" for i in range(len(feature_cols))]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=float)
    lineages = df["lineage_id"].to_numpy()
    y_pred = np.full(len(y), np.nan, dtype=float)

    X_prior = y_prior = None
    if use_prior:
        X_prior, y_prior = load_prior_xy(feature_cols, feature_set="early+math+behavioral")

    uniq = sorted(set(lineages.tolist()))
    for lin in uniq:
        te = lineages == lin
        tr = ~te
        if not np.any(tr):
            continue
        model = SurrogateModel(feature_cols, feature_set="early+math+behavioral")
        if use_prior and X_prior is not None and y_prior is not None:
            model.adapt_with_prior(
                X[tr],
                y[tr],
                X_prior=X_prior,
                y_prior=y_prior,
                dataset_hash=f"lolo_{lin}",
            )
        else:
            model.fit(X[tr], y[tr], dataset_hash=f"lolo_{lin}")
        assert model.pipeline is not None
        y_pred[te] = model.pipeline.predict(X[te])

    out = df[
        ["candidate_id", "lineage_id", "run_id", "generation", "y_true"]
    ].copy()
    out = out.rename(columns={"lineage_id": "family_id"})  # pool_simulator group col
    out["y_pred"] = y_pred
    return out.dropna(subset=["y_pred"]).reset_index(drop=True)


def generation_replay(df: pd.DataFrame) -> pd.DataFrame:
    """Secondary: gen0 full pools (all labeled); later gens labeled-only exploratory."""
    rows: list[dict] = []
    for (run_id, gen), g in df.groupby(["run_id", "generation"]):
        g = g.dropna(subset=["live_score"])
        if len(g) < 2:
            continue
        y_true = g["y_true"].to_numpy(dtype=float)
        y_score = g["live_score"].to_numpy(dtype=float)
        n = len(g)
        for k in (5, max(1, int(round(0.5 * n)))):
            if k > n:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "generation": int(gen),
                    "n_labeled_in_gen": n,
                    "k": k,
                    "recall_at_k": recall_at_k(y_true, y_score, k=k),
                    "regret_at_k": regret_at_k(y_true, y_score, k=k),
                    "note": "gen0_full" if int(gen) == 0 else "labeled_only_exploratory",
                }
            )
    return pd.DataFrame(rows)


def apply_thresholds(summary_row: dict, n_lineages: int) -> dict:
    recall = float(summary_row["recall_at_k_mean"])
    regret_med = float(summary_row["regret_at_k_median"])
    delta_lo = float(summary_row.get("recall_minus_random_lo", float("nan")))
    savings = float(summary_row["savings_frac"])
    t1 = recall >= 0.80
    t2 = regret_med <= 0.03
    t3 = np.isfinite(delta_lo) and delta_lo > 0.0
    t4 = savings >= 0.40
    enough = n_lineages >= M_MIN_HELD_OUT
    if not enough:
        verdict = "INCONCLUSIVE"
    elif t1 and t2 and t3 and t4:
        verdict = "PASS"
    elif t1 and t2 and t4 and not t3:
        verdict = "CONDITIONAL"
    elif (not t1) or (not t2) or (not t4):
        verdict = "FAIL"
    else:
        verdict = "CONDITIONAL"
    return {
        "threshold_1_recall_ge_0_80": t1,
        "threshold_2_median_regret_le_0_03": t2,
        "threshold_3_delta_recall_ci_excludes_0": t3,
        "threshold_4_savings_ge_0_40": t4,
        "n_lineages": n_lineages,
        "ge_20_lineages": enough,
        "verdict": verdict,
        "recall_at_k_mean": recall,
        "regret_at_k_median": regret_med,
        "recall_minus_random_lo": delta_lo,
        "recall_minus_random_hi": float(
            summary_row.get("recall_minus_random_hi", float("nan"))
        ),
        "recall_minus_random_mean": float(
            summary_row.get("recall_minus_random_mean", float("nan"))
        ),
        "savings_frac": savings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pools", type=int, default=2000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--no-prior", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_exp = ROOT / "experiments" / "gate2_2d"
    out_res = ROOT / "results" / "gate2_2d"
    for d in (out_exp, out_res):
        if d.exists() and any(d.iterdir()) and not args.force:
            raise SystemExit(f"Refusing non-empty {d}; pass --force")
        d.mkdir(parents=True, exist_ok=True)

    cold = SurrogateModel.load(ROOT / "experiments" / "phase5_surrogate_2d")
    feature_cols = list(cold.feature_cols)

    labeled_df, _ = load_labeled_table(RUN_IDS, feature_cols)
    labeled_df.to_csv(out_exp / "labeled_candidates.csv", index=False)

    lineage_sizes = labeled_df.groupby("lineage_id").size()
    census = {
        "run_ids": list(RUN_IDS),
        "n_labeled": int(len(labeled_df)),
        "n_lineages": int(labeled_df["lineage_id"].nunique()),
        "lineage_size_min": int(lineage_sizes.min()),
        "lineage_size_max": int(lineage_sizes.max()),
        "n_lineages_size_ge_16": int((lineage_sizes >= N_POOL).sum()),
        "within_lineage_N16_possible": bool((lineage_sizes >= N_POOL).any()),
        "feature_set": "early+math+behavioral",
        "use_prior": not args.no_prior,
    }
    (out_exp / "lineage_census.json").write_text(
        json.dumps(census, indent=2), encoding="utf-8"
    )
    print("census:", json.dumps(census))

    print("LOLO OOF…")
    oof = lolo_oof_predictions(
        labeled_df, feature_cols, use_prior=not args.no_prior
    )
    oof.to_csv(out_exp / "oof_predictions.csv", index=False)
    print(f"  oof rows={len(oof)} lineages={oof['family_id'].nunique()}")

    print("pool sim N=16 k=5 + lineage bootstrap…")
    per_df, summary = simulate_pools(
        oof,
        pool_sizes=(N_POOL,),
        n_groups_list=(1,),
        ks=(K_THRESH,),
        n_pools=args.n_pools,
        selectors=("surrogate", "random", "oracle"),
        seed=args.seed,
        feature_steps_per_candidate=FEATURE_STEPS,
        full_run_steps=FULL_STEPS,
        n_boot=args.n_boot,
        expand_groups=True,
        ci_mode="family",
    )
    per_df.to_csv(out_exp / "pool_draws.csv", index=False)
    summary.to_csv(out_exp / "pool_summary.csv", index=False)
    summary.to_csv(out_res / "pool_summary.csv", index=False)

    sur = summary[
        (summary["selector"] == "surrogate")
        & (summary["N"] == N_POOL)
        & (summary["k"] == K_THRESH)
    ].iloc[0]
    # Point Δrecall from paired pool draws (bootstrap CI still from summary).
    base = per_df[
        (per_df["N"] == N_POOL)
        & (per_df["k"] == K_THRESH)
        & (per_df["n_groups_requested"] == 1)
    ]
    sur_p = base[base["selector"] == "surrogate"].set_index("pool_i")
    rnd_p = base[base["selector"] == "random"].set_index("pool_i")
    common = sur_p.index.intersection(rnd_p.index)
    point_delta = float(
        (
            sur_p.loc[common, "recall_at_k"] - rnd_p.loc[common, "recall_at_k"]
        ).mean()
    )
    row = sur.to_dict()
    if "recall_minus_random_mean" not in row or not np.isfinite(
        float(row.get("recall_minus_random_mean", float("nan")))
    ):
        row["recall_minus_random_mean"] = point_delta

    gate = apply_thresholds(row, n_lineages=int(census["n_lineages"]))
    # Override held-out flag: LOLO uses every lineage as a held-out fold; count = n_lineages.
    gate["held_out_lineages_for_ci"] = int(census["n_lineages"])
    gate["ci_mode"] = "lineage_bootstrap_via_family_ci_mode"
    gate["n_pools"] = int(sur["n_pools"])
    gate["n_boot"] = args.n_boot
    cost_k5 = triage_cost(
        N_POOL, K_THRESH, feature_steps_per_candidate=FEATURE_STEPS, full_run_steps=FULL_STEPS
    )
    cost_k8 = triage_cost(
        N_POOL, 8, feature_steps_per_candidate=FEATURE_STEPS, full_run_steps=FULL_STEPS
    )
    gate["operating_k8_savings_frac_report_only"] = cost_k8["savings_frac"]
    gate["formula_k5_savings_frac"] = cost_k5["savings_frac"]

    replay = generation_replay(labeled_df)
    replay.to_csv(out_exp / "generation_replay.csv", index=False)

    (out_exp / "gate2_verdict.json").write_text(
        json.dumps({"census": census, "gate": gate, "primary_summary_row": row}, indent=2),
        encoding="utf-8",
    )
    (out_res / "gate2_verdict.json").write_text(
        json.dumps({"census": census, "gate": gate}, indent=2), encoding="utf-8"
    )
    print("verdict:", gate["verdict"])
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
