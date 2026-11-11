"""Phase 6d — corrected ablations + counterfactual steps-to-target (EXPLORATORY).

No new PPO. Does not change Gate 2 FAIL or frozen thresholds.
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

from reward_triage.gate2_harness import (  # noqa: E402
    RUN_IDS,
    dedupe_by_code_hash,
    feature_columns,
    load_all_records,
    lolo_oof,
    mean_success,
    seed_se,
    records_to_frame,
)
from rsr.evaluation.pool_simulator import draw_pool, evaluate_pool  # noqa: E402
from rsr.evaluation.triage_metrics import break_ties  # noqa: E402

N_POOL = 16
K = 5
FULL_STEPS = 50_000
N_SEEDS = 3


def _slice_oof(oof: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "gen0":
        return oof[oof["generation"] == 0].copy()
    if slice_name == "plateau":
        return oof[oof["generation"] >= 1].copy()
    return oof.copy()


def pool_eval_with_composition(
    oof: pd.DataFrame,
    *,
    n_pools: int,
    seed: int,
    slice_name: str,
) -> dict:
    """Surrogate + random on the SAME pools; track gen0/plateau share."""
    if len(oof) < N_POOL:
        return {
            "EXPLORATORY": True,
            "slice": slice_name,
            "error": "too_few_rows",
            "n": int(len(oof)),
        }
    # family_id must exist for draw_pool
    work = oof.copy()
    if "family_id" not in work.columns:
        work["family_id"] = work["code_hash"].astype(str)
    gen_map = work.set_index("candidate_id")["generation"].to_dict()

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_pools):
        pool = draw_pool(
            work,
            pool_size=N_POOL,
            n_groups=1,
            rng=rng,
            group_col="family_id",
            expand_groups=True,
        )
        if pool is None:
            continue
        child = np.random.default_rng(rng.integers(0, 2**31 - 1))
        # Fix tie-break once for fair surrogate/random compare
        ms = evaluate_pool(pool, k=K, selector="surrogate", rng=child)
        mr = evaluate_pool(
            pool,
            k=K,
            selector="random",
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
        )
        gens = [int(gen_map.get(cid, -1)) for cid in pool.candidate_ids]
        n_g0 = sum(1 for g in gens if g == 0)
        n_plat = sum(1 for g in gens if g >= 1)
        rows.append(
            {
                "recall_s": ms["recall_at_k"],
                "regret_s": ms["regret_at_k"],
                "recall_r": mr["recall_at_k"],
                "regret_r": mr["regret_at_k"],
                "share_gen0": n_g0 / N_POOL,
                "share_plateau": n_plat / N_POOL,
                "n_groups_realized": pool.n_groups,
            }
        )
    if not rows:
        return {"EXPLORATORY": True, "slice": slice_name, "error": "no_pools"}
    df = pd.DataFrame(rows)
    return {
        "EXPLORATORY": True,
        "slice": slice_name,
        "n_candidates": int(len(oof)),
        "n_pools": int(len(df)),
        "surrogate_recall_at_5": float(df["recall_s"].mean()),
        "surrogate_regret_at_5_median": float(df["regret_s"].median()),
        "random_recall_at_5_same_pools": float(df["recall_r"].mean()),
        "random_regret_at_5_median": float(df["regret_r"].median()),
        "delta_recall_s_minus_r": float(df["recall_s"].mean() - df["recall_r"].mean()),
        "mean_share_gen0": float(df["share_gen0"].mean()),
        "mean_share_plateau": float(df["share_plateau"].mean()),
        "mean_n_groups_realized": float(df["n_groups_realized"].mean()),
    }


def simple_uniform_random_baseline(n_unique: int, n_pools: int, seed: int) -> dict:
    """Uniform sample 16 from n_unique; random-k recall (analytic + Monte Carlo)."""
    analytic = K / N_POOL  # any fixed kept set of size k hits best with k/N
    rng = np.random.default_rng(seed)
    # Monte Carlo with distinct y to match "69 distinct candidates" story
    y = np.linspace(0, 1, n_unique)
    hits = []
    comps = []
    for _ in range(n_pools):
        idx = rng.choice(n_unique, size=N_POOL, replace=False)
        yp = y[idx]
        best = int(np.argmax(yp))
        kept = rng.choice(N_POOL, size=K, replace=False)
        hits.append(float(best in set(kept.tolist())))
        comps.append({"n": N_POOL, "k": K})
    return {
        "EXPLORATORY": True,
        "n_unique": n_unique,
        "analytic_random_k_recall": float(analytic),
        "mc_random_k_recall": float(np.mean(hits)),
        "note": (
            "Uniform pools of N=16 from M distinct candidates: random-k recall = k/N "
            f"= {K}/{N_POOL} = {analytic:.4f}, independent of M when sampling without "
            "replacement then picking k uniformly."
        ),
    }


def eps_recall_compare(
    oof: pd.DataFrame,
    *,
    n_pools: int,
    seed: int,
    slice_name: str,
) -> dict:
    if len(oof) < N_POOL:
        return {"EXPLORATORY": True, "slice": slice_name, "error": "too_few"}
    work = oof.copy()
    if "family_id" not in work.columns:
        work["family_id"] = work["code_hash"].astype(str)
    se = pd.to_numeric(work.get("seed_se", 0.02), errors="coerce").fillna(0.02).to_numpy()
    work = work.assign(seed_se=se)
    rng = np.random.default_rng(seed)

    def _eps(y, kept, best_i, tol: float) -> float:
        return float(any(abs(float(y[i]) - float(y[best_i])) <= tol for i in kept))

    rows = []
    for _ in range(n_pools):
        pool = draw_pool(
            work,
            pool_size=N_POOL,
            n_groups=1,
            rng=rng,
            group_col="family_id",
            expand_groups=True,
        )
        if pool is None:
            continue
        # align SE to pool order via candidate ids
        se_map = work.set_index("candidate_id")["seed_se"].to_dict()
        se_p = np.asarray([float(se_map[c]) for c in pool.candidate_ids], dtype=float)
        child = np.random.default_rng(rng.integers(0, 2**31 - 1))
        best = int(break_ties(pool.y_true, rng=child, higher_is_better=True)[0])
        # surrogate top-k
        from rsr.evaluation.triage_metrics import select_topk_indices

        kept_s = select_topk_indices(pool.y_pred, K, rng=child, higher_is_better=True)
        kept_r = np.random.default_rng(rng.integers(0, 2**31 - 1)).choice(
            N_POOL, size=K, replace=False
        )
        tol_1se = float(max(se_p[best], 1e-6))
        tol_002 = 0.02
        rows.append(
            {
                "eps_s_1se": _eps(pool.y_true, kept_s, best, tol_1se),
                "eps_r_1se": _eps(pool.y_true, kept_r, best, tol_1se),
                "eps_s_002": _eps(pool.y_true, kept_s, best, tol_002),
                "eps_r_002": _eps(pool.y_true, kept_r, best, tol_002),
                "hard_s": float(best in set(kept_s.tolist())),
                "hard_r": float(best in set(kept_r.tolist())),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "EXPLORATORY": True,
        "slice": slice_name,
        "n_pools": int(len(df)),
        "eps_recall_1SE_surrogate": float(df["eps_s_1se"].mean()),
        "eps_recall_1SE_random": float(df["eps_r_1se"].mean()),
        "eps_recall_0p02_surrogate": float(df["eps_s_002"].mean()),
        "eps_recall_0p02_random": float(df["eps_r_002"].mean()),
        "hard_recall_surrogate": float(df["hard_s"].mean()),
        "hard_recall_random": float(df["hard_r"].mean()),
    }


def _ordered_labeled(records) -> list:
    """Stable order: run_id, generation, candidate_id."""
    labeled = [r for r in records if mean_success(r) is not None]
    return sorted(
        labeled,
        key=lambda r: (
            r.run_id,
            int(r.candidate.generation),
            r.candidate.candidate_id,
        ),
    )


def counterfactual_curves(records) -> dict:
    """Best-so-far vs cumulative PPO seed-steps under four policies (observed labels)."""
    ordered = _ordered_labeled(records)
    # Precompute per-record step cost and success / SE
    items = []
    for r in ordered:
        done = [lab for lab in r.labels if lab.status == "completed"]
        steps = int(sum(int(lab.train_steps) for lab in done))
        if steps <= 0:
            steps = N_SEEDS * FULL_STEPS
        items.append(
            {
                "run_id": r.run_id,
                "generation": int(r.candidate.generation),
                "code_hash": r.candidate.code_hash,
                "success": float(mean_success(r) or 0.0),
                "se": float(seed_se(r) or 0.02),
                "steps": steps,
                "decision": r.selection.decision,
            }
        )

    def _curve(policy: str) -> list[dict]:
        seen_hash: set[str] = set()
        best = -1.0
        best_se = 0.02
        cum = 0
        pts = []
        # Per-run plateau stop state
        run_best_after_gen: dict[str, dict] = {}
        stopped_runs: set[str] = set()

        for it in items:
            rid = it["run_id"]
            gen = it["generation"]
            if policy == "filter" and it["decision"] not in ("train", "audit"):
                continue
            if policy in ("cache", "plateau", "filter") or policy == "train_all":
                pass
            if policy == "plateau" and rid in stopped_runs:
                continue
            # plateau: after finishing gen g with no improve > 2SE, skip later gens
            if policy == "plateau" and gen >= 1:
                prev = run_best_after_gen.get(rid)
                if prev is not None and gen > prev["last_gen_closed"]:
                    # Check improvement from gen0 best to end of previous gen
                    if prev.get("stop", False):
                        stopped_runs.add(rid)
                        continue

            cost = it["steps"]
            if policy in ("cache", "plateau", "filter") and it["code_hash"] in seen_hash:
                cost = 0
            if policy == "train_all" and it["code_hash"] in seen_hash:
                # train-all still re-trains duplicates across runs in this counterfactual
                cost = it["steps"]
            if policy == "cache" or policy == "plateau" or policy == "filter":
                seen_hash.add(it["code_hash"])
            elif policy == "train_all":
                seen_hash.add(it["code_hash"])

            cum += cost
            if it["success"] > best:
                best = it["success"]
                best_se = it["se"]
            pts.append(
                {
                    "cum_steps": cum,
                    "best_success": best,
                    "generation": gen,
                    "run_id": rid,
                }
            )

            # Update plateau bookkeeping at generation boundaries
            if policy == "plateau":
                st = run_best_after_gen.setdefault(
                    rid,
                    {
                        "gen0_best": -1.0,
                        "gen0_se": 0.02,
                        "last_gen_closed": -1,
                        "best_at_gen_end": {},
                        "stop": False,
                    },
                )
                # track best within each gen for this run
                # When we see first item of new gen, close previous gen
                # Simpler: update gen-end best continuously; decide stop when gen increments
        # Second pass for plateau stop flags using per-run gen maxima
        if policy == "plateau":
            return _plateau_curve(items)
        return pts

    def _plateau_curve(items_list: list[dict]) -> list[dict]:
        pts = []
        cum = 0
        best = -1.0
        seen: set[str] = set()
        by_run: dict[str, list] = {}
        for it in items_list:
            by_run.setdefault(it["run_id"], []).append(it)
        for rid, run_items in by_run.items():
            gens = sorted({it["generation"] for it in run_items})
            gen0_best = max(
                (it["success"] for it in run_items if it["generation"] == 0),
                default=-1.0,
            )
            gen0_se = np.median(
                [it["se"] for it in run_items if it["generation"] == 0] or [0.02]
            )
            stop_after = None
            prev_best = gen0_best
            for g in gens:
                g_items = [it for it in run_items if it["generation"] == g]
                g_best = max(it["success"] for it in g_items)
                if stop_after is not None and g > stop_after:
                    break
                for it in g_items:
                    cost = 0 if it["code_hash"] in seen else it["steps"]
                    seen.add(it["code_hash"])
                    cum += cost
                    if it["success"] > best:
                        best = it["success"]
                    pts.append(
                        {
                            "cum_steps": cum,
                            "best_success": best,
                            "generation": g,
                            "run_id": rid,
                        }
                    )
                # After finishing generation g>=0, check improvement vs previous best
                if g >= 0:
                    improved = g_best > prev_best + 2.0 * float(gen0_se)
                    if g >= 1 and not improved:
                        stop_after = g  # stop further gens
                    prev_best = max(prev_best, g_best)
        return pts

    out = {}
    for name, key in (
        ("train_all", "train_all"),
        ("plus_label_cache", "cache"),
        ("plus_plateau_stop", "plateau"),
        ("plus_filter", "filter"),
    ):
        curve = _curve(key)
        out[name] = {
            "EXPLORATORY": True,
            "n_points": len(curve),
            "final_best_success": float(curve[-1]["best_success"]) if curve else None,
            "final_cum_steps": int(curve[-1]["cum_steps"]) if curve else None,
            "curve": curve[:: max(1, len(curve) // 40)] if curve else [],  # downsample
            "curve_full_len": len(curve),
        }
    out["caveats"] = [
        "All labels treated as observed; skipped candidates have no labels — train-all is incomplete vs a true train-all.",
        "Final picks need independent re-evaluation (not done here).",
        "Policies were not saved in these runs; future runs must save final policies.",
        "Plateau stop: halt further generations after one gen with no improvement beyond 2 SE of the running best (using gen0 median SE).",
        "Label cache: duplicate code_hash costs 0 steps and reuses the observed label.",
        "Filter: only selection.decision in {train, audit}.",
    ]
    out["EXPLORATORY"] = True
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pools", type=int, default=800)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / "experiments" / "gate2d"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty {out_dir}; pass --force")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_records(RUN_IDS)
    report: dict = {
        "EXPLORATORY": True,
        "gate2_verdict_unchanged": "FAIL",
        "push_status_item0": (
            "experiments/gate2b_diagnostics and experiments/gate2c_harness "
            "already on origin/main as of commit 7dc160a"
        ),
    }

    ablation = {}
    for fs in ("early+math+behavioral", "math+behavioral", "early"):
        print(f"ablation {fs}…")
        cols = feature_columns(fs)  # type: ignore[arg-type]
        df = records_to_frame(records, cols, labeled_only=True)
        deduped = dedupe_by_code_hash(df)
        oof = lolo_oof(
            deduped,
            cols,
            fold_col="code_hash",
            use_prior=False,
            feature_set=fs,  # type: ignore[arg-type]
            enforce_no_hash_leak=True,
        )
        oof.to_csv(out_dir / f"oof_{fs.replace('+', '_')}.csv", index=False)
        slice_metrics = {}
        for sl in ("gen0", "plateau"):
            sub = _slice_oof(oof, sl)
            print(f"  slice {sl}: n={len(sub)}")
            m = pool_eval_with_composition(
                sub, n_pools=args.n_pools, seed=40 + hash(fs) % 100, slice_name=sl
            )
            # Never claim on pooled gen0+plateau
            print(
                f"    s_recall={m.get('surrogate_recall_at_5')} "
                f"r_recall={m.get('random_recall_at_5_same_pools')} "
                f"share_g0={m.get('mean_share_gen0')} share_plat={m.get('mean_share_plateau')}"
            )
            slice_metrics[sl] = m
            e = eps_recall_compare(
                sub, n_pools=args.n_pools, seed=90, slice_name=sl
            )
            slice_metrics[f"{sl}_eps"] = e
            print(
                f"    eps1SE s/r={e.get('eps_recall_1SE_surrogate')}/{e.get('eps_recall_1SE_random')} "
                f"eps0.02 s/r={e.get('eps_recall_0p02_surrogate')}/{e.get('eps_recall_0p02_random')}"
            )
        ablation[fs] = {
            "EXPLORATORY": True,
            "n_deduped": int(len(deduped)),
            "slices": slice_metrics,
            "pooled_claim": "FORBIDDEN — gen0 and plateau reported separately only",
        }
    report["ablations_codehash"] = ablation

    # Gap explanation vs simple random
    n_unique = int(
        dedupe_by_code_hash(
            records_to_frame(
                records, feature_columns("early+math+behavioral"), labeled_only=True
            )
        ).shape[0]
    )
    report["random_gap_explanation"] = {
        "EXPLORATORY": True,
        "gate2c_pooled_surrogate_recall": 0.493,
        "simple_uniform": simple_uniform_random_baseline(
            n_unique, n_pools=args.n_pools, seed=7
        ),
        "composition_note": (
            "Gate 2c pooled recall 0.493 mixes gen0+plateau under code_hash folds with "
            "expand_groups pool draws (often multi-hash pools). Simple random-k on a "
            f"uniform N={N_POOL} pool has analytic recall k/N={K}/{N_POOL}=0.3125. "
            "The ~0.18 gap is surrogate skill vs chance on mixed pools — NOT evidence "
            "for a confirmatory claim (EXPLORATORY; leakage-fixed but still selection-biased "
            "on plateau). Gen0-only N=16 with 16 hashes is degenerate (share_gen0=1)."
        ),
    }

    print("counterfactual curves…")
    report["counterfactual_steps_to_target"] = counterfactual_curves(records)
    cf = report["counterfactual_steps_to_target"]
    for k in ("train_all", "plus_label_cache", "plus_plateau_stop", "plus_filter"):
        print(
            f"  {k}: best={cf[k]['final_best_success']} steps={cf[k]['final_cum_steps']}"
        )

    report["real_llm_pilot_plan"] = {
        "EXPLORATORY": True,
        "do_not_run_in_this_phase": True,
        "plan": [
            "One run only: N=16 × G=3 × 3 seeds, env=2D, run_id=phase6d_xai_pilot1.",
            "Provider xAI with verified model id (prereg: grok-4.3 — VERIFY before launch).",
            "Requires XAI_API_KEY; until then keep --provider local.",
            "Temperature ≥ 0.9 (higher than 0.7 campaign) for diversity.",
            "Mutation prompt: require novelty vs parents (explicit diversity constraint / reject near-duplicates by code_hash or mb NN).",
            "Use run_id-dependent seeds (default); enable label cache.",
            "Save final policies (model.zip already written under runs/; also snapshot best-of-run policy path in run_summary).",
            "Independent re-eval of final pick on held-out eval seeds before any claim.",
            "Pilot only with --max-jobs 1 first; full pilot needs explicit user go-ahead.",
            "New prereg required before treating pilot as confirmatory.",
        ],
    }

    (out_dir / "gate2d_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("wrote", out_dir / "gate2d_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
