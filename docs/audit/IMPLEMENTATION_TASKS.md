# Implementation Tasks (Post-Audit)

**Do not implement during the audit.** This is the engineering breakdown for when authorization is given. Prefer additive scripts/outputs; do not overwrite confirm artifacts.

---

## Task T0 — Provenance freeze

- **Create:** `experiments/archives/model_comparison_confirm_<date>/` (copy)  
- **Modify:** none required; optional note in `docs/DECISIONS.md`  
- **Outputs:** archived metrics + OOF + run_config  
- **Tests:** checksum or file-count assert in a small test optional  
- **Acceptance:** confirm path unchanged; archive exists  

---

## Task T1 — Identifiability analysis script

- **Create:** `scripts/analyze_within_family_identifiability.py`  
- **Use:** `experiments/phase1/dataset/...`, `experiments/model_comparison_confirm/oof_predictions.csv`  
- **Outputs:** `research_audit/artifacts/identifiability_table.csv`, short markdown summary under `results/` new folder  
- **Interfaces:** CLI `--dataset`, `--oof`, `--se-multiplier`  
- **Tests:** synthetic tiny dataframe with known flat family  
- **Deps:** pandas, numpy, scipy  
- **Acceptance:** reproduces distance_dense as non-identifiable; writes per-family Spearman  

---

## Task T2 — Metric policy helper

- **Modify:** `rsr/surrogate/model_comparison.py` `_within_family_spearman` **or** wrapper that filters families by range/SE without changing default confirm path  
- **Create:** config flag `within_family_identifiability: {enabled, min_range, min_range_over_se}`  
- **Outputs:** additional metrics columns `within_family_spearman_identifiable_mean`  
- **Tests:** constant family excluded; diverse family kept  
- **Acceptance:** confirm baseline numbers still reproducible with flag off  

---

## Task T3 — Fair benchmark runner

- **Create:** `scripts/run_fair_benchmark.py` + `configs/benchmark.yaml`  
- **Reuse:** `rsr/surrogate/model_comparison.py` LOFO loop  
- **Outputs:** `experiments/benchmark_<id>/` (metrics_summary, oof, paired_vs_baseline)  
- **Tests:** dry-run on 2 families smoke  
- **Acceptance:** includes mean, ridge, extra_trees; writes paired Δ CI  

---

## Task T4 — Pairwise ranking path

- **Modify/extend:** existing `pairwise_logistic` / `xgb_rank_pairwise` wiring in `model_comparison.py`  
- **Ensure:** pairs only within train families; soft-label weight by ΔY/SE  
- **Outputs:** benchmark rows comparable to ExtraTrees  
- **Tests:** held-out family IDs never appear in training pairs  
- **Acceptance:** leakage test green; metrics on identifiable families reported  

---

## Task T5 — Feature deltas / ratios

- **Create:** `rsr/features/ratio_features.py` or extend static/math extractors  
- **Modify:** `feature_builder.py` to optional `ratios` set  
- **Outputs:** new feature CSV; ablation via benchmark config  
- **Tests:** ratios finite; no target columns  
- **Acceptance:** LOFO ablation table produced  

---

## Task T6 — Adaptive seed planner (no auto-train)

- **Create:** `scripts/plan_adaptive_seeds.py`  
- **Outputs:** CSV of `(candidate_id, reason, recommended_extra_seeds)`  
- **Deps:** Phase-1 seed-level dataset  
- **Acceptance:** human-reviewable plan; **does not** launch PPO  

---

## Task T7 — Offline selection simulator

- **Create:** `scripts/simulate_surrogate_selection.py`  
- **Inputs:** OOF or held-out predictions + true y  
- **Outputs:** regret-vs-budget curves under policies  
- **Tests:** oracle policy regret → 0 at full budget  
- **Acceptance:** figure + CSV in `results/selection_sim_<id>/`  

---

## Task T8 — AL prototype (only after T3–T7)

- **Create:** `rsr/surrogate/active_learning.py`, `scripts/run_active_learning.py`  
- **Hard requirement:** new experiment run_id; never write into phase1 archive  
- **Acceptance:** documented acquisition; logging; stop rule  

---

## Suggested engineering order

T0 → T1 → T2 → T3 → T4/T5 → T6 → T7 → (authorize) T8 / PPO from T6
