# Research Audit Package

**Date:** 2026-11-05  
**Mode:** Evidence-only audit. No research code modified. No new PPO.

## Documents

1. [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)  
2. [REPOSITORY_ARCHITECTURE_AUDIT.md](REPOSITORY_ARCHITECTURE_AUDIT.md)  
3. [STATISTICAL_VALIDITY_AUDIT.md](STATISTICAL_VALIDITY_AUDIT.md)  
4. [WITHIN_FAMILY_DIAGNOSIS.md](WITHIN_FAMILY_DIAGNOSIS.md)  
5. [MODEL_AND_FEATURE_RESEARCH.md](MODEL_AND_FEATURE_RESEARCH.md)  
6. [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)  
7. [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md)  
8. [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)  
9. [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md)  
10. [RESEARCH_RISKS_AND_LIMITATIONS.md](RESEARCH_RISKS_AND_LIMITATIONS.md)  
11. [EXECUTION_PLAN.md](EXECUTION_PLAN.md) — prioritized fix plan  
12. [METRIC_POLICY.md](METRIC_POLICY.md) — adopted within-family estimand  

## Phase 0–2 outputs

- Archive: `experiments/archives/model_comparison_confirm_2026-11-05/`
- Diagnostic: `results/diagnostics_within_family/` (`scripts/analyze_within_family_identifiability.py`)
- Fair benchmark (quick): `experiments/benchmark_phase2_base/` + `results/benchmark_phase2_base/REPORT.md` (`scripts/run_fair_benchmark.py`)
- Phase 3.1 features (quick): `experiments/benchmark_phase3_features/` + `results/benchmark_phase3_features/REPORT.md`
- Phase 3.2 ranking (quick): `experiments/benchmark_phase3_ranking/` + `results/benchmark_phase3_ranking/REPORT.md` → keep ExtraTrees
- Phase 4 seed plan (no PPO): `results/adaptive_seed_plan/` — 36 runs on 2 weak families
- Phase 5 selection sim: `results/selection_sim_phase5/` — ET top-k finds global best by k=5

## Primary verified baseline (2D confirm)

ExtraTrees / `early+math+behavioral` / `success_rate` — mean of 3 repeats:

- Pooled Spearman ≈ **0.935**
- Within-family Spearman ≈ **0.648**
- Top-3 hit ≈ **0.833**

Source: `experiments/model_comparison_confirm/metrics_summary.csv`
