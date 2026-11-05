# Phase 3.1 — Feature extras (quick)

**Date:** 2026-11-05  
**Model:** ExtraTrees  
**Mode:** `--quick` (not confirmatory)  
**Artifacts:** `experiments/benchmark_phase3_features/`, prune run in `experiments/benchmark_phase3_prune/`

## Built extras (0 PPO)

| File | Cols | Source |
|------|-----:|--------|
| `features_ratios.csv` | 51 | weight ratios / L1 shares |
| `features_early_deltas.csv` | 12 | existing early dynamics subset |

Script: `scripts/build_phase3_feature_extras.py`

## Identifiable within-Spearman (primary)

| feature_set | pooled Spearman | within (all) | within (id) | top3 (id) |
|-------------|----------------:|-------------:|------------:|----------:|
| **early+math+behavioral** (ops) | 0.725 | 0.415 | **0.467** | 0.727 |
| early+math+ratios | 0.751 | 0.436 | 0.450 | 0.727 |
| early+math+behavioral+ratios | 0.733 | 0.396 | 0.446 | 0.727 |
| early+math | 0.676 | 0.421 | 0.434 | 0.727 |
| early+math+ratios+early_deltas | 0.698 | 0.404 | 0.399 | 0.545 |
| early+math+behavioral + WF-prune | 0.725 | 0.415 | 0.467 | 0.727 |

## Interpretation (honest)

1. Under **quick** settings, no feature variant beats the operating set on **identifiable within-Spearman**.
2. `ratios` helps **pooled** Spearman slightly but not the primary within-id metric.
3. Explicit `early_deltas` stack **hurts** (dynamics already inside `early`).
4. Within-family variance prune (`1e-8`) did not change predictions — likely because Spearman top-k already ignores flat-within behavioral noise.

## Decision

- Keep **early+math+behavioral** as operating features for now.
- Ratios remain available for full (non-quick) re-check / ranking Stage 3.2.
- Do **not** promote ratios or early_deltas on the basis of this quick screen.
- Next: Phase 3.2 pairwise ranking contrast vs ExtraTrees (pre-registered), still 0 PPO.
