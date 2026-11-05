# Phase 3.2 — Pairwise ranking vs ExtraTrees

**Date:** 2026-11-05  
**Pre-registered contrast:** `pairwise_logistic` (within-family Δφ, `min_abs_delta=0.02`) vs `extra_trees`  
**Features:** `early+math+behavioral` → `success_rate`  
**Mode:** `--quick` (same folds/protocol for both; not the locked confirm ET numbers)  
**Primary metric:** identifiable within-family Spearman  
**Inferential unit:** family (n=11 identifiable); bootstrap CI over families  

## Macro (quick)

| model | pooled Spearman | within (all) | within (id) | top3 (id) |
|-------|----------------:|-------------:|------------:|----------:|
| pairwise_logistic | 0.815 | 0.548 | **0.543** | 0.818 |
| extra_trees | 0.727 | 0.419 | 0.471 | 0.636 |

## Family-paired Δ (challenger − baseline) on identifiable families

| Statistic | Value |
|-----------|------:|
| Mean Δ | +0.072 |
| Median Δ | 0.000 |
| 90% family bootstrap CI | **[−0.134, +0.303]** |
| CI excludes 0? | **No** |
| **Decision** | **`fail_to_reject_keep_extra_trees`** |

Artifacts: `family_paired_deltas.csv`, `ranking_contrast_summary.json`.

## Interpretation

1. Point estimates favor pairwise logistic under this quick screen.  
2. With only 11 identifiable families, uncertainty is large; CI includes zero.  
3. Per Stage-3 stopping rule → **do not replace ExtraTrees**; do not escalate to deeper rankers/NNs.  
4. Optional later: full (non-quick) re-run of the **same** contrast for a stronger claim — still not required to proceed to Phase 4.

## Code changes supporting this experiment

- `PairwiseLogisticRanker`: `within_family_only=True` (default), `min_abs_delta`  
- Config: `configs/benchmark_phase3_ranking.yaml`  
- Analysis: `scripts/analyze_ranking_contrast.py`
