# Fair benchmark (Phase 2) — quick run

**Date:** 2026-11-05  
**Config:** `configs/benchmark.yaml`  
**Command:** `python scripts/run_fair_benchmark.py --quick`  
**Output:** `experiments/benchmark_phase2_base/`  
**Note:** Quick mode (truncated HPO, `n_repeats=1`, `n_bootstrap=100`). **Not** a replacement for the locked confirm ExtraTrees numbers (~0.935 / 0.648). Use for scaffold + relative ordering only.

## Setup

| Item | Value |
|------|-------|
| Target | `success_rate` |
| Features | `early+math+behavioral` |
| Models | mean, ridge, extra_trees |
| Identifiability | enabled (`min_range=0.05`, `se_multiplier=2`) |
| n_families_identifiable | 11 |

## Results (repeat 0)

| model | pooled Spearman | within (all) | within (identifiable) | top3 (all) | top3 (id) | MAE |
|-------|----------------:|-------------:|----------------------:|-----------:|----------:|----:|
| extra_trees | 0.725 | 0.415 | 0.467 | 0.750 | 0.727 | 0.124 |
| ridge | 0.747 | 0.402 | 0.462 | 0.750 | 0.818 | 0.135 |
| mean | NaN | NaN | NaN | 0.417 | 0.455 | 0.257 |

## Takeaways

1. Pipeline writes identifiable metrics correctly without touching confirm.
2. Under **quick** settings, Ridge ≈ ExtraTrees on identifiable within-Spearman; mean is a weak floor.
3. For Phase 3 claims, re-run **without** `--quick` (and preferably `n_repeats: 3`) into a new `experiments/benchmark_phase2_full/` directory.
