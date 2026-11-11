# Gate 2b — EXPLORATORY diagnostics (no PPO; Gate 2 verdict unchanged)

**Status: EXPLORATORY only.** Does **not** change the Gate 2 **FAIL** or frozen thresholds.  
Artifacts: `experiments/gate2b_diagnostics/` (`diagnostics.json`).  
Script: `scripts/run_gate2b_diagnostics.py`.

All numbers below are tagged **EXPLORATORY**.

## 1. Prior OFF (campaign labels only)

Same lineage-LOLO + pool protocol as Gate 2 (N=16, k=5), but train folds **exclude** phase1_108 prior.

| Metric | With prior (Gate 2 locked) | Prior OFF (exploratory) |
| --- | ---: | ---: |
| recall@5 mean | 0.560 | **0.613** |
| median regret@5 | 0.000 | **0.000** |
| savings restart | 0.448 | **0.448** |
| savings resume | — | **0.523** |

Prior was not the main failure mode; recall stays well below 0.80.

## 2. Feature ablations (prior OFF)

| Feature set | recall@5 | med regret@5 | savings restart | savings resume |
| --- | ---: | ---: | ---: | ---: |
| `early+math+behavioral` | 0.613 | 0.000 | 0.448 (`e=0.24`) | 0.523 |
| `math+behavioral` (zero early cost) | 0.618 | 0.000 | **0.688** (`e=0`) | **0.688** |
| `early` only | 0.518 | 0.000 | 0.448 | 0.523 |

Zero-cost mb matches early+mb on ranking here; early-alone is weaker. **Caveat:** lineage folds still leak duplicate `code_hash` across runs (see 6c).

## 3. Label-noise ceiling

Median seed SE ≈ **0.018**. Over 1000 pool draws (resample labels within SE):

| Quantity | Value |
| --- | ---: |
| P(observed pool best = latent best) | 0.566 |
| Perfect-scorer recall@5 under noise | **0.980** |
| eps-recall@5 (kept within max(1.5·SE, 0.03) of latent best) | **1.000** |

Noise alone does **not** explain Gate 2’s 0.56 recall — a perfect noisy-label ranker still hits ~0.98.

## 4. Corpus diversity audit

| Check | Result |
| --- | ---: |
| Records / unique `code_hash` | 144 / **108** (25% duplicate-hash rate across runs) |
| Distinct LLM families | **10** |
| mb near-duplicate rate (NN &lt; 1% median pairwise dist) | **0.86** |
| Gen0 label std / range | ~0.235 / ~0.785 (identical across r1b/r2/r3 — corpus collapse) |
| Gen1–2 label std | ~0.02–0.04 (plateau near ~0.79–0.80) |
| Share of labels near ceiling (q95−0.02) | ~0.19 |

## 5. Selection-bias check

| Slice | recall@5 | note |
| --- | ---: | --- |
| Gen0 only (all labeled) | **1.00** | Degenerate / optimistic: many pools are near-replicas of the same 16 gen0 codes across runs |
| Gens 1–2 with IPW (`w=1/p`) | **0.489** | Active-filter plateau; only 6 lineages |

## 6. Cost accounting

**Host loop currently RESTARTs** kept candidates (full PPO from scratch). Early features are collected **during** that same full run — there is **no** resume-from-early-checkpoint path for the final policy.

| Formula | N=16, k=5, e=0.24 |
| --- | ---: |
| Restart: `1 − (e + k/N)` | **0.448** |
| Resume: `(1 − k/N)(1 − e)` | **0.523** |

Realized whole-run (G=3, each of r1b/r2/r3): 34/48 labeled → 102 vs 144 seed-jobs → **~29%** full-run savings (early cost embedded in those jobs). Applying restart/resume formulas with k=34, N=48 gives ~0.05 / ~0.22 (not comparable to the k=5 threshold table).

## Stop

Gate 2 FAIL stands. Diagnostics only. Next confirmatory work requires a new prereg after harness fixes (Phase 6c).
