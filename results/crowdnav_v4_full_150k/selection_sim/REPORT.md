# Surrogate Selection Simulation — CrowdNav v4@150k

**OOF:** ExtraTrees / `early+math+behavioral` / `success_rate`  
**Pool size:** 72 candidates (12 families × 6)  
**Aggregation:** mean over LOFO repeats 0, 1, 2  
**Oracle global best SR:** ≈ 0.808

## Regret vs budget (mean over repeats)

| k | oracle best | surrogate best | random best (mean) | regret vs oracle | P(oracle top-1 ∈ sur top-k) |
|--:|------------:|---------------:|-------------------:|-----------------:|----------------------------:|
| 1 | 0.808 | 0.606 | 0.429 | 0.202 | 0.00 |
| 3 | 0.808 | 0.617 | 0.606 | 0.192 | 0.00 |
| 5 | 0.808 | 0.641 | 0.663 | 0.167 | 0.00 |
| 10 | 0.808 | 0.765 | 0.721 | 0.044 | 0.00 |
| 15 | 0.808 | 0.765 | 0.744 | 0.044 | 0.00 |
| 20 | 0.808 | 0.794 | 0.758 | 0.015 | 0.67 |
| 30 | 0.808 | 0.808 | 0.776 | 0.000 | 1.00 |
| 50 | 0.808 | 0.808 | 0.794 | 0.000 | 1.00 |
| 72 | 0.808 | 0.808 | 0.808 | 0.000 | 1.00 |

CSVs: `regret_vs_budget_mean_over_repeats.csv`, `regret_vs_budget_all_repeats.csv`  
(Per-repeat raw also in `../selection_sim_r1/` for repeat 1.)

## Interpretation

- **k=1–5:** Surrogate alone is **not** a reliable “pick the best” tool. Random’s *best-of-k* can match or beat surrogate’s best at small k (selection-noise regime).
- **k=10:** Surrogate recovers SR ≈ **0.765** vs oracle 0.808 (**regret ≈ 0.044**), and **beats random’s expected best** (≈0.72). Useful as a **shortlist**, not a singleton pick. Oracle #1 is still usually outside the shortlist.
- **k≈20:** Regret ≈ 0.015; oracle top-1 enters the shortlist in ~2/3 of repeats.
- **k≈30:** Surrogate shortlist contains the global best in all repeats (regret 0).

## Practical recommendation (CrowdNav pool)

| Goal | Suggested budget |
|------|------------------|
| Cheap triage / beat random | **k = 10** (~14% of pool) |
| Near-oracle best | **k = 20–30** |
| Exact global best guaranteed in shortlist | **k ≥ 30** |

**Do not** claim “surrogate finds the unique best in one shot” on CrowdNav. Claim: *ExtraTrees OOF ranking yields a shortlist that dominates random under moderate evaluation budgets.*

## Contrast with locked 2D confirm

2D selection sim (confirm OOF) reached near-oracle at much smaller k (efficient budget often ≤10). CrowdNav needs a larger fraction of the pool because (a) harder LOFO ranks, (b) one broken family (`time_pressure`), (c) denser mid-tier SR mass. Narrative should stay environment-specific.

## Files

- `regret_vs_budget.csv` — repeat 0 detail from script default run  
- `regret_vs_budget_mean_over_repeats.csv` — table above  
- `regret_vs_budget_all_repeats.csv` — per-repeat rows  
