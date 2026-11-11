# Gate 2 report — lineage-grouped 2D triage viability

**Verdict: FAIL**

Pre-registration: `docs/prereg/gate2.md` (committed before this analysis).  
Analysis: `scripts/run_gate2_analysis.py` → `experiments/gate2_2d/`, `results/gate2_2d/`.

## Plain statements

- On Phase-6 live 2D labels (3 runs), ExtraTrees + `early+math+behavioral` with LOLO + phase1_108 prior does **not** meet the frozen recall bar at N=16, k=5.
- **Threshold 1 fail:** mean `recall_at_k` = **0.560** (need ≥ 0.80); lineage-bootstrap CI lo = 0.30.
- **Threshold 3 fail:** Δrecall vs random mean CI **includes 0** (lo = −0.075, hi = 0.675), even though the point Δ ≈ +0.26.
- **Threshold 2 pass:** median `regret_at_k` = **0.000** (≤ 0.03).
- **Threshold 4 pass:** formula `savings_frac` = **0.4475** (≥ 0.40) at early=12k / full=50k, k=5, N=16.
- **Grouping:** **48** labeled lineages (≥ 20) → recall-level claims are allowed to be tested; not INCONCLUSIVE on sample size.
- Strict within-lineage pools at N=16 are **impossible** (max lineage size = 15). Primary pools expand across lineages (same rule as Gate 1 family expansion).

**Allowed claim:** on this campaign, the surrogate beats random on point-estimate pool recall, with near-zero median regret and early-feature step savings at the k=5 formula.  
**Not allowed:** any recall≥0.80 claim, or that the surrogate statistically beats random under lineage bootstrap.

**Stop for human decision.** Do not start CrowdNav++ / IR-SIM / Gate 3 from this report without explicit go-ahead.

## Data

| Run | Records | Labeled | Wall (from run_summary) |
| --- | ---: | ---: | ---: |
| `phase6_2d_r1b` | 48 | 34 | ~3015 s |
| `phase6_2d_r2` | 48 | 34 | ~3069 s |
| `phase6_2d_r3` | 48 | 34 | ~3056 s |
| **Pooled labeled** | — | **102** | — |

- Feature set: `early+math+behavioral` (locked).
- OOF protocol: leave-one-**lineage**-out; train fold = other lineages + phase1_108 prior blend (matches online adapt).
- Pool protocol: N=16, k=5, `n_groups_requested=1` with `expand_groups=True`, 2000 draws; CI via lineage bootstrap (`ci_mode=family` in simulator with `family_id` ← lineage_id), `n_boot=1000`.

## Threshold checklist (frozen)

| # | Criterion | Result | Pass? |
| --- | --- | ---: | --- |
| 1 | Mean recall@5 ≥ 0.80 | **0.560** (CI 0.30–1.00) | **No** |
| 2 | Median regret@5 ≤ 0.03 | **0.000** | **Yes** |
| 3 | Δrecall vs random CI excludes 0 | mean Δ **0.257**; CI **[−0.075, 0.675]** | **No** |
| 4 | savings_frac ≥ 0.40 (early cost) | **0.4475** | **Yes** |
| — | ≥ 20 lineages for CI | **48** | **Yes** |

Baselines (same pools): random recall mean **0.295**; oracle **1.0**.

Operating selector `k_frac=0.5` (k=8) savings formula = **0.26** — report only; not used for pass/fail.

## Secondary: generation replay

Gen-0 warm-up used mode `off` → live `selection.score` is null; no gen-0 score replay.  
Labeled-only replay on gen1/2 (exploratory, selection bias): mixed 0/1 recall@k across runs (see `experiments/gate2_2d/generation_replay.csv`). Not confirmatory.

## Artifacts

- `experiments/gate2_2d/oof_predictions.csv`
- `experiments/gate2_2d/pool_summary.csv` / `results/gate2_2d/pool_summary.csv`
- `experiments/gate2_2d/gate2_verdict.json`
- `experiments/gate2_2d/lineage_census.json`

## Honesty / next options (for human decision only)

- FAIL is on the **pre-registered** bar, not a software crash.
- Possible follow-ups (none started): more diverse LLM rewards (less local-corpus collapse), different k for exploratory ops, or Gate-3 scope change — all require a new prereg if confirmatory.
