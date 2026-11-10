# Gate 1 report — zero-PPO-cost feasibility

**Verdict: PASS (viable).** Pre-registered ExtraTrees + `math+behavioral` meets all four Phase 1 thresholds on 2D N=16 / k=5 single-group pools. **Stop here for human decision before Phase 3.**

Pre-registration: `docs/prereg/gate1.md` (commit `71f537b`) before any Gate 1 results.

## Data availability (VERIFY)

| Campaign | Status |
| --- | --- |
| 2D phase2 features (108) | **Used** — `experiments/phase2/features` |
| CrowdNav v4 (72) | **MISSING** locally (`experiments/crowdnav_v4_full_150k/` absent). Not run. |
| CrowdNav v5 (24) | **MISSING** — skip (top-1/pairwise only if present). |
| IR-SIM v1_nav24 | Dataset/features present but features meta has **1** candidate — not usable for LOFO; skipped (no new IR-SIM jobs). |

Local `experiments/crowdnav/` has 108 rows and is **not** treated as v4.

## Primary confirmatory result (2D)

Run: ExtraTrees × `math+behavioral` × success_rate, `n_repeats=3` → `experiments/gate1_zero_cost/` (~816 s).

| Metric | Value |
| --- | --- |
| Pooled Spearman (mean over repeats) | 0.812 |
| Within-family Spearman (mean) | ~0.57 |
| Within-family top-3 hit | ~0.69 |
| Pool recall@5 (N=16, 1 group, 2000 draws) | **0.829** |
| Median regret@5 | **0.000** |
| Δrecall vs random mean CI lo | **0.488** (>0) |
| Savings vs train-all (feature_steps=0, k=5) | **0.6875** |

### Threshold checklist

| Criterion | Result |
| --- | --- |
| recall@5 ≥ 0.80 | **pass** |
| median regret@5 ≤ 0.03 | **pass** |
| beat random (CI excludes 0) | **pass** |
| save ≥ 50% full-run steps | **pass** |

Baselines on same feature set (`experiments/gate1_zero_cost_baselines/`): Ridge pooled Spearman ~0.67 / within ~0.44; Mean ranking undefined (constant fold preds) — ExtraTrees remains the reference.

## Secondary / exploratory (labeled)

Full secondary grid at `n_repeats=3` exceeds the 10-minute agent job limit. Ran:

1. **Quick** (`n_repeats=1`) grid → `experiments/gate1_secondary_exploratory/` (exploratory).
2. Locked confirm OOF pool-sim for ExtraTrees `early+math+behavioral` as the costly comparator: recall@5≈**0.892**, savings **0.4475** (fails 50% savings because of 12k early features).

Exploratory quick pool-sim recall@5 (single-group) — **not** confirmatory:

| feature_set | recall@5 | feature steps | savings@k=5 |
| --- | --- | --- | --- |
| math+behavioral (confirm) | 0.829 | 0 | 0.688 |
| behavioral | 0.701 | 0 | 0.688 |
| math | 0.544 | 0 | 0.688 |
| static | 0.397 | 0 | 0.688 |
| early (quick) | 0.683 | 12k | 0.448 |
| early+math (quick) | 0.636 | 12k | 0.448 |
| early+math+behavioral (quick) | 0.667 | 12k | 0.448 |
| early+math+behavioral (locked confirm) | 0.892 | 12k | 0.448 |

Frontier figure: `results/gate1_zero_cost/figures/cost_accuracy_frontier.png`.

## Successive halving (offline)

VERIFY checkpoints in `features_early.csv`:

- Requested: 5k, 12k, 25k  
- **Available on 2D: 5k, 12k only** (no `early_t25000_success_frac`)

Family-as-pool SH using `early_t*_success_frac`:

| keep | recall_mean | med regret | mean steps/cand |
| --- | --- | --- | --- |
| 1/2 | 0.583 | 0.000 | 18 950 |
| 1/3 | 0.500 | 0.004 | 13 888 |

SH underperforms zero-cost ExtraTrees triage on recall and spends more steps than feature_steps=0 + k=5 full runs (~15 625 steps/cand average cost).

## Plan B (not required for primary, retained)

If a future env fails zero-cost: **Tier 0 sanity filter → 5k-step early features → SH**. On this 2D gate, Plan B is unnecessary for the primary claim.

## Commands to reproduce

```bash
# already committed pre-reg
python scripts/run_model_comparison.py --config configs/gate1.yaml --run-id gate1_zero_cost \
  --output-dir experiments/gate1_zero_cost --results-dir results/gate1_zero_cost \
  --models extra_trees --feature-sets math+behavioral --force

python scripts/run_gate1_analysis.py --oof experiments/gate1_zero_cost/oof_predictions_combined.csv --force
```

User-launched full secondary (optional, long):

```bash
python scripts/run_model_comparison.py --config configs/gate1.yaml --run-id gate1_secondary_full \
  --output-dir experiments/gate1_secondary_full --results-dir results/gate1_secondary_full --force
```

## Open issues

1. CrowdNav v4/v5 feature trees not on disk — Gate 1 external validity for CrowdNav **untested**.
2. IR-SIM features incomplete (n=1).
3. Agent split long grid: primary confirmatory vs quick secondary (documented).
4. Default branch unify (`main` vs `master`) still pending from revised Phase 0.

---

## GATE 1 — awaiting user decision

Is zero-PPO-cost ExtraTrees + `math+behavioral` accepted as the operating triage feature set for Phase 3+, or do you want Plan B / more CrowdNav data first?
