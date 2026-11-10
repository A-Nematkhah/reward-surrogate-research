# Gate 1 report — zero-PPO-cost feasibility

**Verdict (addendum `gate1_addendum`): PASS with caveat.** ExtraTrees + `math+behavioral` still meets the four numeric thresholds under **family-bootstrap CIs** on multi-family N=16 / k=5 pools, but **only 12 families exist**, so the pre-registered “≥ 20 held-out groups” condition is **not met**. **Stop for human decision before Phase 3.**

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

**Correction (addendum):** the claim that SH always costs more than zero-cost triage was wrong for keep=1/3 (13 888 < 15 625). See addendum SH table on N=16 pools.

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

## Addendum (`run_id=gate1_addendum`, no new PPO)

Artifacts: `results/gate1_addendum/` · script: `scripts/run_gate1_addendum.py`.

### 1. How N=16 pools are built

Code path: `draw_pool` in `rsr/evaluation/pool_simulator.py`.

1. Choose `n_groups` families (default request 1).
2. If their union has fewer than N candidates (**2D families are size 8–10**), **`expand_groups=True` adds more families** until the union can fill N.
3. Sample **N distinct candidates without replacement** (`numpy.choice(..., replace=False)`).
4. **No duplicates** (asserted; 0/500 audit draws had duplicate IDs). For N=16 with `n_groups_requested=1`, realized groups are **always 2**.

`expand_groups=False` is the strict within-family mode (used for N=8 below).

**Why random recall@5 was ~0.283, not 5/16=0.3125:**  
For a unique true best and uniform keep-k without replacement, **E[recall@k]=k/N=0.3125**. The 0.283 figure was a Monte Carlo estimate (2000 pools, seed 123), not a different sampler. Re-estimates are ~0.30–0.32; the gap is sampling noise, not duplicate candidates.

### 2. Within-family vs multi-family (family-bootstrap CIs)

Only **12 families** exist → “≥ 20 held-out groups” is **not** satisfied. CIs resample families, then redraw pools.

| feature_set | regime | N | k | recall@5 (mean [fam CI]) | med regret [CI] | Δrecall vs random [CI] | savings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| math+behavioral | within-family | 8 | 2 | 0.650 [0.40, 0.90] | 0.0 [0, 0.005] | 0.41 [0.06, 0.70] | 0.750 |
| math+behavioral | within-family | 8 | 3 | 0.765 [0.55, 0.99] | 0.0 [0, 0] | 0.39 [0.06, 0.65] | 0.625 |
| math+behavioral | multi-family | 16 | 5 | **0.848 [0.70, 0.98]** | **0.0** | **0.53 [0.33, 0.72]** | **0.688** |
| early+math+behavioral (locked OOF) | within-family | 8 | 2 | 0.658 [0.35, 0.90] | 0.0 | 0.41 [0.05, 0.75] | 0.510 |
| early+math+behavioral (locked OOF) | within-family | 8 | 3 | 0.830 [0.56, 1.00] | 0.0 | 0.45 [0.15, 0.75] | 0.385 |
| early+math+behavioral (locked OOF) | multi-family | 16 | 5 | 0.894 [0.73, 1.00] | 0.0 | 0.57 [0.38, 0.78] | 0.448 |

Within-family N=8 / k=2 for math+behavioral **misses** recall≥0.80; k=3 is borderline on the point estimate (0.765) with a wide family CI.

### 3. Config gap: quick 0.667 vs locked 0.892

| run | n_repeats | quick | ExtraTrees grid | inner LOFO |
| --- | --- | --- | --- | --- |
| locked confirm | 3 | false | n_estimators=[250], max_depth=[null,8], min_samples_leaf=[2,4] | yes |
| gate1_zero_cost primary | 3 | false | **same ET grid** | yes |
| gate1_secondary_exploratory | **1** | **true** (hyperparameter_max_combos=2) | same list, truncated search | truncated |

The ~0.67 pool-sim recall was from the **quick** secondary OOF. The ~0.89 figure is pool-sim on **locked confirm** OOF. Primary `gate1_zero_cost` settings match the locked confirm for ExtraTrees / repeats / non-quick (see `results/gate1_addendum/config_gap_quick_vs_locked.json`).

### 4. SH fix (N=16, k=5 pools)

`early_t2000_success_frac` **exists**. Available checkpoints: **2000, 5000, 12000** (no 25k). SH on multi-family N=16 pools, final keep k=5:

| keep | recall@5 | med regret | steps/cand | vs zero-cost triage (15 625) |
| --- | --- | --- | --- | --- |
| 1/2 | 0.30 | 0.015 | 10 000 | **cheaper**, much worse recall |
| 1/3 | 0.16 | 0.033 | 6 375 | **cheaper**, much worse recall |

Previous family-as-pool paragraph incorrectly implied SH was always costlier than zero-cost triage; keep=1/3 was already cheaper on a per-cand basis. On N=16/k=5 pools, both SH schedules are cheaper than 15 625 steps/cand but **lose badly on recall** vs ExtraTrees math+behavioral (~0.85).

### 5. pytest

`pytest -m "not slow" -rs` → **128 passed, 3 deselected, 0 skipped**. Deselected (slow): `tests/test_ppo_trainer.py` (3 tests). No `SKIP` reasons with the current env (gymnasium present).

### Addendum verdict

| Check (math+behavioral, N=16, k=5, family CI) | Result |
| --- | --- |
| recall ≥ 0.80 | pass (0.848; CI lo 0.70 — lo dips below 0.80) |
| median regret ≤ 0.03 | pass |
| beat random (family CI excludes 0) | pass |
| save ≥ 50% | pass (68.75%) |
| ≥ 20 held-out groups | **FAIL (only 12 families)** |

**Still stopped for user decision:** accept zero-cost triage for Phase 3 despite the 12-family limit and the family-CI lower bound on recall (<0.80), wait for CrowdNav data, or require Plan B.
