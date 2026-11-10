# Gate 1 report — zero-PPO-cost feasibility

**Verdict (amendments `gate1_amendments`): CONDITIONAL: feasibility signal on 2D only, not statistically established.**

Thresholds were applied **exactly as proposed** in `docs/METRIC_SPEC_TRIAGE.md` (frozen at `docs/prereg/gate1_thresholds_frozen.md`, commit before amendment analyses). No threshold values were changed.

### Plain statements

- **Threshold 1** (recall@5 ≥ 0.80): met by the point estimate (**0.829** primary; amendments re-estimate ~0.858) but **not** by the family-bootstrap CI (**lo = 0.70**).
- Only **12 families** exist per environment, so **“≥ 20 held-out groups” cannot be met at family level**.
- Confirmatory testing moves to **Gate 2**, where groups are **lineages** (Phase 4 data).
- **Allowed claim:** on 2D under LOFO, zero-PPO-cost features rank better than random.
- **Not allowed:** any recall-level claim, or any claim about CrowdNav++ or IR-SIM.

**Stop for human decision before Phase 3.** Do not start Phase 3 from this report.

Pre-registration: `docs/prereg/gate1.md` (commit `71f537b`) before any Gate 1 results.  
Threshold freeze: `docs/prereg/gate1_thresholds_frozen.md` (before `gate1_amendments` analyses).

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

Pool regime for N=16: **2-family pool (n_groups=1 expanded to 2 families)** — file names such as `summary_single_group.csv` are kept for script compatibility.

| Metric | Value |
| --- | --- |
| Pooled Spearman (mean over repeats) | 0.812 |
| Within-family Spearman (mean) | ~0.57 |
| Within-family top-3 hit | ~0.69 |
| Pool recall@5 (N=16, 2-family pool, 2000 draws) | **0.829** (point); family CI lo **0.70** |
| Median regret@5 | **0.000** |
| Δrecall vs random mean CI lo | **0.488** (>0) |
| Savings vs train-all (feature_steps=0, k=5) | **0.6875** |

### Threshold checklist (applied as proposed; not statistically established)

| Criterion | Point estimate | Family-bootstrap / grouping |
| --- | --- | --- |
| recall@5 ≥ 0.80 | met (0.829) | **not met** (CI lo = 0.70) |
| median regret@5 ≤ 0.03 | met | met |
| beat random (CI excludes 0) | met | met |
| save ≥ 50% full-run steps | met (68.75%) | met |
| ≥ 20 held-out groups | — | **FAIL (12 families)** |

Baselines on same feature set (`experiments/gate1_zero_cost_baselines/`): Ridge pooled Spearman ~0.67 / within ~0.44; Mean ranking undefined (constant fold preds) — ExtraTrees remains the reference.

## Secondary / exploratory (labeled)

Full secondary grid at `n_repeats=3` exceeds the 10-minute agent job limit. Ran:

1. **Quick** (`n_repeats=1`) grid → `experiments/gate1_secondary_exploratory/` (exploratory).
2. Locked confirm OOF pool-sim for ExtraTrees `early+math+behavioral` as the costly comparator: recall@5≈**0.892**, savings **0.4475** (fails 50% savings because of 12k early features).

Exploratory quick pool-sim recall@5 (**2-family pool (n_groups=1 expanded to 2 families)**) — **not** confirmatory:

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

## Successive halving (offline) — see amendments for fair comparison

VERIFY checkpoints in `features_early.csv`:

- Requested: 5k, 12k, 25k  
- **Available on 2D: 2k, 5k, 12k** (no `early_t25000_success_frac`)

**Do not compare SH recall to k=5 triage when survivor counts differ.** Under keep 1/2 and keep 1/3 on N=16 pools, final survivors are **2** and **1**, not 5. Fair baselines use the same final count (and equal total PPO cost). See amendments §5.

## Plan B (not required for primary, retained)

If a future env fails zero-cost: **Tier 0 sanity filter → 5k-step early features → SH**. On this 2D gate, Plan B is unnecessary for the primary (conditional) claim.

## Commands to reproduce

```bash
# already committed pre-reg + threshold freeze
python scripts/run_model_comparison.py --config configs/gate1.yaml --run-id gate1_zero_cost \
  --output-dir experiments/gate1_zero_cost --results-dir results/gate1_zero_cost \
  --models extra_trees --feature-sets math+behavioral --force

python scripts/run_gate1_analysis.py --oof experiments/gate1_zero_cost/oof_predictions_combined.csv --force
python scripts/run_gate1_amendments.py --force
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
4. Default branch unify (`main` vs `master`) still pending — **user decision required** (see end of amendments).

---

## Addendum (`run_id=gate1_addendum`, no new PPO)

Artifacts: `results/gate1_addendum/` · script: `scripts/run_gate1_addendum.py`.

### 1. How N=16 pools are built

Code path: `draw_pool` in `rsr/evaluation/pool_simulator.py`.

1. Choose `n_groups` families (default request 1).
2. If their union has fewer than N candidates (**2D families are size 8–10**), **`expand_groups=True` adds more families** until the union can fill N.
3. Sample **N distinct candidates without replacement** (`numpy.choice(..., replace=False)`).
4. **No duplicates** (asserted; 0/500 audit draws had duplicate IDs). For N=16 with `n_groups_requested=1`, realized groups are **always 2** → label **2-family pool (n_groups=1 expanded to 2 families)**.

`expand_groups=False` is the strict within-family mode (used for N=8 below).

**Why random recall@5 was ~0.283, not 5/16=0.3125:**  
For a unique true best and uniform keep-k without replacement, **E[recall@k]=k/N=0.3125**. The 0.283 figure was a Monte Carlo estimate (2000 pools, seed 123), not a different sampler. Re-estimates are ~0.30–0.32; the gap is sampling noise, not duplicate candidates.

### 2. Within-family vs multi-family (family-bootstrap CIs)

Only **12 families** exist → “≥ 20 held-out groups” is **not** satisfied. CIs resample families, then redraw pools.

| feature_set | regime | N | k | recall (mean [fam CI]) | med regret [CI] | Δrecall vs random [CI] | savings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| math+behavioral | within-family | 8 | 2 | 0.650 [0.40, 0.90] | 0.0 [0, 0.005] | 0.41 [0.06, 0.70] | 0.750 |
| math+behavioral | within-family | 8 | 3 | 0.765 [0.55, 0.99] | 0.0 [0, 0] | 0.39 [0.06, 0.65] | 0.625 |
| math+behavioral | 2-family pool (n_groups=1 expanded to 2 families) | 16 | 5 | **0.848 [0.70, 0.98]** | **0.0** | **0.53 [0.33, 0.72]** | **0.688** |
| early+math+behavioral (locked OOF) | within-family | 8 | 2 | 0.658 [0.35, 0.90] | 0.0 | 0.41 [0.05, 0.75] | 0.510 |
| early+math+behavioral (locked OOF) | within-family | 8 | 3 | 0.830 [0.56, 1.00] | 0.0 | 0.45 [0.15, 0.75] | 0.385 |
| early+math+behavioral (locked OOF) | 2-family pool (n_groups=1 expanded to 2 families) | 16 | 5 | 0.894 [0.73, 1.00] | 0.0 | 0.57 [0.38, 0.78] | 0.448 |

Within-family N=8 / k=2 for math+behavioral **misses** recall≥0.80; k=3 is borderline on the point estimate (0.765) with a wide family CI.

### 3. Config gap: quick 0.667 vs locked 0.892

| run | n_repeats | quick | ExtraTrees grid | inner LOFO |
| --- | --- | --- | --- | --- |
| locked confirm | 3 | false | n_estimators=[250], max_depth=[null,8], min_samples_leaf=[2,4] | yes |
| gate1_zero_cost primary | 3 | false | **same ET grid** | yes |
| gate1_secondary_exploratory | **1** | **true** (hyperparameter_max_combos=2) | same list, truncated search | truncated |

The ~0.67 pool-sim recall was from the **quick** secondary OOF. The ~0.89 figure is pool-sim on **locked confirm** OOF. Primary `gate1_zero_cost` settings match the locked confirm for ExtraTrees / repeats / non-quick (see `results/gate1_addendum/config_gap_quick_vs_locked.json`).

### 4. SH cost note (superseded by amendments §5 for fairness)

`early_t2000_success_frac` **exists**. Available checkpoints: **2000, 5000, 12000** (no 25k). On N=16 pools, keep 1/2 and 1/3 are cheaper per candidate than zero-cost triage at k=5 (15 625 steps/cand) but **final survivor counts are 2 and 1**, so recall must not be compared to triage@k=5.

### 5. pytest (addendum)

`pytest -m "not slow" -rs` → **128 passed, 3 deselected, 0 skipped**. Deselected (slow): `tests/test_ppo_trainer.py` (3 tests).

### Addendum checklist (informational; superseded by CONDITIONAL verdict)

| Check (math+behavioral, N=16, k=5, family CI) | Result |
| --- | --- |
| recall ≥ 0.80 | point pass; CI lo 0.70 — **not established** |
| median regret ≤ 0.03 | pass |
| beat random (family CI excludes 0) | pass |
| save ≥ 50% | pass (68.75%) |
| ≥ 20 held-out groups | **FAIL (only 12 families)** |

---

## Amendments (`run_id=gate1_amendments`, no new PPO)

Artifacts: `results/gate1_amendments/` · script: `scripts/run_gate1_amendments.py`.  
Settings: `n_pools=600`, `n_boot=200`, family-level bootstrap (resample 12 families, then pools). Existing OOF only.

### 1. Verdict wording

See top of this report: **CONDITIONAL: feasibility signal on 2D only, not statistically established.** Allowed vs not-allowed claims listed there. Thresholds applied as proposed (frozen).

### 2. Labels

N=16 with `n_groups_requested=1` is labeled **2-family pool (n_groups=1 expanded to 2 families)** in tables and headers. Filenames such as `summary_single_group.csv` remain for compatibility.

### 3. Within-family numbers (N=8, ExtraTrees OOF)

| feature_set | k | recall@k | random MC | random theory | mean regret | median regret | false-reject | recall CI | mean regret CI | median regret CI | false-reject CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math+behavioral (0 steps) | 2 | 0.640 | 0.265 | 0.250 | 0.0083 | 0.000 | 0.360 | [0.40, 0.90] | [0.0007, 0.021] | [0, 0.002] | [0.10, 0.60] |
| math+behavioral (0 steps) | 3 | 0.750 | 0.400 | 0.375 | 0.0035 | 0.000 | 0.250 | [0.55, 0.95] | [0.0003, 0.011] | [0, 0] | [0.05, 0.45] |
| early+math+behavioral (12k) | 2 | 0.667 | 0.265 | 0.250 | 0.0074 | 0.000 | 0.333 | [0.45, 0.88] | [0.0017, 0.016] | [0, 0.008] | [0.13, 0.55] |
| early+math+behavioral (12k) | 3 | 0.822 | 0.400 | 0.375 | 0.0029 | 0.000 | 0.178 | [0.68, 1.00] | [0, 0.006] | [0, 0] | [0, 0.33] |

Source: `results/gate1_amendments/within_family_N8.csv`.

### 4. Paired feature-set comparison (family-bootstrap Δrecall)

`early+math+behavioral` − `math+behavioral` (cost: **12k vs 0** feature steps per candidate):

| regime | N | k | recall emb | recall mb | Δrecall mean | Δrecall fam CI | CI excludes 0? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2-family pool (n_groups=1 expanded to 2 families) | 16 | 5 | 0.897 | 0.858 | +0.038 | **[-0.125, 0.226]** | no |
| within-family | 8 | 3 | 0.817 | 0.742 | +0.075 | **[-0.200, 0.301]** | no |

Source: `results/gate1_amendments/paired_feature_set_delta.csv`. Point estimate favors early features; family CI includes 0 under the 12-family limit.

### 5. Successive halving fairness (N=16)

Checkpoints `[2000, 5000, 12000]`. Final survivor count is the post-halving set (**not** forced to k=5).

| keep | final survivors | recall SH | random same count | random equal total PPO cost | steps/cand |
| --- | --- | --- | --- | --- | --- |
| 1/2 | **2** | 0.255 | 0.133 | 0.185 | 10 000 |
| 1/3 | **1** | 0.152 | 0.075 | 0.122 | 6 375 |

Source: `results/gate1_amendments/sh_fairness_N16.csv`. SH beats random at matched survivor count and at equal cost, but survivor counts (**2** / **1**) are not comparable to triage keep-k=5.

### 6. Freeze

`docs/prereg/gate1_thresholds_frozen.md` lists the four thresholds from `METRIC_SPEC_TRIAGE.md` and notes Gate 2 uses lineages with ≥ 20 held-out lineages. Committed **before** steps 3–5.

### Amendments verdict (unchanged)

**CONDITIONAL: feasibility signal on 2D only, not statistically established.**  
Allowed: zero-PPO-cost features beat random on 2D LOFO.  
Not allowed: recall-level claims; CrowdNav++ / IR-SIM claims.

**Stopped — wait for user.** Default branch merge (`main` vs `master`) needs explicit confirmation; no push.
