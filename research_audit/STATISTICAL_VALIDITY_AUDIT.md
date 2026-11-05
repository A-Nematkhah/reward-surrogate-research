# Statistical & Experimental Validity Audit

## 5.1 Independent sample size — correct units

| Quantity | 2D Phase-1 archive | Role |
|----------|-------------------:|------|
| Reward candidates | **108** | Surrogate sample size after aggregation |
| Reward families | **12** | LOFO fold unit; CI unit |
| PPO runs (seeds) | **324** completed (CSV may be 322 post-filter) | Replicates **within** candidate — **not** independent rewards |
| Model comparison repeats | **3** | RNG repeats of LOFO+HPO — **not** new labels |

**Rules:**
- Never treat 324 as n for Spearman CI on rewards.
- Family-level bootstrap (n=12) is the honest uncertainty for within-family means.
- Pairwise comparisons among candidates inside 12 families are **dependent**; do not treat hundreds of pairs as independent n.

CrowdNav v1: 72 candidates × 12 families × 3 seeds — same logic.

---

## 5.2 LOFO implementation verdict: **PASS with caveats**

Evidence: `src/surrogate/model_comparison.py`.

| Check | Status |
|-------|--------|
| Outer fold = one family held out (`LeaveOneGroupOut` on `family_id`) | PASS |
| Seeds aggregated to one row/candidate before LOFO (`seed_aggregation: mean`) | PASS |
| Imputation / constant drop / feature select / scaling / target transform fit on train only | PASS |
| Inner HPO = LOFO on training families; score = within-family Spearman | PASS |
| Family one-hot disabled | PASS |
| Target columns blocked from X | PASS |

### Leakage / validity pathways

| Pathway | Status |
|---------|--------|
| Final eval metrics as features | Blocked |
| HPO seeing test family | Blocked |
| **Early features from same PPO run as label** | **Present** — shared-run dependence; disclose; not classical fold leakage |
| Probe trajectories using success labels | Guarded (fixed probes) |
| `nan_to_num` on full X before CV | Soft / minor |
| Overwriting confirm after peeking at same 12 folds repeatedly | **Process risk** (researcher df) |

---

## 5.3 Metric validity

### Pooled Spearman
- SciPy Spearman on all OOF candidates.
- Dominated by between-family structure when family means differ a lot (**supported:** ~64% SS between families on 2D seed-mean SR).
- Useful for coarse screening; **insufficient** as primary claim for within-family selection.

### Within-family Spearman (macro)
- Per-family Spearman, then mean/median/std over families with `min_size` (outer default **4**).
- Equal family weight — good.
- Families with constant labels/preds dropped (`association_status`), not scored as 0.
- **Problem:** near-constant but non-identical labels still produce unstable correlations (`distance_dense`).

### Within-family Top-3 hit (macro)
- Whether true best index ∈ predicted top-3 within family.
- Tie-breaking via `nanargmax` / stable argsort — **arbitrary among ties**.
- Still a strong practical metric when ties are rare.

### Top-10 selected mean / regret
- **Pooled over all candidates**, not within-family.
- Must not be marketed as within-family selection quality.
- Regret = oracle top-10 mean − selected top-10 mean.

### MAE / RMSE
- Absolute scale; less aligned with ranking use-case but useful for calibration sanity.

### Uncertainty / calibration
- Ensemble/uncertainty paths exist in broader codebase history; confirm primary ExtraTrees claim is point-prediction + selection metrics.
- Do not claim calibrated uncertainty without dedicated evaluation.

---

## 5.4 Statistical uncertainty

| Method | In repo? | Appropriate? |
|--------|----------|--------------|
| Family bootstrap (n_boot=2000) | Yes | **Yes** for within-family mean / paired diffs |
| Candidate bootstrap ignoring family | Avoid as primary | Inflates n |
| Paired per-family Spearman diffs vs ExtraTrees | Yes | Correct design; **low power** (CI wide; often includes 0) |
| Permutation test (shuffle labels within train) | Not primary | Useful Stage-1 null |
| Multiple-comparison correction | Weak / incomplete | Exploratory grids need explicit “exploratory” label |

**Power reality:** With 12 families, detecting a +0.05 within-Spearman lift is hard. Prefer pre-registered single contrast + family bootstrap; avoid claiming significance from large model grids.

---

## 5.5 Target noise (seed analysis, 2D)

Computed on Phase-1 seed-mean aggregation:

| Statistic | Value |
|-----------|------:|
| Median within-candidate seed std (SR) | ≈ 0.0225 |
| Median within-family SR range | ≈ 0.214 |
| Between-family mean SR span | ≈ 0.631 |
| Fraction within-family pairs with \|Δ\| < SE | ≈ 0.25 |
| Fraction with \|Δ\| < 2·SE | ≈ 0.44 |

**Interpretation:**
- Many close pairs are **not stably ordered** with 3 seeds.
- Families with large ranges (`efficiency`, `energy_aware`, `balanced`) are more identifiable.
- Adaptive replication is justified for close calls; blanket 10 seeds × 108 is wasteful.

**Recommended estimand for labels:**
- Primary: \(\bar Y_i = \mathrm{mean}_s Y_{i,s}\)
- Keep \(\hat\sigma_i\) for uncertainty-aware selection / weighting
- Optional: soft pairwise labels \(P(i\succ j)\) from seed bootstrap

---

## Required corrections (methodology, not code changes in this audit)

1. **Cite verified confirm metrics**, not the unverified brief table.  
2. **Split reporting:** pooled vs within-family vs top-k (and state pooled top-10 clearly).  
3. **Identifiability gate** before averaging within-Spearman.  
4. **Disclose early-feature cost and coupling** in all claims.  
5. **Archive confirm outputs** before re-running pipelines.  
6. Mark exploratory grid maxima as **biased**; primary = pre-registered contrast.  
7. Treat n=12 for inference; pairs are dependent.
