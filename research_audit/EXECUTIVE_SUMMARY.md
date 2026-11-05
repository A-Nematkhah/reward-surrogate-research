# Executive Summary — Research Audit

**Audit date:** 2026-11-05  
**Scope:** Toy 2D `RewardNavigation2D-v0` (primary). CrowdNav is secondary and **not** assumed to transfer.  
**Rule:** No code changes; no new PPO; evidence from repository artifacts only.

---

## Current scientific status

The project has a **credible Phase-1 → Phase-2 pipeline** for predicting PPO navigation performance from reward features under **leave-one-family-out (LOFO)**. The locked confirmatory result on the 108-candidate 2D archive is scientifically stronger than the numbers quoted in the audit brief.

### Verified locked 2D confirm (ExtraTrees / `early+math+behavioral` / `success_rate`)

Source: `experiments/model_comparison_confirm/metrics_summary.csv` (mean of 3 repeats):

| Metric | Verified mean |
|--------|--------------:|
| Pooled Spearman | **0.935** |
| Within-family Spearman mean | **0.648** |
| Within-family Spearman std | **0.428** |
| Within-family Top-3 hit | **0.833** |
| Top-10 selected mean (pooled) | **0.810** |
| Top-10 regret (pooled) | **0.009** |
| MAE | **0.067** |
| Feature PPO cost accounting | 12,000 steps |

**Claimed brief numbers (0.8448 / 0.5751 / …) do not match any artifact exactly.** Closest archived campaign is CrowdNav `crowdnav_v1_50k` (~0.860 / 0.533). Do not treat the brief table as the 2D baseline.

README / `docs/DECISIONS.md` (~0.65 / ~0.94 / ~0.83) **match** the 2D confirm, not the brief.

---

## Main strengths

1. **Correct scientific unit of evaluation:** LOFO over reward families, with nested train-only preprocessing/HPO (implementation audit: PASS with caveats).
2. **Task metrics ≠ training return:** success/collision/timeout kept separate from PPO return — aligned with Booth et al. (AAAI 2023) warnings about reward–metric mismatch.
3. **Explicit anti-leakage guards:** target columns blocked from features; family one-hot disabled under LOFO; tests cover LOFO/leakage.
4. **Confirmatory protocol:** pre-registration language, 3 repeats, family bootstrap, selection-bias guard, ExtraTrees preserved as reference.
5. **Practical triage signal exists:** within-family Top-3 hit 0.83 and low pooled top-10 regret on 2D confirm.

---

## Main weaknesses

1. **Within-family ranking is heterogeneous, not uniformly weak.** Mean 0.65 with std 0.43 is driven by a few hard/near-constant families (esp. `distance_dense` ≈ −0.64) while many families score 0.85–0.95. Treating “within-family failure” as a single disease misdirects methods.
2. **~64% of label variance is between-family** (ANOVA-style on seed-mean SR). Pooled Spearman can look excellent while within-family remains the hard estimand.
3. **Seed noise vs small within-family gaps:** median seed std ≈ 0.023; ~25% of within-family pairs have \|ΔSR\| < SE; ~44% within 2·SE. Some rankings are not identifiable with 3 seeds.
4. **Early features couple to the labeled PPO run** (checkpoints of the same training). Not classical CV leakage, but optimistic for “zero-PPO” claims and for AL cost accounting.
5. **Documentation / artifact drift:** config targets 48 candidates while archive is 108; exploratory `experiments/model_comparison/` missing; Phase-2 report missing; AL not implemented; quality pipeline can overwrite confirm dirs.
6. **Statistical power:** n_families = 12. Paired model comparisons have wide CIs; exploratory grid winners are selection-biased.

---

## Most important findings

1. **Falsified claim framing:** The brief’s ExtraTrees table is not the locked 2D result; verify provenance before designing improvements around it.
2. **H1 partially supported:** between-family signal dominates variance and pooled correlation.
3. **H7 strongly supported:** within-Spearman is unstable for near-constant families (`distance_dense` range ≈ 0.017).
4. **H4 partially supported:** seed noise explains many close pairs; does not explain all ranking failures.
5. **H6 weakly supported so far:** rankers already demoted in DECISIONS; ranking is a hypothesis, not a proven fix.
6. **H2 mixed:** behavioral features show very low within-family variation (≈80% near-zero within-std); early features retain more within variation — feature design for intra-family deltas is a real lever.
7. **Objective B (AL) is not yet an empirical result** — only gate prose exists.

---

## Highest-priority recommendations

| Priority | Action | Why |
|----------|--------|-----|
| P0 | Freeze provenance: cite **verified 2D confirm** as baseline; never overwrite confirm without a new run_id | Scientific integrity |
| P1 | **Diagnostic report only (no PPO):** per-family range, seed-SE, identifiable ranking flag, OOF Spearman | Fixes estimand before models |
| P2 | Redefine primary within-family metrics **conditional on identifiable families** + report flat families separately | Stops metric gaming |
| P3 | Low-cost: pairwise Δφ ranking + Ridge/ET ablations on **same LOFO folds**, no new PPO | Tests H6 cheaply |
| P4 | Adaptive extra seeds only for close-call / scientifically critical families | Addresses H4 efficiently |
| P5 | Defer CrowdNav claims and deep nets until 2D within-family estimand is clean | Avoids confounding |

**Strongest scientifically defensible direction:**  
*Surrogate-assisted reward triage under family shift, with explicit separation of between-family screening vs within-family ranking among identifiable candidates — evaluated by selection regret and PPO budget, not pooled Spearman alone.*

---

## Five most important findings (compact)

1. Brief metrics ≠ locked 2D confirm (0.935/0.648).  
2. Within-family weakness is **heterogeneous** (few bad/flat families).  
3. Between-family variance ≈ 64% of total.  
4. LOFO implementation is largely correct; early↔label coupling is the main interpretability caveat.  
5. AL / Objective B is unimplemented; ranking models are unproven at this n.

## Five highest-value experiments

1. Identifiability + per-family error analysis (0 PPO).  
2. Fair LOFO benchmark: mean / Ridge / ET / best tree / pairwise ranker (0 PPO).  
3. Feature ablation emphasizing within-family signal (early deltas, ratios) (0–low PPO).  
4. Adaptive seed replication for close pairs (targeted PPO).  
5. Frozen-surrogate selection simulation → true PPO cost–regret curve (controlled PPO).

## Recommended first experiment

**Stage 1 diagnostic:** build the per-family identifiability table from existing Phase-1 + confirm OOF (see `EXPERIMENTAL_ROADMAP.md` Exp-D1). No PPO. Acceptance: written policy for which families enter the primary within-family mean.

## Realistic execution order

Stage 0 provenance → Stage 1 diagnostics → Stage 2 fair benchmark + ranking/features → Stage 3 adaptive seeds → Stage 4 selection/AL simulation → Stage 5 robustness (held-out pool / env caution).

## Expected computational budget (next phase)

| Stage | Wall / PPO |
|-------|------------|
| 0–2 | CPU hours; **0 new PPO** |
| 3 | ~10–30 extra seed runs (not full 108×3) if adaptive |
| 4 | Simulation first; then ≤1 small PPO validation batch |
| 5 | Optional; large if CrowdNav — **authorize separately** |
