# Experimental Roadmap

**Goal:** Maximize scientific information per compute. Prefer 0-PPO diagnostics before new training. Preserve ExtraTrees confirm as immutable baseline.

---

## Priority table

| Priority | Experiment | Hypothesis | Expected Benefit | PPO Cost | Complexity | Evaluation |
|----------|------------|------------|------------------|----------|------------|------------|
| P0 | Provenance freeze + metric citation fix | Artifacts mis-cited → bad decisions | Integrity | 0 | Low | Checklist |
| P1 | Exp-D1 Identifiability & per-family error atlas | H4/H7 drive mean within-Spearman | Correct estimand | 0 | Low | Per-family table |
| P2 | Exp-D2 Between vs within variance + residual analysis | H1 | Clarify pooled gap | 0 | Low | SS shares, partial corr |
| P3 | Exp-B1 Fair LOFO benchmark (locked folds) | Model class matters | Honest model ranking | 0 | Med | Paired family Δ |
| P4 | Exp-B2 Pairwise Δφ ranker vs ET | H6 | Within-rank lift? | 0 | Med | Identifiable-family Spearman |
| P5 | Exp-F1 Feature cleanup + ratios + early deltas | H2/H8 | Signal↑ | 0 | Med | Ablation LOFO |
| P6 | Exp-S1 Adaptive extra seeds (close pairs) | H4 | Stable labels | Low–Med | Med | Order agreement |
| P7 | Exp-A1 Offline selection / budget-regret sim | Objective B | AL readiness | 0 (+later PPO) | Med | Regret vs budget |
| P8 | Exp-G1 New flat-family candidates (optional) | H5 | More identifiable ranks | High | High | New run_id only |
| P9 | CrowdNav transfer (optional, authorize) | Env shift | External validity | Very high | High | Separate protocol |

---

## Stage 0 — Reproducibility & correctness

**Actions (process, no research code edits required for audit compliance):**
- Snapshot `experiments/model_comparison_confirm/` and `results/model_comparison_confirm/` to a dated archive path.
- Document verified metrics (0.935 / 0.648 / 0.833) as baseline.
- Ban overwriting without new directory.
- List missing artifacts (`experiments/model_comparison/`, `results/phase2_report.md`).

**Acceptance:** Written provenance note in audit or DECISIONS; baseline numbers match CSV.

---

## Stage 1 — Diagnostics (0 PPO)

### Exp-D1 — Identifiability atlas
- **H:** Many families are not stably rankable; they inflate within-Spearman variance.
- **Data:** Phase-1 dataset + confirm OOF.
- **IV:** family; **controls:** same seeds/eval.
- **Primary metric:** fraction identifiable families (range ≥ κ·median SE).
- **Secondary:** per-family Spearman, Top-3, seed pairwise agreement.
- **Baseline:** current unweighted within-mean.
- **Stats:** descriptive + bootstrap over families.
- **Cost:** CPU minutes.
- **Expected:** 2–4 families flagged flat/unstable.
- **Fail:** all families identifiable yet within-mean still low → deepen H2/H3/H6.
- **Accept:** policy written for primary vs secondary family sets.

### Exp-D2 — Between/within decomposition
- **H:** H1.
- **Primary:** SS between/within; pooled Spearman on family means vs within residuals.
- **Accept:** quantitative split published in diagnosis doc (already ≈64/36 on 2D).

---

## Stage 2 — Low-cost improvements (0 PPO)

### Exp-B1 — Fair benchmark
See `BENCHMARK_PROTOCOL.md`. Same outer folds; nested HPO; report regression + ranking + selection + cost.

### Exp-B2 — Pairwise ranking
- **H:** H6 — ranking loss improves identifiable within-Spearman.
- **Model:** pairwise logistic or existing `xgb_rank_pairwise` / `lgbm_lambdarank`.
- **Primary:** mean within-Spearman on identifiable families; paired Δ vs ExtraTrees.
- **Secondary:** Top-3 hit, Kendall, regret@3.
- **Stats:** family bootstrap CI on Δ; pre-register one contrast.
- **Fail:** CI includes 0 → keep ET; do not escalate model complexity.

### Exp-F1 — Features
- Drop/near-zero within-family behavioral columns (train-fold rule).
- Add weight ratios; early ckpt deltas.
- Ablation matrix under LOFO.
- **Fail:** no lift on identifiable set → stop feature sprawl.

---

## Stage 3 — Controlled PPO

### Exp-S1 — Adaptive seeds
- **Trigger:** pairs with \|Δ\| < 2·SE or families with mid-range + weak Spearman.
- **Cap:** pre-register max extra seed-runs (e.g. ≤36).
- **Primary:** fraction of pairwise order flips; change in within-Spearman.
- **Fail:** negligible flips → seed noise not dominant for remaining errors.

---

## Stage 4 — Surrogate-assisted selection

### Exp-A1 — Offline simulation
- Frozen predictions from Stage 2 winner.
- Budgets: k = 1..N candidates fully evaluated.
- Policies: random, ET top-k, UCB/EI if uncertainty, diversity.
- **Primary:** true best-found SR vs budget; regret.
- **Then (authorize):** small online AL batch on **new** candidates or held-out generation with new run_id.

---

## Stage 5 — Robustness

- Stability across 3 model repeats (already partially done).
- New candidate pool (48) **without** destroying 108 archive.
- CrowdNav only under separate pre-registration; do not mix metrics.

---

## Dependencies & stopping criteria

```
D1 → (metric policy) → B1
B1 → B2 / F1 (parallel OK)
B2+F1 → choose winner → A1 simulation
A1 promising → S1 and/or online AL (authorize PPO)
```

**Stop escalating models if:** identifiable within-Spearman Δ vs ET has family-bootstrap CI containing 0 after B2+F1.

**Stop new PPO if:** simulation shows selection regret already near oracle at low budget.

---

## Recommended first experiment

**Exp-D1** (identifiability atlas). Unblocks honest metrics and prevents wasted ranking work on coin-flip families.
