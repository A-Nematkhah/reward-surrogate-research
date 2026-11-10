# Triage metric specification (Phase 1)

Frozen after merge of Phase 1. All Gate 1+ reports must use
`rsr.evaluation.triage_metrics` (and `cost.py` / `pool_simulator.py`).

## Pool

A **pool** is N candidates with true labels `y_true` (usually `success_rate`) and
scores `y_score` (surrogate predictions). The selector keeps **k** candidates.

## Metrics

| Metric | Definition | Notes |
| --- | --- | --- |
| `recall_at_k` | 1 if the true best pool member is in the kept set | Tie-break true-best and scores with seeded noise |
| `regret_at_k` | `max(y_true) - max(y_true[kept])` | Lower better; success-rate units |
| `top1_hit` | `recall_at_k` with k=1 | Primary for n=2 families (CrowdNav v5 / IR-SIM) |
| `pairwise_accuracy` | Fraction of label-ordered pairs with matching score order | Optional weight `\|Δy\| / SE`; skip label ties |
| `spearman` | Spearman(y_true, y_score) | **Only if** n ≥ 4 and neither side constant; else NaN |

## Edge cases

- **n=2:** Spearman is NaN; use `top1_hit` and `pairwise_accuracy`.
- **n=3:** Spearman still NaN (min_n=4); recall/regret/pairwise OK.
- **Constant labels or scores:** Spearman NaN; pairwise skips zero-Δ pairs.
- **Ties:** `break_ties` adds seeded `U(0,1e-12)` noise before argsort (reproducible).
- **k ≥ N:** keep everyone; recall=1, regret=0.
- **k ≤ 0:** empty kept set; recall/regret NaN.

## Cost

```
cost_steps = feature_steps_per_candidate * N + full_run_steps * k
baseline_steps = full_run_steps * N
savings_frac = 1 - cost_steps / baseline_steps
```

Default 2D confirm costs: feature early = 12 000, full run = 50 000.

## Pool simulator

From locked OOF predictions, sample pools of size N ∈ {8,16,32} from 1–3
families (add families if one is too small). **Caveat:** multi-family pools mix
OOF scores from different LOFO fold models. On 2D, families have size 8–10, so
N=16 with `n_groups_requested=1` is a **2-family pool (n_groups=1 expanded to 2
families)** — not a within-family pool. Always also report strict within-family
pools (`expand_groups=False`) separately.

Baselines: `surrogate` (top-k by y_pred), `random-k`, `oracle-k`.

## Proposed viability thresholds (editable before Gate 1 freeze)

On **N=16**, **k=5**, ≥20 held-out groups (families):

1. `recall_at_k` mean ≥ **0.80**
2. median `regret_at_k` ≤ **0.03** success rate
3. beats random-k on recall with bootstrap CI on (surrogate − random) excluding 0
4. saves ≥ **50%** of full-run steps at equal-or-better best-found success

Zero-PPO-cost features (feature_steps=0, k=5, N=16) save `1 - 5/16 = 68.75%` steps
by the cost formula alone; equal-or-better best-found is judged via regret/recall.
