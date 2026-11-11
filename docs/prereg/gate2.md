# Pre-registration — Gate 2 (Phase 6 confirmatory)

**Date:** 2026-11-11  
**Status:** pre-registered **before** any confirmatory Phase-6 / Gate-2 live analysis.  
**Depends on:** `docs/METRIC_SPEC_TRIAGE.md`, `docs/prereg/gate1_thresholds_frozen.md`, Phase-5 lock (`early+math+behavioral`).

Commit this file **before** inspecting Gate-2 pool metrics or writing `docs/phase_reports/gate2.md`.

---

## Goal

Decide whether the **operating** per-env surrogate (ExtraTrees + `early+math+behavioral`, online adapt) is viable for **generation-level triage** on **2D** under lineage-grouped confirmatory tests.

Gate 1 was CONDITIONAL (family groups; only 12 families). Gate 2 re-tests recall-level claims with **lineages** as groups.

## Scope (Phase 6)

| In | Out |
| --- | --- |
| 2D only (`RewardNavigation2D-v0`) | CrowdNav++, IR-SIM (deferred until after Gate 2 + user go-ahead) |
| Live labels from mini-host loop | Pre-collected transfer datasets; cross-env transfer |
| Local isolation (Docker deferred) | New PPO outside the registered 2D campaign |
| LLM: `local` corpus until xAI key; then `grok-4.3` (VERIFY id) | Rewriting locked confirm / archive artifacts |

### Live campaign (sizing locked in `docs/phase4_sizing.md`)

- Per run: **N=16** × **G=3** (=48 candidates), **3 seeds**, workers=4.
- Independent runs for CIs: **3** (new `run_id` each; do not overwrite).
- Jobs/run ≈ 144; jobs total ≈ 432 (audit overhead small).
- Warm-up: `--mode off` (train-all) until labeled count ≥ **`m_min=12`**, then `--mode active` with surrogate + **auto-adapt** (prior blend on).
- Operating selector: `k_frac=0.5`, `audit_frac=0.15`, OOD always kept.
- Early feature checkpoints: 2k / 5k / 12k (match `configs/ppo.yaml`).

Pilot only with `--dry-run` / `--max-jobs 1` until the user launches the full campaign.

## Grouping unit: lineage

- **Group key:** `candidate.lineage_id` (required on every store record).
- Gen-0: new lineage per candidate (`lin_<run_id>_<i>`). Mutate: child inherits parent lineage.
- Confirmatory CIs / LOO / bootstrap: resample **lineages**, not families.
- **Minimum for a confirmatory claim:** ≥ **20 held-out lineages** (as stated in `docs/prereg/gate1_thresholds_frozen.md`).
- If a finished campaign yields &lt; 20 held-out lineages under the chosen split, Gate 2 verdict is **INCONCLUSIVE** on recall-level claims (still report point estimates; no pass).

Suggested split (exploratory only if changed after seeing counts): ~20% lineage hold-out when total lineages ≥ 100; with the 3×48 plan expect on the order of ~100+ lineages if most gen-0 starts are unique — **verify from the store before claiming**.

## Target label

- Primary: absolute **`success_rate`** (mean over completed seeds).
- Diagnostic only (not fitted, not in Gate-2 pass/fail): collision, timeout, return.

## Feature set and model (locked)

| Role | Choice |
| --- | --- |
| Operating features | **`early+math+behavioral`** |
| Ablations (exploratory) | `math+behavioral`, `early` |
| Model | ExtraTrees (+ median impute); residual conformal; Mahalanobis OOD @ train 99th pct |
| Online | `adapt_after_generation` with phase1_108 prior blend (default on) |

Anything tuned after seeing Phase-6 live metrics is **exploratory**.

## Pool and metrics

Use `rsr.evaluation.triage_metrics` / pool simulator (same as Gate 1).

| Setting | Value | Role |
| --- | ---: | --- |
| Pool size **N** | **16** | Primary (generation-sized) |
| Keep **k** (threshold eval) | **5** | Continuity with Gate 1 / `METRIC_SPEC_TRIAGE.md` |
| Keep **k** (live loop) | `k_frac×N` (=8 at 0.5) | Operating report only; **not** Gate-2 pass/fail |
| Cost early steps | **12 000** | Per candidate scored with early features |
| Cost full steps | **50 000** | Per fully trained candidate |

### Metrics (definitions unchanged)

1. `recall_at_k`
2. `regret_at_k` (success-rate units)
3. Δ`recall_at_k` vs **random-k** (lineage-bootstrap CI)
4. `savings_frac` vs train-all, with equal-or-better best-found (via regret/recall)
5. Secondary (report, not pass/fail): pairwise accuracy; within-generation ranks; Spearman only if n≥4

Baselines: surrogate top-k, random-k, oracle-k. Also report Tier-0-only if available.

### Pool construction

- Primary: pools of N=16 drawn from **held-out lineages** (expand across lineages if one lineage is too small; always also report **strict within-lineage** pools when lineage size ≥ N, else note impossible).
- Generation-replay (secondary): treat each live generation as a pool; score with the surrogate version **available before that generation’s adapt** (no peeking).

## Viability thresholds (frozen for Gate 2)

Applied to **lineage-held-out** estimates at **N=16, k=5**:

| # | Criterion | Threshold |
| --- | --- | --- |
| 1 | Mean `recall_at_k` | ≥ **0.80** |
| 2 | Median `regret_at_k` | ≤ **0.03** |
| 3 | Beats random-k on recall | Lineage-bootstrap CI on (surrogate − random) **excludes 0** |
| 4 | Step savings at equal-or-better best-found | `savings_frac` ≥ **0.40** |

### Threshold 4 amendment (pre-registered)

Gate 1 / `METRIC_SPEC` used **≥ 0.50**, calibrated for **zero-PPO-cost** features (`feature_steps=0` → k=5 saves 5/16 ≈ 68.75%).

With locked **early** features (`feature_steps=12 000`):

```
savings_frac = 1 - (12000/50000 + k/N) = 1 - (0.24 + k/N)
```

At **k=5, N=16**: max formula savings ≈ **0.447**. A 0.50 bar is unreachable without lowering k. Gate 2 therefore freezes threshold 4 at **≥ 0.40** under early=12k / full=50k, still requiring equal-or-better best-found via regret/recall.

Do **not** change thresholds 1–3. Do not raise or lower 0.40 after seeing Gate-2 numbers.

## Pass / fail / stop

| Verdict | Rule |
| --- | --- |
| **PASS** | All four thresholds met **and** ≥20 held-out lineages used for CIs |
| **CONDITIONAL** | Point estimates meet 1–2–4 but CI (criterion 3) or lineage count fails |
| **FAIL** | Point estimate misses 1, 2, or 4 after a complete 3-run 2D campaign |
| **INCONCLUSIVE** | Campaign incomplete, or &lt;20 held-out lineages |

**Stop for human decision** after `docs/phase_reports/gate2.md`. Do not start CrowdNav / IR-SIM or Gate 3 from that report without explicit go-ahead.

## Outputs

- Live stores: `experiments/<run_id>/` (label_store, surrogate/gen_*, run_summary) — one new `run_id` per independent run
- Analysis: `experiments/gate2_2d/` and/or `results/gate2_2d/` (new roots only)
- Report: `docs/phase_reports/gate2.md` (verdict + threshold checklist + lineage counts)
- Phase wrap: `docs/phase_reports/phase_6.md`

## Non-goals

- No overwrite of `experiments/model_comparison_confirm/`, `results/model_comparison_confirm/`, archives
- No Gate 1b / cross-environment transfer
- No claiming operating `k_frac=0.5` (k=8) against the k=5 threshold table
- No threshold edits after results are visible
