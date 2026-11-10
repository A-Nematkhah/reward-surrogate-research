# Pre-registration: Gate 1 — zero-PPO-cost feasibility

Committed **before** looking at Gate 1 results.

## Goal

Decide whether a **zero-PPO-cost** surrogate (no early PPO features) is viable for
generation-level triage under the Phase 1 thresholds in `docs/METRIC_SPEC_TRIAGE.md`.

## Target

- Label: `success_rate` only (collision/timeout stay diagnostic, not fitted).

## Feature sets

| Role | Feature set | PPO feature cost |
| --- | --- | --- |
| **Primary (pre-registered)** | `math+behavioral` | 0 |
| Secondary | `math`, `behavioral` | 0 |
| Secondary (research only) | `static` | 0 |
| Secondary (costly comparators) | `early`, `early+math`, `early+math+behavioral` | 12 000 (2D) |

Names verified in `rsr/surrogate/model_comparison.py` `COMBO_FEATURE_SETS` / base CSVs.
Builder semantics unchanged.

## Models

`mean`, `ridge`, `extra_trees` (reference). Inner LOFO hyperparameter search as in
`configs/model_comparison.yaml`. Outer protocol: leave-one-family-out, `n_repeats=3`.

## Data

- **2D (108):** `experiments/phase2/features` (required).
- **CrowdNav v4 (72):** run only if `experiments/crowdnav_v4_full_150k/` features exist locally.
- **CrowdNav v5 / IR-SIM v1_nav24 (24=12×2):** report top-1 and pairwise only if local
  datasets/features exist; do not launch new PPO/IR-SIM jobs.

## Metrics (from Phase 1 module)

Pool simulator on OOF predictions: recall@k, regret@k vs random/oracle; cost savings.
Primary pool setting for the verdict: **N=16, k=5**, **2-family pool (n_groups=1 expanded to 2 families)** and multi-group noted.

Also: offline successive halving (SH) if `early_t{5k,12k,25k}_success_frac` columns exist.

## Viability thresholds (proposed; freeze with this gate unless user edits)

On N=16, k=5, ≥20 held-out groups:

1. recall@k ≥ 0.80  
2. median regret@k ≤ 0.03  
3. beats random-k (bootstrap CI on Δrecall excludes 0)  
4. saves ≥ 50% of full-run steps at equal-or-better best-found success  

**Primary claim** uses ExtraTrees + `math+behavioral`. Secondary sets are exploratory
for the frontier plot only.

## Outputs

- `experiments/gate1_zero_cost/`, `results/gate1_zero_cost/`
- `docs/phase_reports/gate1.md` with pass/fail + Plan B if zero-cost fails

## Non-goals

No new PPO training. No writes to locked confirm dirs.
