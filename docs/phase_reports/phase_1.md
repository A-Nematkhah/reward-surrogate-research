# Phase 1 report — Generation-level triage metrics

## Changes

- `rsr/evaluation/triage_metrics.py` — recall@k, regret@k, top1_hit, pairwise_accuracy, spearman_safe, bootstrap CIs, seeded tie-break.
- `rsr/evaluation/pool_simulator.py` — OOF pool sampling (N∈{8,16,32}, 1–3 groups, add groups if too small); selectors surrogate / random / oracle.
- `rsr/evaluation/cost.py` — feature×N + full×k cost and savings vs train-all.
- `docs/METRIC_SPEC_TRIAGE.md` — frozen definitions + proposed viability thresholds.
- `scripts/run_triage_phase1.py` — reproducible runner → `results/triage_phase1/`.
- Tests: `tests/test_triage_metrics.py`, `tests/test_pool_simulator.py`.
- Master Context rule updated for revised pack (no HighwayEnv; 2D / CrowdNav++ / IR-SIM).

## Commands

```bash
pytest tests/test_triage_metrics.py tests/test_pool_simulator.py -q
python scripts/run_triage_phase1.py --force
pytest -m "not slow"
```

## Numbers (locked OOF: ExtraTrees / early+math+behavioral / success_rate, repeat=0)

Primary focus **N=16, 2-family pool (n_groups=1 expanded to 2 families), 2000 draws**:

| k | recall@k (sur) | med regret | Δrecall vs random (mean [CI]) |
| --- | --- | --- | --- |
| 3 | 0.876 | 0.000 | 0.693 [0.670, 0.715] |
| 4 | 0.890 | 0.000 | 0.656 [0.634, 0.679] |
| 5 | **0.896** | **0.000** | **0.613 [0.589, 0.635]** |
| 6 | 0.910 | 0.000 | 0.532 [0.507, 0.556] |

N=16, k=5, 2-family pool (n_groups=1 expanded to 2 families) baselines: random recall≈0.283; oracle=1.0.

Cost at N=16, k=5 with feature_steps=12 000, full=50 000: **savings_frac=0.4475** (< 0.50).

### Vs proposed thresholds (not frozen)

| Criterion | Result |
| --- | --- |
| recall@5 ≥ 0.80 | **pass** (0.896) |
| median regret@5 ≤ 0.03 | **pass** (0.0) |
| beat random (CI excludes 0) | **pass** |
| save ≥ 50% full-run steps | **fail** at early=12k (44.8%); would pass if feature_steps=0 |

## Caveats

- Multi-group pools mix OOF scores from different LOFO fold models; `summary_single_group.csv` (filename kept) is the 2-family-pool matched-model view.
- Thresholds remain **proposed** until user edits / Gate 1 freeze.

## Open issues / Phase 0 gaps from revised pack

1. **Need decision:** unify `main` vs `master` (currently on `master`, ahead 4 phase0 commits). Which name stays default?
2. Pin IR-SIM as well (`scripts/setup_env_backends.sh` or extend setup script) and export `experiments/irsim_v1_nav24` into `data/` (dataset exists locally).
3. LLM provider / compute budget still open before Phase 4.
