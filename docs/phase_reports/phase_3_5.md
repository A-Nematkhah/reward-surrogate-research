# Phase 3.5 report — Deployable extractor gap close (no PPO)

**Status: complete.** Closes the Gate-1 → `reward_triage` extractor gap: default `mb` features, 108-spec regression, sparse Tier-0, container isolation scaffold, Phase-4 sizing.

## 1. Feature group `mb`

- `reward_triage/mb_features.py`: Gate-1 `math+behavioral` extractors from sandbox reward streams (no torch).
- Behavioral components match `GATE1_BEHAVIORAL_COMPONENTS` (11; no `human_*` / `crowded`).
- `triage_candidate` / `extract_features` default groups: `("mb",)`. Optional: `bb_*`, `code`.

## 2. Regression (108 2D specs)

Script: `scripts/regress_phase35_mb.py` → `results/phase3_5_mb/report.json`.

| Criterion | Result |
| --- | --- |
| Feature max \|diff\| per column < 1e-9 vs Gate-1 table | **pass** (max 2.84e-14; 214 columns) |
| LOFO within-family Spearman vs Gate-1 table within 0.01 | **pass** (pipeline = table = **0.586**; Δ = 0.0) |

Note: Gate-1 prose often cites ~0.57; the locked feature-table LOFO is 0.586. Regression criterion is equality to the table (±0.01), which holds exactly.

## 3. Tier 0 sparse rewards

- Rule: near-zero constant (`std≈0` and `|mean|≈0`) → hard reject `constant`.
- Non-zero constant (e.g. pure time penalty) → `warn` only (episode returns still vary with horizon).
- Tests: `tests/test_reward_triage_sparse_tier0.py` — goal-only, collision-only, time-only on real probes for 2D / CrowdNav / IR-SIM are **not** hard-rejected.

## 4. Container isolation

- `Dockerfile.triage`, `docker-compose.triage.yml`, `scripts/run_in_container.sh`
- `network_mode: none`, non-root uid 10001, read-only bind of repo, tmpfs `/tmp`, CPU/mem limits, empty API key envs.
- `reward_triage/container_selftest.py` + `tests/test_reward_triage_container.py` (explicit skip if docker unavailable).

## 5. Phase 4 sizing

- `scripts/estimate_phase4_cost.py` → `docs/phase4_sizing.md`
- ≥20 held-out lineages at 20% hold-out → **100** total lineages (20 held-out / 80 train).
- Target **864** candidates × 3 seeds / env; parallel wall-h (4 workers): 2D 21.6, IR-SIM 37.8, CrowdNav 72.7.

## Tests

`pytest -m 'not slow'` green (container test skipped when docker unavailable).

## Out of scope (intentionally)

- No PPO launched.
- Phase 4 generation / labeling not started.
- Locked Gate-1 confirm artifacts untouched.
