# Gate 2c — harness fix + novelty-skip (EXPLORATORY metrics; verdict unchanged)

**Gate 2 verdict remains FAIL.** Frozen thresholds unchanged.  
`docs/phase_reports/gate2.md` now marks the original analysis **INVALID FOR INFERENCE** (duplicate-code leakage + plateau) and removes the former “allowed claim.”

Artifacts: `experiments/gate2c_harness/`. Scripts/modules: `scripts/run_gate2c_harness.py`, `reward_triage/gate2_harness.py`, `reward_triage/label_cache.py`; host wiring in `host_loop.py` / `mini_host_loop.py`. Tests: `tests/test_gate2c_harness.py`.

## 1. Leakage census

| | |
| --- | ---: |
| Labeled rows | 102 |
| Unique `code_hash` | **69** |
| Hashes with &gt;1 row (cross-run reuse) | **17** |

## 2. Corrected harness (dedupe + code_hash folds)

- Fold key: **`code_hash`** (after averaging duplicate-hash labels).
- Test: `assert_no_code_hash_leak` — no hash in both train and test (`enforce_no_hash_leak=True`).
- Pool metrics (**EXPLORATORY**, not a new Gate):

| Slice | n | recall@5 | med regret@5 | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Pooled deduped | 69 | **0.493** | 0.000 | 0.697 |
| Gen0 only | 16 | 1.000† | 0.000 | 0.732 |
| Plateau (gen≥1) | 53 | **0.365** | 0.004 | 0.389 |

†Gen0 has only 16 unique hashes → N=16 pools are the full set → recall is **degenerate**, not evidence of skill.

## 3. Label cache (pipeline)

- `LabelCache` keyed by `code_hash` under `experiments/_label_cache/`.
- Host loop: on real PPO, **get** before train; **put** after; log `label_cache_hit_saved_steps=…` in selection reasons; `run_summary.label_cache_stats` records hits/misses/`saved_ppo_steps`.
- Disable with `--no-label-cache`.

## 4. Novelty-skip (offline, CV threshold)

mb-feature NN; skip if distance ≤ CV-chosen threshold; predicted label = neighbor label.

| | |
| --- | ---: |
| CV threshold | ~0 (exact / near-exact duplicates dominate) |
| Skip rate | **0.28** (29/102) |
| Label error | **0.0 SE** on skipped (identical neighbors) |

Useful as a **cache/dedup** layer for this local corpus; not a learned novelty filter yet.

## 5. Noise-aware metrics (**EXPLORATORY** until preregistered)

| Metric | Value |
| --- | ---: |
| eps-recall@5 (surrogate; within max(1.5·SE, 0.03) of pool best) | **0.986** |
| Perfect-scorer recall under label noise | **0.951** |

## 6. Run-id-dependent PPO seeds

Default: `seeds_for_run(run_id)` → independent triples per run (logged in `run_summary.seeds`).  
Opt out: `--fixed-seeds` (legacy `[0,1,2]`).

Examples: r1b=`(871806425, 25110619, 1671963997)` vs r2=`(558773587, …)` — distinct.

## Stop

No new PPO. No threshold edits. Gate 2 FAIL + INVALID FOR INFERENCE. Await human decision before Gate 3 / other envs / new confirmatory prereg.
