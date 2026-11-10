# Phase 3 report — Black-box extractor, sandbox, Tier 0

**Status:** complete (no PPO). Package `reward_triage/` is importable without torch.

## Deliverables

| Item | Location |
| --- | --- |
| Schema | `reward_triage/schema.py` |
| Sandbox executor | `reward_triage/executor.py` |
| Adapters | `reward_triage/adapters/` + `configs/adapters/{eureka,card,evonav}.yaml` |
| Probes (frozen NPZ) | `data/probes/{navigation_2d,crowdnav,irsim}.npz` (+ `.sha256`) |
| Black-box features | `reward_triage/features.py` (`bb_stats`, `bb_corr`, `bb_policy`, `bb_shape`, `bb_comp`, `code_*`) |
| Tier 0 | `reward_triage/tier0.py` |
| Spec → code | `reward_triage/spec_to_code.py` |
| Builder / regress | `scripts/build_probe_sets.py`, `scripts/regress_blackbox.py` |
| Tests | `tests/test_reward_triage_*.py` |

## Windows isolation (VERIFY)

On Windows, the sandbox uses `multiprocessing` (spawn) plus a wall-clock join timeout. There is **no** reliable POSIX `RLIMIT` memory cap in the stdlib path, so isolation is weaker than on Linux. Import allowlist (`math`, `numpy`, `typing`; `torch` behind a flag) and blocked `open`/disallowed imports still apply. See `reward_triage.executor.platform_isolation_note()`.

## Tier 0 policy

Hard reject **only** for: `exec_error`, `timeout`, `non_finite`, `constant` (reward std < 1e-8 across probes). All other issues are `warn` + features (never reject), to limit false rejects.

## Probe sets

Synthetic frozen probes committed for all three env ids (CI-safe; no CrowdNav/IR-SIM third_party required). Content hashes in sibling `.sha256` files. Real env rollouts remain optional via `scripts/build_probe_sets.py --from-env` (falls back to synthetic with a printed note).

## Regression

`scripts/regress_blackbox.py` (slow / local datasets) ran on 24 parametric specs with synthetic probes:

- Tier0 ok fraction: 1.0
- Spearman bb_stats vs math magnitude columns: ~0.13–0.49 — **below 0.95**; documented in `results/phase3_blackbox_regress/report.json` (probe-based bb features are not literal copies of weight-derived math features on synthetic probes).
- LOFO within-family Spearman parity vs parametric OOF is deferred to a full local run with confirm artifacts (noted in report).

## Tests

`pytest -m 'not slow' -rs` → **144 passed, 3 deselected, 0 skipped**. Deselected: slow PPO trainer tests.

## Packaging

`pyproject.toml` includes `reward_triage*`. CI ruff checks `reward_triage`.

## Next

Phase 4 (LLM-native dataset) is **not** started from this report.
