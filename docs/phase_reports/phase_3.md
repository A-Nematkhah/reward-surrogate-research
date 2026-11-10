# Phase 3 report — Black-box extractor, sandbox, Tier 0

**Status: complete (phase3fix applied).** Items 1–5 below were blockers; all now pass under `pytest -m 'not slow'` and `scripts/regress_blackbox.py`.

## Verified

### 1. `spec_to_code` ≡ `WeightedRewardFunction` (blocking — fixed)

- `specs_from_rows` reads `weight_*` / `scale_*` (fallback `w_*` / `s_*`).
- All `COMPONENT_NAMES` (14) inlined with formulas from `rsr/rewards/components.py` (collision +1 with out_of_bounds, smoothness `||da||/sqrt(2)`, time +1, obstacle clip proximity, energy/risk/potential/nonlinear_progress/distance/human_*/crowded_progress).
- Test: all **108** specs on phase2 real 2D probes; sandbox loader vs `WeightedRewardFunction`; **max |diff| = 1.78e-15** (< 1e-9).

### 2. Regression (real test — fixed)

Artifacts: `results/phase3_blackbox_regress/report.json` (status **pass**).

| Criterion | Result |
| --- | --- |
| reward max \|diff\| < 1e-9 | **pass** (1.78e-15) |
| bb feature Spearman ≥ 0.95 vs oracle (same probes) | **pass** (65/65) |
| LOFO within-family Spearman delta ≥ −0.05 vs parametric math+behavioral | **pass** (delta **0.0**; within = 0.581) |

Secondary note: `bb_*` groups alone give within-Spearman 0.444 (Δ −0.136 vs math+behavioral). Primary parity uses math+behavioral extractors on streams that match WRF after item 1.

Probes for this regression: `experiments/phase2/probe_trajectories.npz` (same set as existing math/behavioral).

### 3. Sandbox hardening (blocking — fixed)

- Removed `typing` from allowlist; expose `math` + numpy **proxy** (no `save`/`load`/real module).
- AST rejects `_` attributes, `sys`/`os`/`modules`/save/load/…, and dynamic `getattr`.
- Child clears `os.environ`; POSIX applies `RLIMIT_AS` + `RLIMIT_CPU`.
- **Honest docs:** Python-level sandbox is not a security boundary; Phase 4 LLM code must run in container/bwrap/firejail without network/secrets (`platform_isolation_note()`).
- Escape tests (`typing.sys`, `np.save`, `().__class__.__mro__`, env read) → `exec_error`.

### 4. Harmless builtins (fixed)

Added `round`, `pow`, `any`, `all`, `sorted`, `divmod`, `reversed`, `__build_class__`, `__name__`. Corpus of **20** EUREKA-style snippets runs without NameError.

### 5. Probes (fixed)

- Frozen real probes in `data/probes/{navigation_2d,crowdnav,irsim}.npz` with `partial_expert` + `edge_states` present.
- Synthetic only under `tests/fixtures/probes/`.
- `--from-env` required for `data/probes`; **fails loudly** if env missing (no silent synthetic).
- Seeds via `zlib.crc32` (stable); seed recorded in NPZ meta.

## Tests

`pytest -m 'not slow' -rs` → green, **0 skipped** (slow PPO tests deselected only).

## Remaining gaps / honesty

- Windows still has no RLIMIT; isolation weaker than POSIX.
- Container/bwrap requirement for Phase 4 is documented in code notes; `docs/INTEGRATION.md` host section still deferred to Phase 7.
- Secondary `bb_*`-only LOFO is weaker than math+behavioral; do not claim bb groups alone match parametric ranking quality.

## Package layout (unchanged intent)

`reward_triage/` — schema, executor, adapters, features, tier0, probes, spec_to_code; importable without torch.
