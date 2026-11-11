# Pre-registration — Phase 6 (live 2D → Gate 2)

**Date:** 2026-11-11  
**Status:** pre-registered before confirmatory Phase-6 runs.

Phase 6 **collects** lineage-native 2D labels with the Phase-5 host loop and **evaluates** Gate 2. Decision criteria, metrics, lineage grouping, and frozen thresholds live in:

→ **`docs/prereg/gate2.md`** (source of truth)

## Phase-6 checklist (execution order)

1. Commit `docs/prereg/gate2.md` + this file before looking at Gate-2 metrics.
2. Pilot: `mini_host_loop` with `--max-jobs 1` (real PPO) once; no full campaign until user go-ahead.
3. Full 2D campaign: 3 independent runs (N=16×G=3×3 seeds); warm-up `off` → `active` after `m_min=12`; auto-adapt on.
4. Analyze with lineage groups; write `docs/phase_reports/gate2.md` + `phase_6.md`.
5. **Stop** for human decision (Gate 2). No CrowdNav / IR-SIM auto-start.
