# Frozen Gate 1 thresholds

Frozen **before** Gate 1 amendment analyses (`run_id=gate1_amendments`).
Threshold values are copied exactly from `docs/METRIC_SPEC_TRIAGE.md` and were
applied as proposed (no value edits).

## Viability thresholds (unchanged)

On **N=16**, **k=5**, ≥20 held-out groups (families):

1. `recall_at_k` mean ≥ **0.80**
2. median `regret_at_k` ≤ **0.03** success rate
3. beats random-k on recall with bootstrap CI on (surrogate − random) excluding 0
4. saves ≥ **50%** of full-run steps at equal-or-better best-found success

## Grouping note (Gate 1 vs Gate 2)

- Gate 1 used **families** as groups. Each 2D / CrowdNav / IR-SIM campaign has
  only **12 families**, so the “≥ 20 held-out groups” clause **cannot** be met
  at family level. Family-bootstrap CIs are descriptive under that limit.
- **Gate 2** confirmatory testing uses **lineages** as groups, with a requirement
  of **≥ 20 held-out lineages** (from Phase 4 LLM-native data).

## Application record

These four thresholds were applied as proposed to Gate 1 / addendum results.
The Gate 1 operating claim is **conditional** (see `docs/phase_reports/gate1.md`);
confirmatory establishment of recall levels is deferred to Gate 2.
