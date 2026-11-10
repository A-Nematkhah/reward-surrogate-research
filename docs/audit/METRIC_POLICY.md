# Within-Family Metric Policy (Phase 1)

**Status:** Adopted 2026-11-05 after audit + identifiability diagnostic.  
**Environment:** Toy 2D `RewardNavigation2D` (108-candidate archive). CrowdNav uses a separate report.

---

## Baseline (immutable)

| Item | Value |
|------|-------|
| Archive | `experiments/archives/model_comparison_confirm_2026-11-05/` |
| Model | ExtraTrees |
| Features | `early+math+behavioral` |
| Target | `success_rate` |
| Historical all-family within Spearman (confirm mean) | ≈ 0.648 |
| Pooled Spearman | ≈ 0.935 |
| Top-3 hit (all families) | 0.833 |

Do not overwrite live confirm dirs without a new output run_id.

---

## Identifiability gate

A family is **identifiable** for primary within-family ranking claims if:

\[
\mathrm{range}(Y_f) \ge \max\big(0.05,\; 2 \cdot \mathrm{median\_SE}_f\big)
\]

where \(Y_f\) are seed-mean `success_rate` values of candidates in family \(f\), and \(\mathrm{median\_SE}_f\) is the median across candidates of \(\mathrm{std}_s(Y)/\sqrt{n_{\mathrm{seeds}}}\).

Regenerate the table with:

```bash
python scripts/analyze_within_family_identifiability.py
```

Artifacts: `results/diagnostics_within_family/`.

### Snapshot (2026-11-05 run)

| Set | Mean OOF Spearman | Mean Top-3 hit |
|-----|------------------:|---------------:|
| All 12 families | 0.648 | 0.833 |
| Identifiable (11) | **0.765** | **0.909** |
| Non-identifiable (`distance_dense` only) | −0.635 | 0.000 |

Remaining hard **identifiable** families (priority for Phase 3/4): `risk_sensitive` (0.39), `nonlinear_progress` (0.43).

---

## What to report

| Role | Metric | Scope |
|------|--------|--------|
| **Primary (within)** | Mean within-family Spearman | **Identifiable families only** |
| **Primary (selection)** | Within-family Top-3 hit | Identifiable families; also report all-12 |
| **Secondary (within)** | Mean within-family Spearman | All 12 families (transparency) |
| **Secondary** | Pooled Spearman, MAE | All candidates |
| **Label clearly** | Top-10 regret | **Pooled**, not within-family |
| **Always list** | Non-identifiable families | Separate section; do not silently drop from secondary |

---

## Interpretation rules

1. Do not treat a drop in all-12 within-Spearman as model failure if it is driven by flat families.  
2. Do not treat high Spearman on near-zero SR families as evidence of triage skill.  
3. Pairwise ranking experiments must exclude or down-weight pairs with \(|\Delta Y| < \mathrm{SE}\).  
4. Family-level bootstrap remains the uncertainty unit (n=12), even when filtering to identifiable subsets (smaller n → wider CI).  
5. CrowdNav numbers must never be mixed into 2D primary tables.

---

## Next pipeline change (Phase 2)

Add optional metrics (default **off** so historical confirm CSV stays comparable):

- `within_family_spearman_identifiable_mean`
- `n_families_identifiable`

See `EXECUTION_PLAN.md` Phase 2.1.
