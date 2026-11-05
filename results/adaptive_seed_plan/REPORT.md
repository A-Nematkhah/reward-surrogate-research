# Adaptive seed plan (Phase 4) — no PPO launched

- Close pairs (|Δ| < 2.0·SE): **193**
- Weak identifiable families (Spearman < 0.55): **2**
- Budget cap: **36** extra seed-runs
- Selected candidates: **12** → **36** runs
- Families touched: `nonlinear_progress, risk_sensitive`

## Authorization gate

This plan does **not** start PPO. To execute later:
1. Review `selected_extra_seeds.csv`
2. Use a **new** experiment root (do not overwrite phase1 archive)
3. Cap wall-clock to the planned run count

## Top selected

| candidate | family | extra seeds | score | reason |
|-----------|--------|------------:|------:|--------|
| nonlinear_progress_01 | nonlinear_progress | 3 | 6.50 | weak_family_spearman,close_pairs=8,high_seed_std |
| nonlinear_progress_09 | nonlinear_progress | 3 | 6.25 | weak_family_spearman,close_pairs=7,high_seed_std |
| risk_sensitive_01 | risk_sensitive | 3 | 6.25 | weak_family_spearman,close_pairs=7,high_seed_std |
| risk_sensitive_05 | risk_sensitive | 3 | 6.25 | weak_family_spearman,close_pairs=7,high_seed_std |
| risk_sensitive_00 | risk_sensitive | 3 | 6.00 | weak_family_spearman,close_pairs=6,high_seed_std |
| nonlinear_progress_07 | nonlinear_progress | 3 | 5.75 | weak_family_spearman,close_pairs=5,high_seed_std |
| nonlinear_progress_03 | nonlinear_progress | 3 | 5.50 | weak_family_spearman,close_pairs=6 |
| nonlinear_progress_05 | nonlinear_progress | 3 | 5.25 | weak_family_spearman,close_pairs=5 |
| risk_sensitive_02 | risk_sensitive | 3 | 5.25 | weak_family_spearman,close_pairs=5 |
| risk_sensitive_07 | risk_sensitive | 3 | 5.25 | weak_family_spearman,close_pairs=5 |
| risk_sensitive_08 | risk_sensitive | 3 | 5.25 | weak_family_spearman,close_pairs=5 |
| nonlinear_progress_04 | nonlinear_progress | 3 | 5.00 | weak_family_spearman,close_pairs=4 |
