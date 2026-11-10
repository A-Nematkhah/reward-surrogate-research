# Phase 0 report — Repo hygiene and provenance

## Changes

- Master Context rule: `.cursor/rules/reward-triage.mdc` (`alwaysApply`).
- `docs/PROVENANCE.md` maps README headline numbers to locked confirm artifacts; README Key results now report **pre-registered** (`early+math`) and **operating** (`early+math+behavioral`, post hoc).
- Comment-only note in `configs/model_comparison.yaml` (values unchanged).
- `scripts/export_datasets.py` exports seed-aggregated candidates + feature tables to `data/<env>/<run_id>/` with `MANIFEST.json`. Total size **5.36 MiB** (< 20 MiB; no LFS).
- CrowdNav++ pin: `scripts/setup_third_party.sh` → `env/crowdnav` @ `390773137be04ed14e27620dc5fd7c5e1a5b1f62` (verified local). README / `.gitignore` wording: not vendored.
- Package rename: `src` → `rsr` (imports, scripts, tests, gym entry point).
- `pyproject.toml` (requires-python `>=3.10,<3.13`; extras `rl` / `extra` / `dev`) + `requirements.lock`.
- Lazy SB3 typing imports in `rsr/features/early_behavioral.py` and `rsr/evaluation/evaluator.py`; `reproducibility` catches `(ImportError, OSError)`; slim package `__init__`s; metrics no longer imports gymnasium.
- Stale CrowdNav pool test already expected 24; documented 48-vs-108 2D mismatch; PPO tests `@pytest.mark.slow`.
- Report Artifacts paths are repo-relative (`scripts/report_model_comparison.py`). Locked reports not rewritten.
- `CHECKSUMS.sha256` for `experiments/model_comparison_confirm/`; CI verifies.
- `run_model_comparison.py` requires `--run-id`, refuses non-empty dirs without `--force`, blocks locked confirm paths.
- MIT `LICENSE`, `CITATION.cff`, `.github/workflows/ci.yml`.
- `research_audit/` → `docs/audit/` + `INDEX.md`.
- `scripts/reproduce_confirm_table.py` acceptance helper.

## Commands

```bash
pip install -e ".[dev]"
bash scripts/setup_third_party.sh   # CrowdNav++ pin
python scripts/export_datasets.py --force
python scripts/reproduce_confirm_table.py
pytest -m "not slow"
sha256sum -c CHECKSUMS.sha256       # or Get-FileHash on Windows
```

## Numbers

| Item | Value |
| --- | --- |
| Confirm commit (locked) | 2fb024c |
| ExtraTrees early+math (pre-reg) | pooled 0.933 / within 0.616 / top3 0.722 |
| ExtraTrees early+math+behavioral (operating, post hoc) | pooled 0.935 / within 0.648 / top3 0.833 |
| reproduce_confirm_table | matches metrics_summary within 1e-3 |
| data/ export size | 5.36 MiB |
| pytest -m 'not slow' | 117 passed, 3 deselected (slow) |

## VERIFY notes

- Local `experiments/phase1/dataset` and `experiments/phase2/features` present → exported as `data/navigation_2d/phase1_108/`.
- `experiments/crowdnav/{dataset,features}` present → `data/crowdnav/crowdnav_local/`.
- Versioned trees `experiments/crowdnav_v{1,4,5}_*` **missing** on this machine (gitignored); export records `skipped_missing_source` in `data/EXPORT_SUMMARY.json`. Results reports under `results/crowdnav_v*` remain.
- Upstream CrowdNav++ URL/commit verified against `env/crowdnav`.

## Open issues

- Full fresh-clone acceptance without a preinstalled gymnasium/torch stack was exercised via `tests/test_optional_rl_imports.py` and gymnasium `importorskip` on env/feature tests; CI installs `.[dev]` only.
- Historical Persian CrowdNav report path updated to `rsr/...` (one line).
- Thin legacy `requirements.txt` still present; prefer `pip install -e ".[rl,dev]"`.
