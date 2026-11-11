# Phase 4 report — Live pipeline scaffolding (2D first)

**Status: scaffolding complete; full 2D PPO campaign not launched.**

## Design locked

- Per-environment labels only; no cross-env transfer; no pre-collected dataset.
- One env at a time: **2D only** until user reviews results.
- Config: `configs/phase4_2d.yaml` — N=16 × G=3 = **48** candidates/run; 3 seeds; mode `off` warm-up.
- LLM: default **`local`** hand-written corpus (LLM-like: only “try-to-be-good” styles). `xai` when key exists.
- **Operating feature set (locked): `early+math+behavioral`** — primary surrogate input. Probe LOFO within=0.641 on 108. Requires early PPO checkpoints before scoring; `mb` alone is not the scorer.
- Isolation: **Docker deferred**; local Python workers for now.
- Sizing: `docs/phase4_sizing.md` — ~3.6 parallel wall-hours for 3×2D runs @ 4 workers.

## Delivered

| Piece | Path |
| --- | --- |
| Schema v2 store | `reward_triage/store.py` |
| Host loop | `reward_triage/host_loop.py` |
| LLM client (mock + xAI) | `reward_triage/llm_client.py` |
| CLI | `scripts/mini_host_loop.py` |
| PPO pilot entry | `scripts/label_phase4_job.py` |
| Prompts | `configs/llm_prompts/{gen0,mutate}.txt` |
| Tests | `tests/test_phase4_host_loop.py` |

## Acceptance checks

```text
python scripts/mini_host_loop.py --dry-run --max-jobs 1 --run-id phase4_2d_dry_pilot
# validators pass; label_store.jsonl + run_summary.json written

python scripts/mini_host_loop.py --print-real-ppo-cmd --run-id phase4_2d_v1
# prints (does not run):
# docker compose -f docker-compose.triage.yml run --rm triage-labeler \
#   python scripts/label_phase4_job.py --run-id phase4_2d_v1 --max-jobs 1
```

`pytest tests/test_phase4_host_loop.py` green.

## Not done yet (needs user go-ahead)

1. Local PPO pilot — done (`label_phase4_job.py`, 2000 steps completed).
2. **Code rewards wired** — `reward_triage/labeling.py` + `train_reward_fn_candidate_seed`; host loop `--real-ppo` runs PPO on `local`/LLM code, collects early features when budget allows.
3. Full 2D campaign: `python scripts/mini_host_loop.py --provider local --real-ppo --run-id phase4_2d_v1` (no `RSR_PILOT_TIMESTEPS`; long). Start with `--max-jobs` caps.
4. CrowdNav/IR-SIM deferred. Docker optional later.

## Open issues

- YAML unquoted `off` parses as boolean; config now quotes `"off"` and CLI coerces.
- Code-reward PPO path still TODO; host loop dry-run uses mock labels after Tier0/`mb`.
