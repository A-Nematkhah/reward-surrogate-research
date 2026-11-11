#!/usr/bin/env python3
"""Estimate Phase 4 live-run PPO jobs and wall-hours (no PPO launched)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Reference timings from prior campaigns (minutes per candidate x seed job).
REF_MIN_PER_JOB = {
    "2d": 2.0,  # ~50k PPO on RewardNavigation2D
    "irsim": 3.5,  # ~50k PPO static obstacles
    "crowdnav": 6.73,  # v5 reference at 150k steps
}

# Live pipeline train-all / warm-up design (new Phase 4 pack).
N_PER_GEN = 16
N_GENERATIONS = 5
N_SEEDS = 3
N_WORKERS = 4
# Gate 2 CIs need repeated 2D runs. Other envs are scheduled only after 2D review.
N_RUNS_2D = 5
N_RUNS_OTHER = 1


def wall_hours(n_jobs: int, env: str, n_workers: int) -> dict:
    minutes = REF_MIN_PER_JOB[env] * n_jobs
    hours_serial = minutes / 60.0
    hours_parallel = hours_serial / max(n_workers, 1)
    return {
        "env": env,
        "n_jobs": n_jobs,
        "min_per_job": REF_MIN_PER_JOB[env],
        "n_workers": n_workers,
        "wall_hours_serial": round(hours_serial, 2),
        "wall_hours_parallel": round(hours_parallel, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--generations", type=int, default=N_GENERATIONS)
    p.add_argument("--n-per-gen", type=int, default=N_PER_GEN)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--runs-2d", type=int, default=N_RUNS_2D)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "phase4_sizing.md")
    p.add_argument("--json-out", type=Path, default=ROOT / "results" / "phase4_sizing.json")
    args = p.parse_args()

    jobs_run = args.n_per_gen * args.generations * args.seeds
    # Active now: 2D only. Other envs are reference estimates for later user go-ahead.
    active = wall_hours(jobs_run * args.runs_2d, "2d", args.workers)
    active["n_runs"] = args.runs_2d
    active["jobs_per_run"] = jobs_run
    active["candidates_per_run"] = args.n_per_gen * args.generations
    active["status"] = "active_now"

    later = []
    for env in ("irsim", "crowdnav"):
        c = wall_hours(jobs_run * N_RUNS_OTHER, env, args.workers)
        c["n_runs"] = N_RUNS_OTHER
        c["jobs_per_run"] = jobs_run
        c["candidates_per_run"] = args.n_per_gen * args.generations
        c["status"] = "deferred_until_user_go_ahead_after_2d_review"
        later.append(c)

    payload = {
        "design": {
            "n_per_generation": args.n_per_gen,
            "generations": args.generations,
            "seeds": args.seeds,
            "jobs_per_run": jobs_run,
            "candidates_per_run": args.n_per_gen * args.generations,
            "mode": "train-all / off warm-up (labels produced by the run)",
            "no_cross_env_transfer": True,
            "one_env_at_a_time": True,
            "active_env": "2d",
            "llm": {"provider": "xai", "model": "grok-4.3"},
            "isolation": "docker",
            "workers": args.workers,
            "runs_2d_for_gate2_ci": args.runs_2d,
            "policy": (
                "Run 2D to completion and review results before starting another env. "
                "Do not auto-chain CrowdNav++ or IR-SIM."
            ),
        },
        "active_now": active,
        "deferred_reference": later,
        "active_wall_hours_parallel": active["wall_hours_parallel"],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 4 sizing (live runs; no separate dataset)",
        "",
        "Computed by `scripts/estimate_phase4_cost.py`. Labels are a by-product of each",
        "live pipeline run (train-all / off warm-up). No cross-environment transfer.",
        "",
        "## Policy: one environment at a time",
        "",
        "1. Run **2D** to completion (including repeated runs for Gate 2 CIs).",
        "2. Review results with the user.",
        "3. Only then start **CrowdNav++** or **IR-SIM** from scratch — never auto-chain.",
        "",
        "## Runtime choices",
        "",
        "- LLM: **xAI / grok-4.3** (VERIFY exact API model id at client wiring)",
        "- Isolation: **Docker**",
        f"- Workers: **{args.workers}**",
        "",
        "## One live run (train-all)",
        "",
        f"- N={args.n_per_gen} candidates/generation × G={args.generations} × seeds={args.seeds}",
        f"- Candidates/run: **{args.n_per_gen * args.generations}**; jobs/run: **{jobs_run}**",
        "",
        "## Active now: 2D",
        "",
        f"- Independent runs for Gate 2 CIs: **{args.runs_2d}**",
        f"- Jobs: **{active['n_jobs']}**",
        f"- Wall-hours @ {args.workers} workers: **{active['wall_hours_parallel']}** parallel "
        f"({active['wall_hours_serial']} serial)",
        "",
        "## Deferred (reference only — do not launch until user go-ahead)",
        "",
        "| env | runs | jobs | min/job | wall-h parallel |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for c in later:
        lines.append(
            f"| {c['env']} | {c['n_runs']} | {c['n_jobs']} | {c['min_per_job']} | "
            f"{c['wall_hours_parallel']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Dry-run / `--max-jobs 1` pilots only until the user launches full labeling.",
        "- Audit sample (~15% after active) adds a small overhead not counted above.",
        f"- Machine-readable copy: `{args.json_out.relative_to(ROOT).as_posix()}`",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
