#!/usr/bin/env python3
"""Estimate Phase 4 candidate/lineage counts and wall-hours (no PPO launched)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Reference timings from prior campaigns (minutes per candidate×seed job).
REF_MIN_PER_JOB = {
    "2d": 2.0,  # ~50k PPO on RewardNavigation2D
    "irsim": 3.5,  # ~50k PPO static obstacles
    "crowdnav": 6.73,  # v5 reference at 150k steps
}

# Generation design from Phase 4 prompt.
N_PROMPTS = 6
TEMPERATURES = (0.3, 0.7, 1.0)
N_PER_PROMPT_TEMP = 16  # generation 0
N_SEEDS = 3
HELD_OUT_PROMPTS = 2
MIN_HELD_OUT_LINEAGES = 20
HELD_OUT_LINEAGE_FRAC = 0.20


def gen0_candidates() -> int:
    return N_PROMPTS * len(TEMPERATURES) * N_PER_PROMPT_TEMP


def lineages_for_gate2(min_held_out: int = MIN_HELD_OUT_LINEAGES, frac: float = HELD_OUT_LINEAGE_FRAC) -> dict:
    # Need >= max(min_held_out, frac * total) held-out lineages.
    # If we hold out 20%, total >= min_held_out / 0.20.
    total_by_frac = int(math.ceil(min_held_out / frac))
    total = max(total_by_frac, min_held_out + 1)
    held = max(min_held_out, int(math.ceil(frac * total)))
    # Bump total until held-out count from frac meets min.
    while int(math.floor(frac * total)) < min_held_out and int(math.ceil(frac * total)) < min_held_out:
        total += 1
        held = max(min_held_out, int(math.ceil(frac * total)))
    held = max(min_held_out, int(math.ceil(frac * total)))
    return {
        "min_held_out_lineages": min_held_out,
        "held_out_fraction": frac,
        "total_lineages_needed": total,
        "held_out_lineages": held,
        "train_lineages": total - held,
    }


def wall_hours(n_candidates: int, env: str, n_seeds: int = N_SEEDS, n_workers: int = 4) -> dict:
    minutes = REF_MIN_PER_JOB[env] * n_candidates * n_seeds
    hours_serial = minutes / 60.0
    hours_parallel = hours_serial / max(n_workers, 1)
    return {
        "env": env,
        "n_candidates": n_candidates,
        "n_seeds": n_seeds,
        "n_jobs": n_candidates * n_seeds,
        "min_per_job": REF_MIN_PER_JOB[env],
        "n_workers": n_workers,
        "wall_hours_serial": round(hours_serial, 2),
        "wall_hours_parallel": round(hours_parallel, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--generations", type=int, default=3, help="gen0 + mutation generations")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "phase4_sizing.md")
    p.add_argument("--json-out", type=Path, default=ROOT / "results" / "phase4_sizing.json")
    args = p.parse_args()

    g0 = gen0_candidates()
    # Subsequent generations: mutate top-2 parents → assume ~16 new cands per prompt/temp
    # Simplified: each extra generation adds another gen0-sized cohort (upper bound).
    n_cand = g0 * max(args.generations, 1)
    # Target floor from prompt: >=300 candidates × 3 seeds per env
    n_cand = max(n_cand, 300)

    lin = lineages_for_gate2()
    # One lineage per generation-0 (prompt, temperature, slot) family of mutations.
    # Conservative: need at least lin['total_lineages_needed'] distinct lineages.
    n_lineages = max(lin["total_lineages_needed"], N_PROMPTS * len(TEMPERATURES))

    envs = ["2d", "irsim", "crowdnav"]
    costs = [wall_hours(n_cand, e, n_workers=args.workers) for e in envs]
    payload = {
        "design": {
            "n_prompts": N_PROMPTS,
            "temperatures": list(TEMPERATURES),
            "n_per_prompt_temp_gen0": N_PER_PROMPT_TEMP,
            "gen0_candidates": g0,
            "generations": args.generations,
            "candidates_per_env_target": n_cand,
            "seeds": N_SEEDS,
            "held_out_prompts": HELD_OUT_PROMPTS,
        },
        "lineages_gate2": {**lin, "planned_lineages": n_lineages},
        "costs": costs,
        "total_wall_hours_parallel_all_envs": round(
            sum(c["wall_hours_parallel"] for c in costs), 2
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 4 sizing (no PPO launched)",
        "",
        "Computed by `scripts/estimate_phase4_cost.py`. Reference CrowdNav job time:",
        f"{REF_MIN_PER_JOB['crowdnav']} min/job at 150k steps (v5).",
        "",
        "## Candidate design",
        "",
        f"- Prompts: {N_PROMPTS}; temperatures: {list(TEMPERATURES)}; gen0 size per (prompt,T): {N_PER_PROMPT_TEMP}",
        f"- Generation-0 candidates: **{g0}**",
        f"- Generations planned: **{args.generations}** → target candidates/env: **{n_cand}** (≥300 floor)",
        f"- Seeds/candidate: **{N_SEEDS}** → jobs/env: **{n_cand * N_SEEDS}**",
        "",
        "## Lineages for Gate 2 (≥20 held-out)",
        "",
        f"- Held-out fraction: {HELD_OUT_LINEAGE_FRAC:.0%}",
        f"- Total lineages needed: **{lin['total_lineages_needed']}**",
        f"- Held-out lineages: **{lin['held_out_lineages']}**; train: **{lin['train_lineages']}**",
        f"- Planned lineages (≥ prompts×temps): **{n_lineages}**",
        f"- Also hold out **{HELD_OUT_PROMPTS}** prompts in `SPLITS.json`",
        "",
        "## Wall-hour estimates",
        "",
        f"Workers assumed: **{args.workers}**.",
        "",
        "| env | candidates | jobs | min/job | wall-h serial | wall-h parallel |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in costs:
        lines.append(
            f"| {c['env']} | {c['n_candidates']} | {c['n_jobs']} | {c['min_per_job']} | "
            f"{c['wall_hours_serial']} | {c['wall_hours_parallel']} |"
        )
    lines += [
        "",
        f"**Total parallel wall-hours (all envs, sequential envs):** "
        f"{payload['total_wall_hours_parallel_all_envs']}",
        "",
        "Env order: 2D → IR-SIM → CrowdNav++.",
        "",
        "## Notes",
        "",
        "- Dry-run / `--max-jobs 1` pilots only until the user launches full labeling.",
        "- Audit quota (10% of Tier-0 rejects) adds a small job overhead not counted above.",
        f"- Machine-readable copy: `{args.json_out.relative_to(ROOT).as_posix()}`",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
