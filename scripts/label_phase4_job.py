#!/usr/bin/env python3
"""Label one Phase-4 candidate seed with PPO (container worker entrypoint).

Pilot path uses a fixed parametric RewardSpec so the container job does not
depend on LLM code loading yet. Full code-reward PPO wiring follows once the
dry-run host loop is green. Never reads LLM API keys from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert_no_secrets() -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "HF_TOKEN"):
        if os.environ.get(key):
            raise RuntimeError(f"secret {key} must not be set inside the labeler")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    _assert_no_secrets()
    out_dir = ROOT / "experiments" / args.run_id / "ppo_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.max_jobs <= 0:
        payload = {
            "status": "dry_run_skip",
            "run_id": args.run_id,
            "seed": args.seed,
        }
        (out_dir / "pilot_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    # Real PPO: reuse parametric trainer with a simple balanced spec.
    import yaml

    from rsr.rewards.reward_function import RewardSpec
    from rsr.training.parallel_run_jobs import phase1_train_job

    def _load(path: Path) -> dict:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(path)
        return data

    env_config = _load(ROOT / "configs" / "environment.yaml")
    ppo_config = _load(ROOT / "configs" / "ppo.yaml")
    eval_config = _load(ROOT / "configs" / "evaluation.yaml")
    # Keep pilot short only when env RSR_PILOT_TIMESTEPS is set (tests); else full config.
    pilot_ts = os.environ.get("RSR_PILOT_TIMESTEPS")
    if pilot_ts:
        ppo_config = dict(ppo_config)
        ppo_config["total_timesteps"] = int(pilot_ts)
        early = dict(ppo_config.get("early_behavioral") or {})
        early["enabled"] = False
        ppo_config["early_behavioral"] = early

    spec = RewardSpec(
        candidate_id=f"{args.run_id}_pilot",
        family_id="balanced",
        weights={
            "progress": 1.0,
            "goal": 10.0,
            "collision": 10.0,
            "obstacle": 0.5,
            "smoothness": 0.1,
            "time": 0.01,
        },
        scales={
            "progress": 1.0,
            "goal": 1.0,
            "collision": 1.0,
            "obstacle": 1.0,
            "smoothness": 1.0,
            "time": 1.0,
        },
    )
    run_dir = out_dir / f"seed_{args.seed}"
    job = {
        "reward_spec": spec.to_dict(),
        "env_config": env_config,
        "ppo_config": ppo_config,
        "eval_config": eval_config,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "candidate_id": spec.candidate_id,
        "family_id": spec.family_id,
        "resume": True,
    }
    print(f"starting PPO pilot job seed={args.seed} timesteps={ppo_config['total_timesteps']}")
    result = phase1_train_job(job)
    (out_dir / "pilot_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": result.get("training_status", "unknown"), "run_dir": str(run_dir)}, indent=2))
    return 0 if result.get("training_status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
