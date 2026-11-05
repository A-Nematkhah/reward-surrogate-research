#!/usr/bin/env python3
"""Smoke-run the vendored CrowdNav++ simulator.

Official ORCA humans need Python-RVO2 (``rvo2``), which has no Windows wheel.
This entry point uses the repo's social-force humans and an SRNN action clip
so a few episodes can run in the project environment without that package.

From the repo root:

    python scripts/run_crowdnav.py
    python scripts/run_crowdnav.py --episodes 2 --humans 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROWDNAV_ROOT = PROJECT_ROOT / "third_party" / "CrowdNavPlusPlus"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--humans", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train",
        action="store_true",
        help="Train a short PPO run through the project reward on CrowdNav++",
    )
    p.add_argument("--timesteps", type=int, default=256)
    return p.parse_args()


def _train(args: argparse.Namespace) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.rewards.reward_function import RewardSpec
    from src.training.ppo_trainer import train_candidate_seed
    from src.utils.config import load_merged_configs

    configs = load_merged_configs()
    env_config = configs["environment"]
    env_config["env_id"] = "CrowdNavPlusPlus-v0"
    env_config["episode"] = {"max_steps": 20}
    env_config["crowdnav"] = {"human_num": int(args.humans)}
    ppo_config = configs["ppo"]
    ppo_config.update(
        {
            "total_timesteps": int(args.timesteps),
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "device": "cpu",
            "early_behavioral": {"enabled": False},
        }
    )
    eval_config = configs["evaluation"]
    eval_config["n_eval_episodes"] = 2
    eval_config["save_episode_details"] = False
    spec = RewardSpec(
        candidate_id="crowdnav_smoke",
        family_id="balanced",
        weights={
            "progress": 1.5,
            "goal": 10.0,
            "collision": -10.0,
            "obstacle": -1.0,
            "smoothness": -0.1,
            "time": -0.01,
        },
        scales={name: 1.0 for name in (
            "progress", "goal", "collision", "obstacle", "smoothness", "time",
            "energy", "risk", "potential", "nonlinear_progress", "distance",
        )},
    )
    run_dir = PROJECT_ROOT / "experiments" / "crowdnav_smoke" / "seed_0"
    status = train_candidate_seed(
        spec,
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        seed=int(args.seed),
        run_dir=run_dir,
        resume_if_complete=False,
    )
    metrics = status.get("metrics") or {}
    print(
        f"crowdnav train status={status.get('training_status')} "
        f"success_rate={metrics.get('success_rate')} "
        f"collision_rate={metrics.get('collision_rate')}"
    )
    return 0 if status.get("training_status") == "completed" else 1


def main() -> int:
    args = parse_args()
    if args.train:
        return _train(args)
    if not CROWDNAV_ROOT.is_dir():
        print(f"CrowdNav++ is not cloned at {CROWDNAV_ROOT}", file=sys.stderr)
        return 1

    # CrowdNav's Config() calls argparse on import. Keep its defaults.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(CROWDNAV_ROOT))

    import gym

    import crowd_sim  # noqa: F401  registers gym ids
    from crowd_nav.configs.config import Config

    logging.basicConfig(level=logging.INFO)
    config = Config()
    config.sim.predict_method = "none"
    config.env.use_wrapper = False
    config.sim.human_num = int(args.humans)
    config.sim.render = False
    config.env.randomize_attributes = False
    config.humans.policy = "orca"
    config.humans.random_goal_changing = False
    config.humans.end_goal_changing = False
    config.robot.policy = "srnn"
    config.robot.visible = True

    env = gym.make("CrowdSimVarNum-v0")
    env.configure(config)
    env.thisSeed = int(args.seed)
    env.nenv = 1
    env.phase = "train"
    if hasattr(env, "seed"):
        env.seed(int(args.seed))

    successes = collisions = timeouts = 0
    for episode in range(int(args.episodes)):
        obs = env.reset(phase="train")
        done = False
        steps = 0
        info = None
        while not done and steps < int(args.max_steps):
            # Holonomic (vx, vy) toward the goal. SRNN.clip_action scales it.
            robot = env.robot
            dx = float(robot.gx - robot.px)
            dy = float(robot.gy - robot.py)
            obs, _reward, done, info = env.step(np.array([dx, dy], dtype=np.float32))
            steps += 1
            del obs
        if isinstance(info, dict) and "info" in info:
            info = info["info"]
        kind = type(info).__name__ if info is not None else "Unknown"
        if kind == "ReachGoal":
            successes += 1
        elif kind == "Collision":
            collisions += 1
        else:
            timeouts += 1
        print(f"episode={episode} steps={steps} info={kind}")

    print(
        f"done episodes={args.episodes} success={successes} "
        f"collision={collisions} other={timeouts}"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
