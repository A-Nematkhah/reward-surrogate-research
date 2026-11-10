#!/usr/bin/env python3
"""Run random-policy episodes in RewardNavigation2D and save visualizations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.environment.navigation_2d import make_env
from rsr.environment.visualization import plot_trajectory
from rsr.utils.config import load_named_config
from rsr.utils.logging import get_logger, setup_logging
from rsr.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of random-policy episodes to run",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "env_demo",
        help="Directory for trajectory plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("env_demo")
    seed_everything(args.seed)

    config = load_named_config("environment", config_dir=PROJECT_ROOT / "configs")
    env = make_env(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        reason = info.get("termination_reason", "unknown")
        out_path = args.out_dir / f"episode_{ep:02d}_{reason}.png"
        plot_trajectory(
            env,
            save_path=out_path,
            title=f"Episode {ep} | {reason} | R={total_reward:.2f} | steps={steps}",
        )
        logger.info(
            "Episode %d: reason=%s steps=%d return=%.3f path=%.2f obstacles=%d",
            ep,
            reason,
            steps,
            total_reward,
            info.get("path_length", float("nan")),
            info.get("n_obstacles", len(env.obstacles)),
        )
        logger.info("  saved %s", out_path)

    env.close()
    logger.info("Demo complete. Plots in %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
