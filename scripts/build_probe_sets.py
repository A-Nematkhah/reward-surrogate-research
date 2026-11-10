#!/usr/bin/env python3
"""Build frozen probe NPZ sets from real environments.

Synthetic probes belong only under tests/fixtures/probes/ (see --write-fixtures).
``--from-env`` fails loudly if the requested backend is unavailable — no silent
fallback to synthetic for data/probes/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.probes import (
    REQUIRED_POLICIES,
    assert_required_policies,
    episodes_to_batch,
    make_synthetic_probe_batch,
    save_probe_npz,
    stable_env_seed,
)


def _load_env_config() -> dict:
    path = ROOT / "configs" / "environment.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_2d(n_per_policy: int, seed: int, max_steps: int):
    from rsr.features.probe_trajectories import generate_probe_set

    env_config = _load_env_config()
    probe_config = {
        "seed": seed,
        "n_trajectories_per_policy": n_per_policy,
        "max_steps": max_steps,
        "policies": list(REQUIRED_POLICIES),
    }
    return generate_probe_set(env_config, probe_config)


def _build_irsim(n_per_policy: int, seed: int, max_steps: int):
    try:
        from rsr.environment.irsim_env import IRSimNavEnv  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "IR-SIM backend unavailable; install/configure IR-SIM before "
            "--from-env --env irsim"
        ) from exc
    from rsr.features.probe_trajectories import generate_probe_set

    env_config = _load_env_config()
    env_config = dict(env_config)
    irsim_cfg = dict(env_config.get("irsim") or {})
    env_config["env_id"] = irsim_cfg.get("env_id", "IRSimNav-v0")
    probe_config = {
        "seed": seed,
        "n_trajectories_per_policy": n_per_policy,
        "max_steps": max_steps,
        "policies": list(REQUIRED_POLICIES),
    }
    try:
        return generate_probe_set(env_config, probe_config)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build IR-SIM probes from env: {exc}. "
            "No synthetic fallback for --from-env."
        ) from exc


def _build_crowdnav(n_per_policy: int, seed: int, max_steps: int):
    try:
        from rsr.environment.crowdnav_env import CrowdNavPlusPlusEnv  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "CrowdNav++ backend unavailable; install/configure CrowdNav before "
            "--from-env --env crowdnav"
        ) from exc
    from rsr.features.probe_trajectories import generate_probe_set

    env_config = _load_env_config()
    env_config = dict(env_config)
    env_config["env_id"] = "CrowdNavPlusPlus-v0"
    probe_config = {
        "seed": seed,
        "n_trajectories_per_policy": n_per_policy,
        "max_steps": max_steps,
        "policies": list(REQUIRED_POLICIES),
    }
    try:
        return generate_probe_set(env_config, probe_config)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build CrowdNav++ probes from env: {exc}. "
            "No synthetic fallback for --from-env."
        ) from exc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["2d", "crowdnav", "irsim", "all"], default="2d")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "probes")
    p.add_argument("--n-per-policy", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--from-env",
        action="store_true",
        help="Build from real env rollouts (required for data/probes). Fails if env missing.",
    )
    p.add_argument(
        "--write-fixtures",
        action="store_true",
        help="Write synthetic probes only under tests/fixtures/probes/.",
    )
    args = p.parse_args()

    env_map = {
        "2d": ("navigation_2d", _build_2d),
        "crowdnav": ("crowdnav", _build_crowdnav),
        "irsim": ("irsim", _build_irsim),
    }
    selected = list(env_map.items()) if args.env == "all" else [ (args.env, env_map[args.env]) ]

    if args.write_fixtures:
        out = ROOT / "tests" / "fixtures" / "probes"
        if args.dry_run:
            for key, (env_id, _) in selected:
                print(f"would write fixture {out / (env_id + '.npz')}")
            return 0
        out.mkdir(parents=True, exist_ok=True)
        for key, (env_id, _) in selected:
            seed = stable_env_seed(env_id, args.seed)
            batch = make_synthetic_probe_batch(
                env_id, n_per_policy=args.n_per_policy, seed=seed
            )
            path = out / f"{env_id}.npz"
            digest = save_probe_npz(batch, path)
            print(f"fixture {path} hash={digest} seed={seed} n={len(batch.transitions)}")
        return 0

    if not args.from_env:
        print(
            "ERROR: data/probes requires --from-env (real rollouts). "
            "For synthetic test fixtures use --write-fixtures.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        for key, (env_id, _) in selected:
            seed = stable_env_seed(env_id, args.seed)
            print(f"would build {args.out_dir / (env_id + '.npz')} from-env seed={seed}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, (env_id, builder) in selected:
        seed = stable_env_seed(env_id, args.seed)
        episodes = builder(args.n_per_policy, seed, args.max_steps)
        batch = episodes_to_batch(
            env_id,
            episodes,
            meta={
                "synthetic": False,
                "seed": seed,
                "base_seed": args.seed,
                "n_per_policy": args.n_per_policy,
                "max_steps": args.max_steps,
                "source": f"from-env:{key}",
            },
        )
        assert_required_policies(batch)
        path = args.out_dir / f"{env_id}.npz"
        digest = save_probe_npz(batch, path)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(batch.meta, indent=2), encoding="utf-8")
        print(
            f"wrote {path} hash={digest} seed={seed} n={len(batch.transitions)} "
            f"policies={batch.meta.get('policies')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
