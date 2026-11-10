#!/usr/bin/env python3
"""Build frozen probe NPZ sets for 2D / CrowdNav++ / IR-SIM.

Default path writes synthetic probes (no env deps). For real env rollouts,
pass --from-env (requires local env extras; may be slow).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.probes import (
    ensure_default_synthetic_probes,
    make_synthetic_probe_batch,
    save_probe_npz,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["2d", "crowdnav", "irsim", "all"], default="all")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "probes")
    p.add_argument("--n-per-policy", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--from-env",
        action="store_true",
        help="Attempt real env rollouts (not used by default; prints command only if dry-run).",
    )
    args = p.parse_args()

    env_map = {
        "2d": "navigation_2d",
        "crowdnav": "crowdnav",
        "irsim": "irsim",
    }
    envs = list(env_map.values()) if args.env == "all" else [env_map[args.env]]

    if args.from_env:
        print(
            "Real env probe collection is optional. Use synthetic probes for CI. "
            "Example (2D, when ready):\n"
            "  python scripts/build_probe_sets.py --env 2d --from-env\n"
            "Currently falling back to synthetic NPZ."
        )

    if args.dry_run:
        for env_id in envs:
            print(f"would write {args.out_dir / (env_id + '.npz')} (synthetic)")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for env_id in envs:
        batch = make_synthetic_probe_batch(
            env_id, n_per_policy=args.n_per_policy, seed=args.seed + hash(env_id) % 1000
        )
        path = args.out_dir / f"{env_id}.npz"
        digest = save_probe_npz(batch, path)
        print(f"wrote {path} hash={digest} n={len(batch.transitions)}")

    # Also ensure defaults exist
    ensure_default_synthetic_probes(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
