#!/usr/bin/env python3
"""Phase 4 mini host loop: generate → triage → select → label (mock or real)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.host_loop import LoopConfig, run_loop  # noqa: E402
from reward_triage.labeling import load_default_configs  # noqa: E402
from reward_triage.llm_client import make_llm_client  # noqa: E402
from reward_triage.model import SurrogateModel  # noqa: E402

TASK_2D = (
    "2D continuous navigation: a disk robot moves in a bounded workspace with "
    "circular obstacles toward a goal disk. Episodes end on success, collision, "
    "out-of-bounds, or timeout."
)


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _coerce_mode(value: object) -> str:
    if value is False or value is None:
        return "off"
    if value is True:
        return "active"
    text = str(value).strip().lower()
    if text not in ("off", "shadow", "active"):
        raise ValueError(f"invalid mode: {value!r}")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4_2d.yaml")
    p.add_argument("--mode", choices=["off", "shadow", "active"], default=None)
    p.add_argument("--dry-run", action="store_true", help="Mock labels; no PPO")
    p.add_argument(
        "--real-ppo",
        action="store_true",
        help="Run real PPO labeling (overrides default dry-run safety)",
    )
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--provider", type=str, default=None, help="local|mock|xai")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--no-early", action="store_true", help="Disable early feature collection")
    p.add_argument(
        "--surrogate",
        type=Path,
        default=None,
        help="Path to Phase-5 artifact dir (model.joblib); enables active filtering after m_min",
    )
    p.add_argument("--m-min", type=int, default=None)
    p.add_argument("--n-generations", type=int, default=None)
    p.add_argument("--n-per-generation", type=int, default=None)
    p.add_argument(
        "--no-adapt",
        action="store_true",
        help="Disable per-generation surrogate adapt",
    )
    p.add_argument(
        "--no-adapt-prior",
        action="store_true",
        help="Adapt only on this run's labels (no phase1_108 prior blend)",
    )
    p.add_argument(
        "--print-real-ppo-cmd",
        action="store_true",
        help="Print pilot commands and exit",
    )
    args = p.parse_args()

    cfg_raw = _load_yaml(args.config)
    run_id = args.run_id or str(cfg_raw.get("run_id", "phase4_2d_v1"))
    llm_cfg = dict(cfg_raw.get("llm") or {})
    provider = args.provider or str(llm_cfg.get("provider", "local"))
    model = args.model or str(llm_cfg.get("model", "local-corpus"))

    if args.print_real_ppo_cmd:
        print("# Wired host-loop real PPO (short pilot):")
        print(
            "$env:RSR_PILOT_TIMESTEPS='2000'; "
            f"python scripts/mini_host_loop.py --provider local --real-ppo "
            f"--max-jobs 1 --run-id {run_id}_wire"
        )
        return 0

    dry_run = True
    if args.real_ppo:
        dry_run = False
    if args.dry_run:
        dry_run = True

    env_config, ppo_config, eval_config = load_default_configs(
        env_path=ROOT / str(cfg_raw.get("env_config", "configs/environment.yaml")),
        ppo_path=ROOT / str(cfg_raw.get("ppo_config", "configs/ppo.yaml")),
        eval_path=ROOT / str(cfg_raw.get("eval_config", "configs/evaluation.yaml")),
    )

    store_path = ROOT / "experiments" / run_id / "label_store.jsonl"
    surrogate = None
    if args.surrogate is not None:
        surrogate = SurrogateModel.load(args.surrogate)
        print(f"loaded surrogate from {args.surrogate} (n_train={surrogate.meta.n_train if surrogate.meta else '?'})")

    loop_cfg = LoopConfig(
        run_id=run_id,
        env_id=str(cfg_raw.get("env_id", "RewardNavigation2D-v0")),
        env_config_id=str(cfg_raw.get("env_config_id", "2d")),
        host=str(cfg_raw.get("host", "mini_host")),
        n_per_generation=int(
            args.n_per_generation
            if args.n_per_generation is not None
            else cfg_raw.get("n_per_generation", 16)
        ),
        n_generations=int(
            args.n_generations if args.n_generations is not None else cfg_raw.get("n_generations", 3)
        ),
        n_seeds=int(cfg_raw.get("n_seeds", 3)),
        seeds=tuple(int(s) for s in (cfg_raw.get("seeds") or [0, 1, 2])),
        mode=_coerce_mode(args.mode if args.mode is not None else cfg_raw.get("mode", "off")),
        m_min=int(args.m_min if args.m_min is not None else cfg_raw.get("m_min", 24)),
        k_frac=float(cfg_raw.get("k_frac", 0.5)),
        audit_frac=float(cfg_raw.get("audit_frac", 0.15)),
        temperature=float(cfg_raw.get("temperature", 0.7)),
        prompt_id=str(cfg_raw.get("prompt_id", "gen0_default")),
        probe_path=str(ROOT / cfg_raw.get("probe_path", "data/probes/navigation_2d.npz")),
        store_path=str(store_path),
        dry_run=dry_run,
        max_jobs=args.max_jobs,
        ppo_timesteps=int(ppo_config.get("total_timesteps", 50_000)),
        enable_early=not args.no_early,
        experiments_root=str(ROOT / "experiments"),
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        surrogate=surrogate,
        auto_adapt=not args.no_adapt,
        adapt_use_prior=not args.no_adapt_prior,
    )

    client = make_llm_client(provider, model)
    prompt_gen0 = (ROOT / "configs" / "llm_prompts" / "gen0.txt").read_text(encoding="utf-8")
    prompt_mutate = (ROOT / "configs" / "llm_prompts" / "mutate.txt").read_text(encoding="utf-8")

    if not dry_run:
        print(
            f"REAL PPO labeling enabled (early={loop_cfg.enable_early}, "
            f"max_jobs={loop_cfg.max_jobs}, timesteps={loop_cfg.ppo_timesteps})"
        )

    summary = run_loop(
        loop_cfg,
        client,
        prompt_gen0=prompt_gen0,
        prompt_mutate=prompt_mutate,
        task_text=TASK_2D,
    )
    print(json.dumps(summary, indent=2))
    if summary["validation_errors"]:
        print("VALIDATION FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
