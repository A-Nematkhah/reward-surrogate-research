"""Guards: 2D / CrowdNav / IR-SIM must not share roots or leak config."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("irsim")

from scripts import run_quality_pipeline as qp
from rsr.environment.navigation_2d import make_env
from rsr.utils.config import PROJECT_ROOT, load_merged_configs


def test_versioned_profiles_use_distinct_prefixes() -> None:
    prefixes = {p["prefix"] for p in qp.VERSIONED_ENVS.values()}
    assert prefixes == {"crowdnav", "irsim"}
    env_ids = {p["env_id"] for p in qp.VERSIONED_ENVS.values()}
    assert env_ids == {"CrowdNavPlusPlus-v0", "IRSimNav-v0"}
    reward_cfgs = {p["reward_config"] for p in qp.VERSIONED_ENVS.values()}
    assert len(reward_cfgs) == 2


def test_resolve_roots_do_not_collide() -> None:
    cn = qp.resolve_versioned_roots(
        qp.VERSIONED_ENVS["crowdnav"],
        run_id="iso_cn",
        force_new_run=False,
        dry_run=True,
    )
    ir = qp.resolve_versioned_roots(
        qp.VERSIONED_ENVS["irsim"],
        run_id="iso_ir",
        force_new_run=False,
        dry_run=True,
    )
    assert cn[1] == "experiments/crowdnav_iso_cn"
    assert ir[1] == "experiments/irsim_iso_ir"
    assert cn[2] == "results/crowdnav_iso_cn"
    assert ir[2] == "results/irsim_iso_ir"
    assert cn[1] != ir[1]
    assert "phase1" not in cn[1] and "phase1" not in ir[1]


def test_irsim_ignores_shared_2d_robot_radius() -> None:
    cfg = dict(load_merged_configs(names=["environment"])["environment"])
    cfg["env_id"] = "IRSimNav-v0"
    cfg["robot"] = dict(cfg.get("robot") or {})
    cfg["robot"]["radius"] = 0.15  # 2D default — must not win
    cfg.setdefault("irsim", {})
    cfg["irsim"]["robot_radius"] = 0.2
    cfg["irsim"]["world_yaml"] = str(
        PROJECT_ROOT / "env" / "worlds" / "irsim_pipeline_nav.yaml"
    )
    env = make_env(cfg)
    try:
        assert abs(float(env.robot_radius) - 0.2) < 1e-6
        assert bool(env.workspace.centered) is False
    finally:
        env.close()


def test_make_env_routes_without_cross_import_side_effects() -> None:
    cfg = dict(load_merged_configs(names=["environment"])["environment"])
    # Default stays 2D.
    env2d = make_env(cfg)
    try:
        assert type(env2d).__name__ == "RewardNavigation2D"
        assert abs(float(env2d.robot_radius) - 0.15) < 1e-6
    finally:
        env2d.close()

    ir_cfg = dict(cfg)
    ir_cfg["env_id"] = "IRSimNav-v0"
    ir_cfg.setdefault("irsim", {})
    ir_cfg["irsim"]["world_yaml"] = str(
        PROJECT_ROOT / "env" / "worlds" / "irsim_pipeline_nav.yaml"
    )
    env_ir = make_env(ir_cfg)
    try:
        assert type(env_ir).__name__ == "IRSimNavEnv"
    finally:
        env_ir.close()

    # 2D still works after IR-SIM boot (no permanent config mutation).
    env2d_b = make_env(cfg)
    try:
        assert type(env2d_b).__name__ == "RewardNavigation2D"
    finally:
        env2d_b.close()


def test_irsim_boot_does_not_shadow_project_tests_module() -> None:
    # Ensure importing IR-SIM adapter does not make ``import tests`` resolve to
    # env/ir-sim/tests.
    from rsr.environment import irsim_env as _irsim_env

    _irsim_env._boot_irsim()
    # Project tests package may not be a regular importable package; the
    # critical check is that env/ir-sim/tests is not preferred via sys.path.
    import sys

    irsim_tests = str((PROJECT_ROOT / "env" / "ir-sim").resolve())
    assert irsim_tests not in sys.path
