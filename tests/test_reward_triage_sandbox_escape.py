"""Sandbox escape attempts must fail with exec_error."""

from __future__ import annotations

import os

from reward_triage.executor import run_candidate_on_probes
from reward_triage.probes import make_synthetic_probe_batch


def _tr():
    return make_synthetic_probe_batch("navigation_2d", n_per_policy=1, seed=0).transitions[:1]


def test_escape_typing_sys_blocked():
    code = """
import typing
def compute_reward(transition):
    return float(len(typing.sys.modules))
"""
    result = run_candidate_on_probes(code, _tr(), timeout_s=5.0)
    assert result.report.status == "exec_error"


def test_escape_numpy_save_blocked():
    code = """
import numpy as np
def compute_reward(transition):
    np.save('escape_probe.npy', np.asarray([1.0, 2.0]))
    return 0.0
"""
    result = run_candidate_on_probes(code, _tr(), timeout_s=5.0)
    assert result.report.status == "exec_error"


def test_escape_dunder_class_mro_blocked():
    code = """
def compute_reward(transition):
    return float(len(().__class__.__mro__))
"""
    result = run_candidate_on_probes(code, _tr(), timeout_s=5.0)
    assert result.report.status == "exec_error"


def test_escape_env_var_read_blocked():
    os.environ["REWARD_TRIAGE_TEST_SECRET"] = "should-not-leak"
    code = """
def compute_reward(transition):
    import os
    return float(len(os.environ.get('REWARD_TRIAGE_TEST_SECRET', '')))
"""
    try:
        result = run_candidate_on_probes(code, _tr(), timeout_s=5.0)
        assert result.report.status == "exec_error"
    finally:
        os.environ.pop("REWARD_TRIAGE_TEST_SECRET", None)


def test_escape_os_via_builtins_blocked():
    code = """
def compute_reward(transition):
    return float(len(__builtins__))
"""
    result = run_candidate_on_probes(code, _tr(), timeout_s=5.0)
    # __builtins__ name / attr blocked by AST
    assert result.report.status == "exec_error"
