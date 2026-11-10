"""Import paths that must work when torch / stable-baselines3 are absent."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


_BLOCK = frozenset({"torch", "stable_baselines3"})


@pytest.fixture
def block_rl_modules(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def _guard(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".", 1)[0]
        if root in _BLOCK:
            raise ImportError(f"blocked optional dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    # Drop already-loaded modules so re-imports hit the guard.
    for key in list(sys.modules):
        if key == "torch" or key.startswith("torch."):
            monkeypatch.setitem(sys.modules, key, None)
            sys.modules.pop(key, None)
        if key == "stable_baselines3" or key.startswith("stable_baselines3."):
            sys.modules.pop(key, None)

    monkeypatch.setattr(builtins, "__import__", _guard)
    # Force re-import of packages under test.
    for mod in [
        "rsr.utils.reproducibility",
        "rsr.features.early_behavioral",
        "rsr.evaluation.evaluator",
        "rsr.evaluation.metrics",
        "rsr.features",
        "rsr.evaluation",
        "rsr.surrogate",
    ]:
        sys.modules.pop(mod, None)
    yield
    for mod in list(sys.modules):
        if mod.startswith("rsr."):
            sys.modules.pop(mod, None)


def test_core_modules_import_without_torch_or_sb3(block_rl_modules) -> None:
    metrics = importlib.import_module("rsr.evaluation.metrics")
    features_pkg = importlib.import_module("rsr.features")
    evaluation_pkg = importlib.import_module("rsr.evaluation")
    surrogate_pkg = importlib.import_module("rsr.surrogate")
    repro = importlib.import_module("rsr.utils.reproducibility")

    assert metrics.METRIC_NAMES
    assert features_pkg.__doc__
    assert evaluation_pkg.__doc__
    assert surrogate_pkg.__doc__
    meta = repro.seed_everything(0)
    assert meta["has_torch"] is False
