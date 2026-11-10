"""Sandboxed execution of candidate reward code.

Platform note (VERIFY)
----------------------
On POSIX, optional resource limits can tighten memory. On Windows there is no
reliable RLIMIT equivalent in the stdlib, so isolation is weaker: we use
``multiprocessing`` with a wall-clock join timeout, an import allowlist, and
blocked network/file builtins inside the child. Do not treat this as a full
security sandbox.
"""

from __future__ import annotations

import ast
import hashlib
import multiprocessing as mp
import platform
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reward_triage.schema import ValidationReport

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_ALLOWLIST = frozenset({"math", "numpy", "typing", "collections", "builtins"})


@dataclass
class ExecResult:
    rewards: list[float]
    components: list[dict[str, float] | None]
    report: ValidationReport


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _blocked_open(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError("File I/O is blocked in the reward sandbox")


def _blocked_network(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError("Network access is blocked in the reward sandbox")


def _make_safe_globals(*, allow_torch: bool) -> dict[str, Any]:
    import builtins
    import math
    import typing

    import numpy as np

    allowed = {
        "math": math,
        "numpy": np,
        "np": np,
        "typing": typing,
        "__builtins__": {
            k: getattr(builtins, k)
            for k in (
                "abs",
                "min",
                "max",
                "sum",
                "len",
                "range",
                "enumerate",
                "zip",
                "map",
                "filter",
                "float",
                "int",
                "bool",
                "str",
                "list",
                "dict",
                "tuple",
                "set",
                "frozenset",
                "isinstance",
                "issubclass",
                "hasattr",
                "getattr",
                "setattr",
                "callable",
                "print",
                "True",
                "False",
                "None",
                "Exception",
                "ValueError",
                "TypeError",
                "RuntimeError",
                "KeyError",
                "IndexError",
                "ZeroDivisionError",
            )
            if hasattr(builtins, k)
        },
    }
    allowed["__builtins__"]["open"] = _blocked_open
    # Block common network entry points if present in builtins (usually not).
    for name in ("__import__",):
        # Replaced below with allowlisted import.
        pass

    def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        allow = set(DEFAULT_ALLOWLIST)
        if allow_torch:
            allow.add("torch")
        if root not in allow:
            raise ImportError(f"Import of '{name}' is not allowed in reward sandbox")
        return __import__(name, globals, locals, fromlist, level)

    allowed["__builtins__"]["__import__"] = _safe_import
    if allow_torch:
        try:
            import torch  # type: ignore

            allowed["torch"] = torch
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("torch requested but not importable") from exc
    return allowed


def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            raise TypeError("Async constructs are not allowed in reward code")


def _load_callable(code: str, *, allow_torch: bool) -> Any:
    _validate_ast(code)
    g = _make_safe_globals(allow_torch=allow_torch)
    exec(compile(code, "<reward_candidate>", "exec"), g, g)  # noqa: S102
    for name in ("compute_reward", "reward", "get_reward"):
        fn = g.get(name)
        if callable(fn):
            return fn
    raise AttributeError(
        "Candidate code must define compute_reward (or reward / get_reward)"
    )


def _call_reward(fn: Any, transition: Mapping[str, Any]) -> tuple[float, dict[str, float] | None]:
    out = fn(dict(transition))
    components: dict[str, float] | None = None
    if isinstance(out, tuple) and len(out) == 2:
        total, comps = out
        if comps is not None:
            components = {str(k): float(v) for k, v in dict(comps).items()}
        return float(total), components
    return float(out), None


def _worker(
    code: str,
    transitions: list[dict[str, Any]],
    allow_torch: bool,
    out_q: Any,
) -> None:
    t0 = time.perf_counter()
    try:
        fn = _load_callable(code, allow_torch=allow_torch)
        rewards: list[float] = []
        components: list[dict[str, float] | None] = []
        for tr in transitions:
            r, c = _call_reward(fn, tr)
            rewards.append(r)
            components.append(c)
        out_q.put(
            {
                "ok": True,
                "rewards": rewards,
                "components": components,
                "elapsed": time.perf_counter() - t0,
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface to parent as exec_error
        out_q.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed": time.perf_counter() - t0,
            }
        )


def run_candidate_on_probes(
    code: str,
    transitions: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    allow_torch: bool = False,
) -> ExecResult:
    """Execute candidate code on a batch of probe transitions in a child process."""
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    payload = [dict(t) for t in transitions]
    # Strip non-pickle-friendly objects if any (keep primitives / lists / arrays as lists)
    clean: list[dict[str, Any]] = []
    for tr in payload:
        item: dict[str, Any] = {}
        for k, v in tr.items():
            if hasattr(v, "tolist"):
                item[k] = v.tolist()
            elif isinstance(v, (list, tuple, dict, str, bool, int, float, type(None))):
                item[k] = v
            else:
                item[k] = str(v)
        clean.append(item)

    proc = ctx.Process(target=_worker, args=(code, clean, allow_torch, out_q))
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=float(timeout_s))
    wall = time.perf_counter() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        return ExecResult(
            rewards=[],
            components=[],
            report=ValidationReport(
                status="timeout",
                reasons=[f"exceeded wall-clock timeout of {timeout_s}s"],
                timings={"wall_s": wall, "platform": 0.0},
            ),
        )

    if out_q.empty():
        return ExecResult(
            rewards=[],
            components=[],
            report=ValidationReport(
                status="exec_error",
                reasons=["child process exited without result"],
                timings={"wall_s": wall},
            ),
        )

    msg = out_q.get()
    if not msg.get("ok"):
        return ExecResult(
            rewards=[],
            components=[],
            report=ValidationReport(
                status="exec_error",
                reasons=[str(msg.get("error", "unknown error"))],
                timings={"wall_s": wall, "child_s": float(msg.get("elapsed", 0.0))},
            ),
        )

    rewards = [float(x) for x in msg["rewards"]]
    components = msg["components"]
    return ExecResult(
        rewards=rewards,
        components=components,
        report=ValidationReport(
            status="ok",
            reasons=[],
            timings={
                "wall_s": wall,
                "child_s": float(msg.get("elapsed", 0.0)),
                "n_transitions": float(len(rewards)),
            },
        ),
    )


def platform_isolation_note() -> str:
    system = platform.system()
    if system == "Windows":
        return (
            "Windows: multiprocessing + join timeout only; no POSIX RLIMIT memory "
            "cap. Isolation is weaker than on Linux."
        )
    return "POSIX: multiprocessing + join timeout; optional RLIMIT not enabled by default."
