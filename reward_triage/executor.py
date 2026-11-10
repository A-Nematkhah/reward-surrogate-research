"""Sandboxed execution of candidate reward code.

Honesty note
------------
A Python-level sandbox is **not** a security boundary. Escape attempts are
blocked in depth (AST denylist, stripped env, numpy wrapper, no ``typing``),
but Phase 4 LLM-generated code must still run inside a container or
bwrap/firejail without network and without secrets (see INTEGRATION.md when
added). On Windows there is no reliable POSIX RLIMIT; isolation is weaker.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import math
import multiprocessing as mp
import os
import platform
import time
import traceback
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    import resource as _resource  # type: ignore[attr-defined]
except ImportError:  # Windows
    _resource = None

from reward_triage.schema import ValidationReport

DEFAULT_TIMEOUT_S = 10.0

# Import roots allowed in candidate code (typing intentionally excluded).
DEFAULT_ALLOWLIST = frozenset({"math"})

_FORBIDDEN_ATTR_NAMES = frozenset(
    {
        "sys",
        "os",
        "modules",
        "load",
        "save",
        "fromfile",
        "tofile",
        "ctypeslib",
        "f2py",
        "testing",
        "builtins",
        "__class__",
        "__mro__",
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__base__",
        "__bases__",
        "gi_frame",
        "f_globals",
        "f_builtins",
    }
)


@dataclass
class ExecResult:
    rewards: list[float]
    components: list[dict[str, float] | None]
    report: ValidationReport


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


class _NumpyProxy:
    """Whitelisted numpy-like surface — not the real numpy module."""

    def __init__(self) -> None:
        import numpy as _np

        self._np = _np
        self.asarray = _np.asarray
        self.array = _np.array
        self.zeros = _np.zeros
        self.ones = _np.ones
        self.clip = _np.clip
        self.exp = _np.exp
        self.tanh = _np.tanh
        self.sign = _np.sign
        self.sqrt = _np.sqrt
        self.abs = _np.abs
        self.sum = _np.sum
        self.mean = _np.mean
        self.std = _np.std
        self.minimum = _np.minimum
        self.maximum = _np.maximum
        self.float64 = _np.float64
        self.float32 = _np.float32
        self.int64 = _np.int64
        self.pi = float(_np.pi)
        self.e = float(_np.e)
        self.newaxis = _np.newaxis
        self.linalg = types.SimpleNamespace(norm=_np.linalg.norm)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"numpy proxy has no attribute '{name}'")


def _blocked_open(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError("File I/O is blocked in the reward sandbox")


def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            raise TypeError("Async constructs are not allowed in reward code")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _FORBIDDEN_ATTR_NAMES
        ):
            raise TypeError(f"Attribute access '{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ATTR_NAMES:
            raise TypeError(f"Name '{node.id}' is not allowed")
        if isinstance(node, ast.Call):
            func = node.func
            fname = None
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname in ("getattr", "setattr", "hasattr", "delattr", "globals", "locals", "vars", "dir", "eval", "exec", "compile", "__import__"):
                # getattr/setattr/hasattr only with literal attribute names if present
                if fname in ("getattr", "setattr", "hasattr", "delattr"):
                    if len(node.args) >= 2 and not isinstance(node.args[1], ast.Constant):
                        raise TypeError(f"{fname} with non-literal name is not allowed")
                else:
                    raise TypeError(f"Call to '{fname}' is not allowed")


def _safe_builtins() -> dict[str, Any]:
    import builtins as _b

    names = (
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
        "callable",
        "print",
        "round",
        "pow",
        "any",
        "all",
        "sorted",
        "divmod",
        "reversed",
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
        "ArithmeticError",
        "OverflowError",
        "StopIteration",
        "__build_class__",
        "__name__",
    )
    out: dict[str, Any] = {}
    for k in names:
        if hasattr(_b, k):
            out[k] = getattr(_b, k)
    out["open"] = _blocked_open
    # getattr/setattr/hasattr removed entirely to block dynamic escapes.
    return out


def _make_safe_globals(*, allow_torch: bool) -> dict[str, Any]:
    np_proxy = _NumpyProxy()
    allowed: dict[str, Any] = {
        "math": math,
        "numpy": np_proxy,
        "np": np_proxy,
        "__builtins__": _safe_builtins(),
        "__name__": "__reward_candidate__",
    }

    def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        allow = set(DEFAULT_ALLOWLIST)
        if allow_torch:
            allow.add("torch")
        if root == "numpy":
            # Never return the real numpy module.
            if fromlist:
                return np_proxy
            return np_proxy
        if root not in allow:
            raise ImportError(f"Import of '{name}' is not allowed in reward sandbox")
        if root == "math":
            return math
        if allow_torch and root == "torch":
            import torch  # type: ignore

            return torch
        raise ImportError(f"Import of '{name}' is not allowed in reward sandbox")

    allowed["__builtins__"]["__import__"] = _safe_import
    return allowed


def _apply_posix_limits(*, mem_bytes: int = 512 * 1024 * 1024, cpu_s: int = 15) -> None:
    if _resource is None or platform.system() == "Windows":
        return
    try:
        _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass
    try:
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, OSError):  # pragma: no cover
        pass


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


def _clean_transition(tr: Mapping[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for k, v in tr.items():
        if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "radius"):
            item[k] = [float(v.x), float(v.y), float(v.radius)]
        elif isinstance(v, list) and v and hasattr(v[0], "x"):
            item[k] = [[float(o.x), float(o.y), float(o.radius)] for o in v]
        elif hasattr(v, "tolist"):
            item[k] = v.tolist()
        elif isinstance(v, (list, tuple, dict, str, bool, int, float, type(None))):
            item[k] = v
        else:
            item[k] = str(v)
    return item


def _worker(
    code: str,
    transitions: list[dict[str, Any]],
    allow_torch: bool,
    out_q: Any,
) -> None:
    # Strip secrets / host env before any candidate code runs.
    with contextlib.suppress(Exception):
        os.environ.clear()
    _apply_posix_limits()
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
    except Exception as exc:  # noqa: BLE001
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
    in_process: bool = False,
) -> ExecResult:
    """Execute candidate code on a batch of probe transitions.

    Default path uses a child process. ``in_process=True`` uses the same
    sandbox loader without spawning (for fast numerical regression tests).
    """
    clean = [_clean_transition(t) for t in transitions]

    if in_process:
        t0 = time.perf_counter()
        try:
            # Do not clear the parent environ when in-process.
            fn = _load_callable(code, allow_torch=allow_torch)
            rewards: list[float] = []
            components: list[dict[str, float] | None] = []
            for tr in clean:
                r, c = _call_reward(fn, tr)
                rewards.append(r)
                components.append(c)
            wall = time.perf_counter() - t0
            return ExecResult(
                rewards=rewards,
                components=components,
                report=ValidationReport(
                    status="ok",
                    reasons=[],
                    timings={"wall_s": wall, "n_transitions": float(len(rewards))},
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return ExecResult(
                rewards=[],
                components=[],
                report=ValidationReport(
                    status="exec_error",
                    reasons=[f"{type(exc).__name__}: {exc}"],
                    timings={"wall_s": time.perf_counter() - t0},
                ),
            )

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
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
                timings={"wall_s": wall},
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
            "cap. Python-level sandbox is not a security boundary — use a container "
            "or bwrap/firejail for Phase 4 LLM code."
        )
    return (
        "POSIX: multiprocessing + join timeout + RLIMIT_AS/RLIMIT_CPU in the child. "
        "Python-level sandbox is not a security boundary — use a container or "
        "bwrap/firejail for Phase 4 LLM code."
    )
