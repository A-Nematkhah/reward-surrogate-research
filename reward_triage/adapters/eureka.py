"""EUREKA-style adapter: compute_reward(...) -> (reward, dict).

Host-specific call signatures live in configs/adapters/*.yaml and are verified
when a host is chosen. No external host code is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from reward_triage.adapters.base import RewardAdapter
from reward_triage.executor import _load_callable


def load_signature_template(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Signature template must be a mapping: {path}")
    return data


class EurekaStyleAdapter(RewardAdapter):
    """Call ``compute_reward`` with kwargs drawn from the transition by YAML map."""

    def __init__(
        self,
        signature: Mapping[str, Any] | None = None,
        *,
        allow_torch: bool = False,
    ) -> None:
        self.signature = dict(signature or {"entry": "compute_reward", "args": ["transition"]})
        self._fn: Any = None
        self._allow_torch = allow_torch

    @classmethod
    def from_yaml(cls, path: str | Path, *, allow_torch: bool = False) -> EurekaStyleAdapter:
        return cls(load_signature_template(path), allow_torch=allow_torch)

    def load(self, code: str) -> None:
        self._fn = _load_callable(code, allow_torch=self._allow_torch)

    def call(
        self, transition: Mapping[str, Any]
    ) -> tuple[float, dict[str, float] | None]:
        if self._fn is None:
            raise RuntimeError("EurekaStyleAdapter.load(code) must be called first")
        args_spec: Sequence[str] = list(self.signature.get("args") or ["transition"])
        kwargs_map: Mapping[str, str] = dict(self.signature.get("kwargs_from_transition") or {})
        args = []
        for name in args_spec:
            if name == "transition":
                args.append(dict(transition))
            else:
                args.append(transition.get(name))
        kwargs = {k: transition.get(src) for k, src in kwargs_map.items()}
        out = self._fn(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            total, comps = out
            components = None
            if comps is not None:
                components = {str(k): float(v) for k, v in dict(comps).items()}
            return float(total), components
        return float(out), None
