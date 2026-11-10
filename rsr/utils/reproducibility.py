"""Random seed control for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

# Torch is optional at import time so config/utils tests can run before
# the full dependency stack is installed. Seeding torch happens when available.
try:
    import torch

    _HAS_TORCH = True
except (ImportError, OSError):  # pragma: no cover - absent or broken native wheel
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def seed_everything(
    seed: int,
    *,
    deterministic_torch: bool = True,
    cudnn_benchmark: bool = False,
) -> dict[str, Any]:
    """
    Seed Python, NumPy, and (if available) PyTorch RNGs.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    deterministic_torch:
        If True, enable PyTorch deterministic algorithms where supported.
        This can reduce performance; preferred for research reproducibility.
    cudnn_benchmark:
        CuDNN benchmark mode (usually False when seeking reproducibility).

    Returns
    -------
    dict
        Metadata describing what was seeded (useful to save with runs).
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    meta: dict[str, Any] = {
        "seed": seed,
        "python_hashseed": os.environ["PYTHONHASHSEED"],
        "numpy_seeded": True,
        "torch_seeded": False,
        "deterministic_torch": False,
        "has_torch": _HAS_TORCH,
    }

    if _HAS_TORCH:
        assert torch is not None
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        meta["torch_seeded"] = True
        meta["cuda_available"] = bool(torch.cuda.is_available())

        if deterministic_torch:
            # warn_only avoids hard crashes on ops without deterministic impl
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = cudnn_benchmark
            # Required by some CUDA deterministic matmul paths
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            meta["deterministic_torch"] = True
            meta["cudnn_benchmark"] = cudnn_benchmark

    return meta


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader-style workers.

    Derives a per-worker seed from NumPy's global bit generator state so
    that workers are deterministic given ``seed_everything``.
    """
    del worker_id  # SB3/torch pass this; we derive seed from numpy state
    worker_seed = np.random.randint(0, 2**31 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if _HAS_TORCH:
        assert torch is not None
        torch.manual_seed(worker_seed)


def make_rng(seed: int) -> np.random.Generator:
    """Create an isolated NumPy Generator from ``seed``."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(int(seed))
