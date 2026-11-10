"""Small helpers for ProcessPool job fan-out (Windows-safe top-level callables)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_maybe_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int = 1,
    on_result: Optional[Callable[[int, T, R], None]] = None,
) -> list[R]:
    """
    Run ``fn`` over ``items``.

    ``workers <= 1`` stays sequential (easier debugging). ``workers > 1`` uses
    a process pool. ``fn`` must be picklable (module-level) on Windows.
    """
    n_workers = max(1, int(workers))
    if not items:
        return []
    if n_workers == 1:
        results: list[R] = []
        for i, item in enumerate(items, start=1):
            result = fn(item)
            results.append(result)
            if on_result is not None:
                on_result(i, item, result)
        return results

    results_by_idx: dict[int, R] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(fn, item): (i, item) for i, item in enumerate(items)}
        done_count = 0
        for fut in as_completed(future_map):
            idx, item = future_map[fut]
            result = fut.result()
            results_by_idx[idx] = result
            done_count += 1
            if on_result is not None:
                on_result(done_count, item, result)
    return [results_by_idx[i] for i in range(len(items))]
