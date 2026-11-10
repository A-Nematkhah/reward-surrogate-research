"""Logging setup for experiment scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    name: str = "reward_surrogate",
) -> logging.Logger:
    """
    Configure root project logging once and return the project logger.

    Subsequent calls update the level and optionally add a file handler
    without duplicating stream handlers.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    numeric_level = _resolve_level(level)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _CONFIGURED:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        logger.addHandler(stream_handler)
        # Avoid duplicate messages via the root logger
        logger.propagate = False
        _CONFIGURED = True
    else:
        for handler in logger.handlers:
            handler.setLevel(numeric_level)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid adding the same file handler twice
        resolved = str(path.resolve())
        already = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if str(Path(handler.baseFilename).resolve()) == resolved:
                    already = True
                    break
        if not already:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the project logger namespace."""
    base = "reward_surrogate"
    if name is None or name == base:
        return logging.getLogger(base)
    if name.startswith(base + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{base}.{name}")


def _resolve_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    raise ValueError(f"Invalid log level: {level!r}")
