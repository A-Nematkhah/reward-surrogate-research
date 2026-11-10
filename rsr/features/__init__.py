"""Static and behavioral reward feature extraction."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_feature_tables",
    "ensure_probe_set",
    "extract_static_features_from_spec",
    "save_feature_tables",
]


def __getattr__(name: str) -> Any:
    if name in {"build_feature_tables", "ensure_probe_set", "save_feature_tables"}:
        from rsr.features.feature_builder import (
            build_feature_tables,
            ensure_probe_set,
            save_feature_tables,
        )

        return {
            "build_feature_tables": build_feature_tables,
            "ensure_probe_set": ensure_probe_set,
            "save_feature_tables": save_feature_tables,
        }[name]
    if name == "extract_static_features_from_spec":
        from rsr.features.static_features import extract_static_features_from_spec

        return extract_static_features_from_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
