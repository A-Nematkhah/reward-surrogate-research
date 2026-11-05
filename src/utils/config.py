"""YAML configuration loading and merging utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

# Project root: reward-surrogate-research/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def relpath_to_project(path: str | Path, *, root: Path | None = None) -> str:
    """Return ``path`` relative to the project root when possible (portable JSON)."""
    base = root or PROJECT_ROOT
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()

CONFIG_FILES = {
    "environment": "environment.yaml",
    "ppo": "ppo.yaml",
    "ppo_crowdnav": "ppo_crowdnav.yaml",
    "crowdnav_run": "crowdnav_run.yaml",
    "reward_generation": "reward_generation.yaml",
    "reward_generation_crowdnav": "reward_generation_crowdnav.yaml",
    "evaluation": "evaluation.yaml",
    "surrogate": "surrogate.yaml",
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def _ensure_mapping(data: Any, path: Path) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file into a dictionary."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _ensure_mapping(data, config_path)


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Save a config mapping to YAML. Creates parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(config),
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return out_path


def deep_update(
    base: MutableMapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge ``overrides`` into a deep copy of ``base``.

    Nested dicts are merged; other values are replaced.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config_path(
    name_or_path: str | Path,
    config_dir: str | Path | None = None,
) -> Path:
    """
    Resolve a config name (e.g. ``environment``) or filesystem path.

    Names are looked up in ``CONFIG_FILES`` under ``config_dir``.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path.resolve()

    key = str(name_or_path)
    if key.endswith(".yaml") or key.endswith(".yml"):
        candidate = Path(config_dir or DEFAULT_CONFIG_DIR) / key
        if candidate.is_file():
            return candidate.resolve()
        raise ConfigError(f"Config file not found: {candidate}")

    if key not in CONFIG_FILES:
        raise ConfigError(
            f"Unknown config name '{key}'. "
            f"Expected one of {sorted(CONFIG_FILES)} or a file path."
        )
    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    resolved = base / CONFIG_FILES[key]
    if not resolved.is_file():
        raise ConfigError(f"Config file not found: {resolved}")
    return resolved.resolve()


def load_named_config(
    name: str,
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a named project config and optionally apply overrides."""
    config = load_config(resolve_config_path(name, config_dir=config_dir))
    if overrides:
        config = deep_update(config, overrides)
    return config


def load_merged_configs(
    names: list[str] | None = None,
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load multiple named configs into a single nested dict keyed by name.

    Example::

        {
            "environment": {...},
            "ppo": {...},
            ...
        }
    """
    selected = names if names is not None else list(CONFIG_FILES.keys())
    merged: dict[str, Any] = {}
    for name in selected:
        merged[name] = load_named_config(name, config_dir=config_dir)
    if overrides:
        merged = deep_update(merged, overrides)
    return merged


def parse_override_value(raw: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_dot_overrides(
    config: MutableMapping[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """
    Apply CLI-style dotted overrides.

    Example::

        apply_dot_overrides(cfg, ["ppo.total_timesteps=1000", "ppo.n_seeds=1"])
    """
    result = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(
                f"Invalid override '{item}'. Expected KEY=VALUE "
                f"(e.g. ppo.total_timesteps=1000)."
            )
        key_path, raw_value = item.split("=", 1)
        keys = key_path.strip().split(".")
        if not keys or any(not k for k in keys):
            raise ConfigError(f"Invalid override key path: '{key_path}'")
        value = parse_override_value(raw_value.strip())
        cursor: MutableMapping[str, Any] = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                cursor[key] = {}
            elif not isinstance(existing, dict):
                raise ConfigError(
                    f"Cannot override '{key_path}': "
                    f"'{key}' is not a mapping."
                )
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return result
