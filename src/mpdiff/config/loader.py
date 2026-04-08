"""Config file loading for YAML and TOML experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import ProjectConfig, project_config_from_dict
from .validation import validate_config

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate config from YAML or TOML.

    Parameters
    ----------
    path:
        Path to config file.

    Returns
    -------
    ProjectConfig
        Parsed and validated project config.
    """
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(config_path)
    elif suffix == ".toml":
        data = _load_toml(config_path)
    else:
        raise ValueError("Config must use .yaml/.yml or .toml")

    cfg = project_config_from_dict(data)
    validate_config(cfg)
    return cfg
