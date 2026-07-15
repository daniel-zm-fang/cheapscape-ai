"""Configuration loading contract.

Keep parsing and validation here so every script records the resolved settings.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path, required_keys: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML file and optionally validate required top-level fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")

    if required_keys is not None:
        missing = [key for key in required_keys if key not in data]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required config keys: {joined}")

    return data
