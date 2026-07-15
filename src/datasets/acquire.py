"""Acquire raw text from configured sources into data/raw/."""

import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def acquire_fixture(source: dict[str, Any], output_dir: Path) -> Path:
    """Copy a local fixture file into output_dir and return the destination path."""
    if source.get("type") != "fixture":
        raise ValueError(f"Expected type 'fixture', got {source.get('type')!r}")

    try:
        relative_path = Path(source["path"])
        source_id = str(source["id"])
    except KeyError as exc:
        raise ValueError(f"Fixture source missing required key: {exc.args[0]}") from exc

    fixture_path = relative_path if relative_path.is_absolute() else REPO_ROOT / relative_path
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{source_id}.txt"
    shutil.copy2(fixture_path, destination)
    return destination
