"""Checkpoint writing and resume.

Rented GPUs are usually interruptible, so a run must be able to die at any
moment and continue from its last checkpoint. Writes go to a temporary file and
are renamed into place, so an interrupted write can never leave a checkpoint
that looks complete.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = 1
_STEP_PATTERN = re.compile(r"^step_(\d+)\.pt$")


def checkpoint_name(step: int) -> str:
    """Return the filename holding the state after ``step`` optimizer steps."""
    return f"step_{step:07d}.pt"


def _plain(value: Any) -> Any:
    """Reduce configs to primitives so checkpoints load with ``weights_only``."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_checkpoint(
    directory: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    train_config: Any = None,
    model_config: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint for ``step`` and return its path."""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "train_config": _plain(train_config),
        "model_config": _plain(model_config),
        "torch_rng_state": torch.get_rng_state(),
        "extra": _plain(extra or {}),
    }

    path = directory / checkpoint_name(step)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model, optimizer, and RNG state."""
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    version = int(payload.get("version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint {path} has version {version}, expected {CHECKPOINT_VERSION}")

    if model is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng and payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].to(torch.uint8).cpu())
    return payload


def list_checkpoints(directory: Path) -> list[Path]:
    """Return checkpoint paths under ``directory``, oldest step first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _STEP_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _step, path in sorted(found)]


def latest_checkpoint(directory: Path) -> Path | None:
    """Return the highest-step checkpoint under ``directory``, if any."""
    checkpoints = list_checkpoints(directory)
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(directory: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` checkpoints; return those removed."""
    if keep < 1:
        raise ValueError(f"keep must be positive, got {keep}")
    checkpoints = list_checkpoints(directory)
    removed = checkpoints[:-keep] if len(checkpoints) > keep else []
    for path in removed:
        path.unlink()
    return removed
