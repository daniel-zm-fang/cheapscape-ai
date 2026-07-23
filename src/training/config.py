"""Configuration for a reproducible training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for :func:`~training.loop.train`.

    Phase 5 focuses on a one-batch overfit gate: the same batch is reused for
    ``max_steps`` so the loop, loss, and optimizer can be proven before adding
    shuffling, validation, or checkpoints.
    """

    seed: int
    device: str
    precision: str
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    context_length: int
    packed_dir: Path
    model_config: Path
    overfit_one_batch: bool = True
    checkpoint_every: int | None = None
    validation_every: int | None = None
    budget_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.context_length < 1:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.precision != "fp32":
            raise ValueError(
                "Only fp32 is supported in the Phase 5 baseline; " f"got {self.precision!r}"
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"device must be one of auto/cpu/cuda/mps, got {self.device!r}")

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, repo_root: Path | None = None) -> TrainConfig:
        """Build a config from a ``configs/train.yaml``-style mapping."""
        required = (
            "seed",
            "device",
            "precision",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "learning_rate",
            "context_length",
            "packed_dir",
            "model_config",
        )
        missing = [key for key in required if data.get(key) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing or null train config keys: {joined}")

        root = repo_root if repo_root is not None else Path.cwd()

        def resolve(path_value: str | Path) -> Path:
            path = Path(path_value)
            return path if path.is_absolute() else root / path

        checkpoint_every = data.get("checkpoint_every")
        validation_every = data.get("validation_every")
        return cls(
            seed=int(data["seed"]),
            device=str(data["device"]),
            precision=str(data["precision"]),
            batch_size=int(data["batch_size"]),
            gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
            max_steps=int(data["max_steps"]),
            learning_rate=float(data["learning_rate"]),
            context_length=int(data["context_length"]),
            packed_dir=resolve(str(data["packed_dir"])),
            model_config=resolve(str(data["model_config"])),
            overfit_one_batch=bool(data.get("overfit_one_batch", True)),
            checkpoint_every=int(checkpoint_every) if checkpoint_every is not None else None,
            validation_every=int(validation_every) if validation_every is not None else None,
            budget_usd=float(data.get("budget_usd", 0.0)),
        )
