"""Configuration for a reproducible training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.runtime import PRECISIONS

SCHEDULES = ("constant", "cosine")


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for :func:`~training.loop.train`.

    The same config drives the one-batch overfit gate and full-corpus runs on
    rented GPUs, so it carries the fields a remote run needs: precision,
    checkpoint cadence, validation cadence, and a spend cap.
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
    overfit_one_batch: bool = False
    lr_schedule: str = "cosine"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    checkpoint_dir: Path | None = None
    checkpoint_every: int | None = None
    keep_last_checkpoints: int = 2
    resume: bool = True
    validation_every: int | None = None
    validation_batches: int = 8
    log_every: int = 10
    price_per_hour_usd: float = 0.0
    budget_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("batch_size", "gradient_accumulation_steps", "max_steps", "context_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("keep_last_checkpoints", "validation_batches", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("learning_rate",):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("weight_decay", "grad_clip", "price_per_hour_usd", "budget_usd"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be below max_steps ({self.max_steps})"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.lr_schedule not in SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {SCHEDULES}, got {self.lr_schedule!r}")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"device must be one of auto/cpu/cuda/mps, got {self.device!r}")
        if self.checkpoint_every is not None and self.checkpoint_dir is None:
            raise ValueError("checkpoint_every requires checkpoint_dir")
        if self.checkpoint_every is not None and self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be positive, got {self.checkpoint_every}")
        if self.validation_every is not None and self.validation_every < 1:
            raise ValueError(f"validation_every must be positive, got {self.validation_every}")
        if self.budget_usd > 0.0 and self.price_per_hour_usd <= 0.0:
            raise ValueError("budget_usd requires price_per_hour_usd to estimate spend")

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step."""
        return self.batch_size * self.gradient_accumulation_steps * self.context_length

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

        def optional_int(key: str) -> int | None:
            value = data.get(key)
            return int(value) if value is not None else None

        checkpoint_dir = data.get("checkpoint_dir")
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
            overfit_one_batch=bool(data.get("overfit_one_batch", False)),
            lr_schedule=str(data.get("lr_schedule", "cosine")),
            warmup_steps=int(data.get("warmup_steps", 0)),
            min_lr_ratio=float(data.get("min_lr_ratio", 0.1)),
            weight_decay=float(data.get("weight_decay", 0.1)),
            grad_clip=float(data.get("grad_clip", 1.0)),
            checkpoint_dir=resolve(str(checkpoint_dir)) if checkpoint_dir is not None else None,
            checkpoint_every=optional_int("checkpoint_every"),
            keep_last_checkpoints=int(data.get("keep_last_checkpoints", 2)),
            resume=bool(data.get("resume", True)),
            validation_every=optional_int("validation_every"),
            validation_batches=int(data.get("validation_batches", 8)),
            log_every=int(data.get("log_every", 10)),
            price_per_hour_usd=float(data.get("price_per_hour_usd", 0.0)),
            budget_usd=float(data.get("budget_usd", 0.0)),
        )
