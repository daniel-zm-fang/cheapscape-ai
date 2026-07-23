"""Configuration for the decoder-only GPT baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters for :class:`~model.gpt.GPT`.

    Values are kept explicit so every training run can record the exact model
    that produced a checkpoint. ``position_encoding`` is currently only
    ``\"absolute\"``; RoPE is deferred until the baseline is tested.
    """

    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    d_model: int
    dropout: float = 0.0
    position_encoding: str = "absolute"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.position_encoding != "absolute":
            raise ValueError(
                "Only absolute position encodings are supported in the Phase 4 "
                f"baseline; got {self.position_encoding!r}"
            )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GPTConfig:
        """Build a config from a YAML/dict mapping (e.g. ``configs/model.yaml``)."""
        required = (
            "vocab_size",
            "context_length",
            "n_layers",
            "n_heads",
            "d_model",
        )
        missing = [key for key in required if data.get(key) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing or null model config keys: {joined}")

        return cls(
            vocab_size=int(data["vocab_size"]),
            context_length=int(data["context_length"]),
            n_layers=int(data["n_layers"]),
            n_heads=int(data["n_heads"]),
            d_model=int(data["d_model"]),
            dropout=float(data.get("dropout", 0.0)),
            position_encoding=str(data.get("position_encoding", "absolute")),
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
