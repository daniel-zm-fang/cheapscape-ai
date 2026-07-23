"""Training-loop boundary.

Phase 5 implements the one-batch overfit gate: loss, optimizer, seeding, and
device selection. Validation, checkpoints, resume, mixed precision, and
distributed execution come later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from datasets.packed import PackedTokenDataset, load_manifest
from model import GPT, GPTConfig
from training.config import TrainConfig


@dataclass(frozen=True)
class TrainResult:
    """Summary of one training run."""

    losses: tuple[float, ...]
    initial_loss: float
    final_loss: float
    steps: int
    device: str


def resolve_device(name: str) -> torch.device:
    """Map a config device string to a :class:`torch.device`."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible run."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_shard(packed_dir: Path, context_length: int) -> PackedTokenDataset:
    """Open the first train shard described by ``packed_dir/manifest.json``."""
    manifest = load_manifest(packed_dir)
    splits = manifest.get("splits", {})
    train_split = splits.get("train")
    if not train_split or not train_split.get("shards"):
        raise ValueError(f"No train shards found in manifest under {packed_dir}")

    shard_name = train_split["shards"][0]
    return PackedTokenDataset(
        packed_dir / shard_name,
        context_length=context_length,
        dtype=str(manifest["dtype"]),
    )


def _stack_batch(
    dataset: PackedTokenDataset, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for index in indices:
        x, y = dataset[index]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def train(
    train_config: TrainConfig,
    model_config: GPTConfig,
    *,
    model: GPT | None = None,
    dataset: PackedTokenDataset | None = None,
) -> TrainResult:
    """Run one reproducible training experiment.

    With ``overfit_one_batch=True`` (the Phase 5 default), the first
    ``batch_size`` examples are reused every step so the loop can drive loss
    near zero before full-corpus training is introduced.
    """
    if train_config.context_length > model_config.context_length:
        raise ValueError(
            f"Training context_length {train_config.context_length} exceeds model "
            f"maximum {model_config.context_length}"
        )
    if train_config.checkpoint_every is not None or train_config.validation_every is not None:
        raise NotImplementedError(
            "Checkpoints and validation are deferred until after the one-batch "
            "overfit gate; leave checkpoint_every and validation_every null."
        )
    if not train_config.overfit_one_batch:
        raise NotImplementedError(
            "Full-corpus training is deferred until one-batch overfitting works; "
            "set overfit_one_batch: true."
        )

    seed_everything(train_config.seed)
    device = resolve_device(train_config.device)

    if dataset is None:
        dataset = load_train_shard(train_config.packed_dir, train_config.context_length)
    if len(dataset) < train_config.batch_size:
        raise ValueError(
            f"Need at least {train_config.batch_size} examples to fill a batch; "
            f"shard only has {len(dataset)}"
        )

    if model is None:
        # Keep the architecture max context from model.yaml; training windows may
        # be shorter (e.g. 512 for the overfit gate).
        model = GPT(model_config)
    elif model.config.vocab_size != model_config.vocab_size:
        raise ValueError("Injected model vocab_size does not match model_config")

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    indices = list(range(train_config.batch_size))
    batch_x, batch_y = _stack_batch(dataset, indices)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    losses: list[float] = []
    accum = train_config.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)

    for step in range(train_config.max_steps):
        logits = model(batch_x)
        loss = F.cross_entropy(
            logits.reshape(-1, model_config.vocab_size),
            batch_y.reshape(-1),
        )
        scaled = loss / accum
        scaled.backward()  # type: ignore[no-untyped-call]

        if (step + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))

    # Flush a trailing partial accumulation window, if any.
    if train_config.max_steps % accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return TrainResult(
        losses=tuple(losses),
        initial_loss=losses[0],
        final_loss=losses[-1],
        steps=len(losses),
        device=str(device),
    )


def train_from_configs(
    train_mapping: dict[str, Any],
    model_mapping: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> TrainResult:
    """Convenience wrapper used by ``scripts/train.py``."""
    root = repo_root if repo_root is not None else Path.cwd()
    train_config = TrainConfig.from_mapping(train_mapping, repo_root=root)
    if model_mapping is None:
        from config import load_config

        model_mapping = load_config(train_config.model_config)
    model_config = GPTConfig.from_mapping(model_mapping)
    return train(train_config, model_config)
