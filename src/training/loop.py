"""The training loop.

Built for interruptible rented GPUs: every run streams windows from packed
shards, tracks throughput and spend, checkpoints on a cadence, resumes from the
newest checkpoint, and saves once more when the host signals preemption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from datasets.packed import PackedSplit
from model import GPT, GPTConfig
from training.checkpoint import (
    checkpoint_name,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from training.config import TrainConfig
from training.runtime import (
    PreemptionSignal,
    autocast,
    peak_memory_bytes,
    resolve_device,
    supports_precision,
)
from training.schedule import lr_at_step


class ExampleSource(Protocol):
    """Anything that exposes indexed ``(x, y)`` next-token examples."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class TrainResult:
    """Summary of one training run."""

    losses: tuple[float, ...]
    val_losses: tuple[tuple[int, float], ...]
    initial_loss: float
    final_loss: float
    steps: int
    start_step: int
    device: str
    elapsed_seconds: float
    tokens_per_second: float
    estimated_cost_usd: float
    peak_memory_bytes: int
    stop_reason: str
    checkpoints: tuple[Path, ...]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible run."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stack_batch(
    dataset: ExampleSource, indices: list[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate the examples at ``indices`` into one ``(x, y)`` batch."""
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for index in indices:
        x, y = dataset[int(index)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay matrices but not biases or normalization gains."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: ExampleSource,
    *,
    batch_size: int,
    batches: int,
    device: torch.device,
    precision: str,
    vocab_size: int,
) -> float:
    """Mean loss over the first ``batches`` fixed windows of ``dataset``.

    Windows are always the same, so validation loss stays comparable across
    runs and across resumes.
    """
    available = len(dataset) // batch_size
    if available < 1:
        raise ValueError(
            f"Validation split has {len(dataset)} examples; need at least {batch_size}"
        )

    was_training = model.training
    model.eval()
    total = 0.0
    used = min(batches, available)
    for index in range(used):
        start = index * batch_size
        x, y = stack_batch(dataset, list(range(start, start + batch_size)))
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        total += float(loss)
    if was_training:
        model.train()
    return total / used


def train(
    train_config: TrainConfig,
    model_config: GPTConfig,
    *,
    model: GPT | None = None,
    dataset: ExampleSource | None = None,
    val_dataset: ExampleSource | None = None,
    preemption: PreemptionSignal | None = None,
) -> TrainResult:
    """Run one reproducible training experiment.

    With ``overfit_one_batch`` the first batch is reused every step (the Phase 5
    correctness gate). Otherwise windows are sampled from the packed train split
    with a seeded generator.
    """
    if train_config.context_length > model_config.context_length:
        raise ValueError(
            f"Training context_length {train_config.context_length} exceeds model "
            f"maximum {model_config.context_length}"
        )

    seed_everything(train_config.seed)
    device = resolve_device(train_config.device)
    if not supports_precision(device, train_config.precision):
        raise ValueError(f"{device.type} cannot run precision {train_config.precision!r}")

    if dataset is None:
        dataset = PackedSplit(train_config.packed_dir, "train", train_config.context_length)
    if len(dataset) < train_config.batch_size:
        raise ValueError(
            f"Need at least {train_config.batch_size} examples to fill a batch; "
            f"split only has {len(dataset)}"
        )

    if model is None:
        model = GPT(model_config)
    elif model.config != model_config:
        raise ValueError("Injected model config does not match model_config")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        parameter_groups(model, train_config.weight_decay),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
    )
    generator = torch.Generator().manual_seed(train_config.seed)

    start_step = 0
    if train_config.resume and train_config.checkpoint_dir is not None:
        resume_from = latest_checkpoint(train_config.checkpoint_dir)
        if resume_from is not None:
            payload = load_checkpoint(
                resume_from, model=model, optimizer=optimizer, map_location=device
            )
            start_step = int(payload["step"])
            sampler_state = payload.get("extra", {}).get("sampler_state")
            if sampler_state is not None:
                generator.set_state(sampler_state.to(torch.uint8).cpu())
            print(f"Resumed from {resume_from} at step {start_step}")

    # GradScaler only understands cuda/cpu; fp16 is gated to cuda upstream.
    scaler_device = device.type if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=train_config.precision == "fp16")

    fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    if train_config.overfit_one_batch:
        x, y = stack_batch(dataset, list(range(train_config.batch_size)))
        fixed_batch = (x.to(device), y.to(device))

    if train_config.validation_every is not None:
        if val_dataset is None:
            val_dataset = PackedSplit(train_config.packed_dir, "val", train_config.context_length)
        # Fail before spending GPU time: a mid-run validation error would waste
        # every step already paid for.
        if len(val_dataset) < train_config.batch_size:
            raise ValueError(
                f"Validation split has {len(val_dataset)} examples but batch_size is "
                f"{train_config.batch_size}; pack more validation data or lower batch_size"
            )

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        if fixed_batch is not None:
            return fixed_batch
        indices = torch.randint(
            len(dataset), (train_config.batch_size,), generator=generator
        ).tolist()
        x, y = stack_batch(dataset, indices)
        return x.to(device), y.to(device)

    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    checkpoints: list[Path] = []
    accum = train_config.gradient_accumulation_steps
    stop_reason = "max_steps" if start_step < train_config.max_steps else "already_complete"
    started = time.perf_counter()
    completed = start_step
    interrupted = False

    def elapsed() -> float:
        return time.perf_counter() - started

    def cost(seconds: float) -> float:
        return train_config.price_per_hour_usd * seconds / 3600.0

    def write_checkpoint(at_step: int) -> None:
        if train_config.checkpoint_dir is None:
            return
        path = save_checkpoint(
            train_config.checkpoint_dir,
            step=at_step,
            model=model,
            optimizer=optimizer,
            train_config=train_config,
            model_config=model_config,
            extra={"sampler_state": generator.get_state()},
        )
        checkpoints.append(path)
        prune_checkpoints(train_config.checkpoint_dir, train_config.keep_last_checkpoints)

    for step in range(start_step, train_config.max_steps):
        lr = lr_at_step(
            step,
            base_lr=train_config.learning_rate,
            max_steps=train_config.max_steps,
            warmup_steps=train_config.warmup_steps,
            min_lr_ratio=train_config.min_lr_ratio,
            schedule=train_config.lr_schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _micro in range(accum):
            batch_x, batch_y = sample_batch()
            with autocast(device, train_config.precision):
                logits = model(batch_x)
                loss = F.cross_entropy(
                    logits.reshape(-1, model_config.vocab_size), batch_y.reshape(-1)
                )
            scaler.scale(loss / accum).backward()  # type: ignore[no-untyped-call]
            step_loss += float(loss.detach()) / accum

        if train_config.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(step_loss)
        completed = step + 1

        if train_config.validation_every and completed % train_config.validation_every == 0:
            assert val_dataset is not None
            val_losses.append(
                (
                    completed,
                    evaluate(
                        model,
                        val_dataset,
                        batch_size=train_config.batch_size,
                        batches=train_config.validation_batches,
                        device=device,
                        precision=train_config.precision,
                        vocab_size=model_config.vocab_size,
                    ),
                )
            )

        if completed % train_config.log_every == 0 or completed == train_config.max_steps:
            seconds = elapsed()
            done = completed - start_step
            print(
                f"step {completed}/{train_config.max_steps} loss {step_loss:.4f} "
                f"lr {lr:.2e} {done * train_config.tokens_per_step / max(seconds, 1e-9):,.0f} tok/s "
                f"${cost(seconds):.2f}"
            )

        if train_config.checkpoint_every and completed % train_config.checkpoint_every == 0:
            write_checkpoint(completed)

        if preemption is not None and preemption.triggered:
            stop_reason = f"preempted:{preemption.signal_name}"
            interrupted = True
        elif train_config.budget_usd > 0.0 and cost(elapsed()) >= train_config.budget_usd:
            stop_reason = "budget_exhausted"
            interrupted = True

        if interrupted:
            write_checkpoint(completed)
            break

    # Always leave the newest state on disk, without duplicating a cadence save.
    already_saved = bool(checkpoints) and checkpoints[-1].name == checkpoint_name(completed)
    if not interrupted and train_config.checkpoint_dir is not None and losses and not already_saved:
        write_checkpoint(completed)

    seconds = elapsed()
    steps_run = len(losses)
    return TrainResult(
        losses=tuple(losses),
        val_losses=tuple(val_losses),
        initial_loss=losses[0] if losses else float("nan"),
        final_loss=losses[-1] if losses else float("nan"),
        steps=steps_run,
        start_step=start_step,
        device=str(device),
        elapsed_seconds=seconds,
        tokens_per_second=steps_run * train_config.tokens_per_step / max(seconds, 1e-9),
        estimated_cost_usd=cost(seconds),
        peak_memory_bytes=peak_memory_bytes(device),
        stop_reason=stop_reason,
        checkpoints=tuple(checkpoints),
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
    with PreemptionSignal() as preemption:
        return train(train_config, model_config, preemption=preemption)
