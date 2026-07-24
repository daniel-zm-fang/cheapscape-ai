"""Throughput and memory measurement for sizing a rented GPU run.

The context-length decision needs numbers, not guesses: measure tokens per
second and peak memory at each candidate length on the GPU you intend to rent,
then project what the token budget would cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from training.runtime import autocast, peak_memory_bytes


@dataclass(frozen=True)
class BenchmarkRow:
    """One measured configuration."""

    context_length: int
    batch_size: int
    gradient_accumulation_steps: int
    precision: str
    device: str
    tokens_per_step: int
    seconds_per_step: float
    tokens_per_second: float
    peak_memory_bytes: int

    @property
    def peak_memory_gib(self) -> float:
        return self.peak_memory_bytes / 2**30


def project_cost(
    tokens_per_second: float, token_budget: int, price_per_hour_usd: float
) -> tuple[float, float]:
    """Return ``(hours, usd)`` needed to train ``token_budget`` tokens."""
    if tokens_per_second <= 0.0:
        raise ValueError(f"tokens_per_second must be positive, got {tokens_per_second}")
    if token_budget < 1:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if price_per_hour_usd < 0.0:
        raise ValueError(f"price_per_hour_usd must be non-negative, got {price_per_hour_usd}")

    hours = token_budget / tokens_per_second / 3600.0
    return hours, hours * price_per_hour_usd


def benchmark_context(
    model_config: GPTConfig,
    *,
    context_length: int,
    batch_size: int,
    device: torch.device,
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    steps: int = 5,
    warmup: int = 2,
) -> BenchmarkRow:
    """Time full training steps on synthetic batches of ``context_length`` tokens.

    Synthetic data keeps this runnable before a corpus is packed, which is the
    point: the measurement should inform packing and budget decisions.
    """
    if context_length > model_config.context_length:
        raise ValueError(
            f"context_length {context_length} exceeds model maximum "
            f"{model_config.context_length}"
        )
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    # Size the model to the context under test so position embeddings and peak
    # memory reflect the configuration that would actually run.
    model = GPT(replace(model_config, context_length=context_length)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(
        device.type if device.type == "cuda" else "cpu", enabled=precision == "fp16"
    )

    batch = torch.randint(0, model_config.vocab_size, (batch_size, context_length), device=device)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(gradient_accumulation_steps):
            with autocast(device, precision):
                logits = model(batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, model_config.vocab_size), batch.reshape(-1)
                )
            scaler.scale(loss / gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    tokens_per_step = batch_size * gradient_accumulation_steps * context_length
    seconds_per_step = elapsed / steps
    return BenchmarkRow(
        context_length=context_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
        device=str(device),
        tokens_per_step=tokens_per_step,
        seconds_per_step=seconds_per_step,
        tokens_per_second=tokens_per_step / max(seconds_per_step, 1e-9),
        peak_memory_bytes=peak_memory_bytes(device),
    )
