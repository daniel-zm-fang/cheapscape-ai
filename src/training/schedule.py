"""Learning-rate schedules."""

from __future__ import annotations

import math


def lr_at_step(
    step: int,
    *,
    base_lr: float,
    max_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    """Return the learning rate for a zero-based ``step``.

    ``warmup_steps`` ramps linearly from zero to ``base_lr``; ``cosine`` then
    decays to ``base_lr * min_lr_ratio`` by ``max_steps``. ``constant`` holds
    ``base_lr`` after warmup, which keeps the one-batch overfit gate simple.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if schedule not in {"constant", "cosine"}:
        raise ValueError(f"schedule must be constant or cosine, got {schedule!r}")

    if warmup_steps and step < warmup_steps:
        # Step 0 already receives a non-zero rate so the first update is useful.
        return base_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return base_lr

    decay_steps = max(max_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / decay_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
