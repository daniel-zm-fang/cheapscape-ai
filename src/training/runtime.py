"""Device, precision, and preemption helpers for remote runs."""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Iterator
from types import FrameType
from typing import Any, Self

import torch

PRECISIONS = ("fp32", "bf16", "fp16")

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


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


@contextlib.contextmanager
def autocast(device: torch.device, precision: str) -> Iterator[None]:
    """Run the enclosed block under mixed precision when requested."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "fp32" or device.type == "mps":
        yield
        return
    with torch.autocast(device_type=device.type, dtype=_DTYPES[precision]):
        yield


def supports_precision(device: torch.device, precision: str) -> bool:
    """Return whether ``device`` can run ``precision`` usefully."""
    if precision == "fp32":
        return True
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported() if precision == "bf16" else True
    # CPU autocast supports bf16; fp16 needs a CUDA GradScaler to stay stable.
    return device.type == "cpu" and precision == "bf16"


class PreemptionSignal:
    """Record SIGTERM/SIGINT so the loop can checkpoint before exiting.

    Spot and interruptible instances warn by signalling the process, so the
    handler only sets a flag; the training loop decides when it is safe to stop.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)) -> None:
        self._signals = signals
        self._previous: dict[int, Any] = {}
        self.triggered = False
        self.signal_name: str | None = None

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        self.triggered = True
        self.signal_name = signal.Signals(signum).name

    def __enter__(self) -> Self:
        # Handlers can only be installed from the main thread; tests and worker
        # threads fall back to a flag that nothing sets.
        if threading.current_thread() is threading.main_thread():
            for signum in self._signals:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


def peak_memory_bytes(device: torch.device) -> int:
    """Return peak allocated memory for ``device`` (0 when unavailable)."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0
