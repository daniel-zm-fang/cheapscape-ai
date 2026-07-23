"""Position-wise feed-forward network used inside each Transformer block."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Expand → GELU → contract feed-forward, applied independently per token.

    This is the classic GPT-style baseline from the Transformer notes. Gated
    variants such as SwiGLU are deferred until the absolute-position baseline
    is verified end-to-end.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")

        hidden = expansion * d_model
        self.fc = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.fc(x))
        output: torch.Tensor = self.dropout(self.proj(hidden))
        return output
