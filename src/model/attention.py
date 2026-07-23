"""Multi-head causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Scaled dot-product attention with a causal mask.

    Each position may attend to itself and earlier positions only. Queries,
    keys, and values are projected jointly and then split into heads so the
    block stays a single linear layer on the way in and out.
    """

    def __init__(
        self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_length = context_length

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Persistent buffer so the mask moves with ``.to(device)`` and is saved
        # in state_dict without being treated as a learnable parameter.
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        self.causal_mask: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over a batch of sequences shaped ``[batch, time, d_model]``."""
        batch, time, channels = x.shape
        if channels != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {channels}")
        if time > self.context_length:
            raise ValueError(f"Sequence length {time} exceeds context_length {self.context_length}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # [batch, heads, time, head_dim]
        q = q.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~self.causal_mask[:time, :time], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.d_model)
        output: torch.Tensor = self.resid_dropout(self.proj(attended))
        return output
