"""One pre-normalized decoder block."""

from __future__ import annotations

import torch
from torch import nn

from model.attention import CausalSelfAttention
from model.mlp import MLP


class TransformerBlock(nn.Module):
    """Pre-norm residual block: attention then MLP.

    Following the project's training notes, each sub-layer normalizes its input
    and adds a residual update rather than rewriting the stream::

        h'   = h + Attention(LN(h))
        h''  = h' + MLP(LN(h'))
    """

    def __init__(
        self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
