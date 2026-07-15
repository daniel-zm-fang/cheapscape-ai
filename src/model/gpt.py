"""A decoder-only Transformer contract.

Build components in small tested modules before assembling this class.
"""

import torch
from torch import nn


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int) -> None:
        super().__init__()
        raise NotImplementedError("Phase 4 exercise: assemble your tested Transformer components.")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, time, vocab]."""
        raise NotImplementedError("Phase 4 exercise: define and test the forward pass.")
