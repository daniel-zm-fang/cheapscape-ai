"""A decoder-only Transformer baseline.

Components live in small modules (:mod:`model.attention`, :mod:`model.mlp`,
:mod:`model.block`) and are assembled here into a GPT that maps token ids to
vocabulary logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from model.attention import CausalSelfAttention
from model.block import TransformerBlock
from model.config import GPTConfig
from model.mlp import MLP


class GPT(nn.Module):
    """Dense decoder-only language model with absolute position embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.position_emb = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.context_length,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: the output projection reuses the token embedding table.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # Scale residual projection weights so early updates stay modest with depth.
        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for module in self.modules():
            if isinstance(module, (CausalSelfAttention, MLP)):
                module.proj.weight.data.mul_(residual_scale)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GPT:
        """Construct a model from a ``configs/model.yaml``-style mapping."""
        return cls(GPTConfig.from_mapping(data))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch, time, vocab]``."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape [batch, time], got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"token_ids must be integer dtype, got {token_ids.dtype}")

        _batch, time = token_ids.shape
        if time > self.config.context_length:
            raise ValueError(
                f"Sequence length {time} exceeds context_length " f"{self.config.context_length}"
            )
        if token_ids.numel() > 0:
            min_id = int(token_ids.min())
            max_id = int(token_ids.max())
            if min_id < 0 or max_id >= self.config.vocab_size:
                raise ValueError(
                    f"token_ids must be in [0, {self.config.vocab_size}), "
                    f"got min={min_id}, max={max_id}"
                )

        positions = torch.arange(time, device=token_ids.device)
        x = self.token_emb(token_ids) + self.position_emb(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits: torch.Tensor = self.lm_head(x)
        return logits

    def parameter_count(self) -> int:
        """Return the number of trainable parameters (tied weights counted once)."""
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
