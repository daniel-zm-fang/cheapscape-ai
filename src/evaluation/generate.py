"""Decoding boundary for qualitative evaluation."""

import torch


@torch.no_grad()
def generate() -> None:
    """Start with greedy decoding; add sampling controls after tests exist."""
    raise NotImplementedError("Phase 6 exercise: implement greedy decoding first.")
