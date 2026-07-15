"""Interfaces for packed, memory-mapped next-token datasets."""

from pathlib import Path


class PackedTokenDataset:
    def __init__(self, shard_path: Path, context_length: int) -> None:
        """Open one shard and expose fixed-length x/y next-token examples."""
        raise NotImplementedError("Phase 3 exercise: define shard format and indexing.")
