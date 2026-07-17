"""Packed, memory-mapped next-token datasets.

A packed dataset is a flat stream of token ids stored as ``.npy`` shards plus a
JSON manifest. Storing tokens contiguously means training can memory-map each
shard and slice fixed-length windows without loading everything into RAM.

The writer (:func:`write_shards`) turns per-document token lists into evenly
sized shards; the reader (:class:`PackedTokenDataset`) exposes ``(x, y)``
next-token pairs from a single shard.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_NAME = "manifest.json"

# numpy dtypes allowed for token shards, keyed by the name stored in manifests.
_ALLOWED_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class PackedTokenDataset:
    """Expose fixed-length next-token examples from one memory-mapped shard."""

    def __init__(self, shard_path: Path, context_length: int) -> None:
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")

        self.shard_path = Path(shard_path)
        self.context_length = context_length
        # mmap keeps the shard on disk; slices are read lazily.
        self.tokens = np.load(self.shard_path, mmap_mode="r")
        # Each example needs context_length inputs plus one shifted target, so
        # the last valid start offset is len(tokens) - context_length - 1.
        self._length = max(0, int(self.tokens.shape[0]) - context_length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range for {self._length} examples")

        start = index
        end = start + self.context_length
        x = np.asarray(self.tokens[start:end], dtype=np.int64)
        y = np.asarray(self.tokens[start + 1 : end + 1], dtype=np.int64)
        return x, y


def write_shards(
    token_lists: Iterable[list[int]],
    output_dir: Path,
    shard_size: int,
    dtype: str = "uint16",
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack per-document token lists into ``.npy`` shards and write a manifest.

    Documents are concatenated into a flat stream and cut into shards of exactly
    ``shard_size`` tokens (the final shard holds the remainder). Returns the
    manifest that is also written to ``output_dir/manifest.json``.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    if dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(_ALLOWED_DTYPES)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = _ALLOWED_DTYPES[dtype]

    shards: list[dict[str, Any]] = []
    total_tokens = 0

    def flush(chunk: list[int]) -> None:
        nonlocal total_tokens
        array = np.array(chunk, dtype=np_dtype)
        name = f"shard_{len(shards):05d}.npy"
        np.save(output_dir / name, array)
        shards.append({"path": name, "num_tokens": int(array.shape[0])})
        total_tokens += int(array.shape[0])

    buffer: list[int] = []
    for tokens in token_lists:
        buffer.extend(tokens)
        while len(buffer) >= shard_size:
            flush(buffer[:shard_size])
            del buffer[:shard_size]
    if buffer:
        flush(buffer)

    manifest: dict[str, Any] = {
        "version": 1,
        "dtype": dtype,
        "shard_size": shard_size,
        "total_tokens": total_tokens,
        "shards": shards,
    }
    if extra:
        manifest.update(extra)

    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a shard manifest from a file or its containing directory."""
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest
