"""Packed, memory-mapped next-token datasets.

The training loop wants a long, contiguous stream of token ids that it can read
without holding the whole corpus in memory. Phase 3 turns tokenizer output into
flat binary *shards* of fixed-width integers plus a small JSON manifest, and
exposes each shard as a dataset of contiguous ``(x, y)`` next-token examples.

Shard format
    Each shard is a headerless little-endian array of unsigned integers written
    with :func:`numpy.ndarray.tofile`. Every id occupies a fixed number of bytes
    (see :func:`dtype_for_vocab`), so a shard can be memory-mapped and indexed in
    O(1). The dtype is not stored in the file; it lives in the manifest instead,
    which keeps the shards trivially inspectable with ``numpy.fromfile``.

Example layout
    For a token stream ``t[0], t[1], ...`` and a context length ``T``, example
    ``i`` is the contiguous block starting at ``i * T``::

        x = t[i*T   : i*T + T]
        y = t[i*T+1 : i*T + T + 1]

    ``y`` is ``x`` shifted one token to the right, so a shard of ``n`` tokens
    yields ``(n - 1) // T`` non-overlapping examples.
"""

import bisect
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

_UINT16_MAX_VOCAB = 2**16  # ids 0 .. 65535 fit in uint16


def dtype_for_vocab(vocab_size: int) -> np.dtype[Any]:
    """Return the smallest unsigned integer dtype that can hold every token id.

    Ids range over ``0 .. vocab_size - 1``. ``uint16`` covers vocabularies up to
    65536 tokens (which halves shard size versus ``uint32``); larger vocabularies
    fall back to ``uint32``.
    """
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size <= _UINT16_MAX_VOCAB:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def write_shard(path: Path, tokens: Iterable[int], dtype: np.dtype[Any]) -> int:
    """Write ``tokens`` to ``path`` as a flat ``dtype`` array; return the count."""
    array = np.fromiter(tokens, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return int(array.shape[0])


@dataclass(frozen=True)
class ShardInfo:
    """Where a shard lives (relative to the manifest) and how many tokens it holds."""

    filename: str
    split: str
    num_tokens: int


class ShardWriter:
    """Stream tokens for one split into ``shard_size``-token ``.bin`` files.

    The writer buffers ids and flushes a shard whenever the buffer reaches
    ``shard_size``; :meth:`close` writes any remainder. Shards are named
    ``{split}_{index:05d}.bin`` and never overlap, so appending tokens for
    several splits in a single pass over the corpus keeps memory bounded by the
    buffer rather than the whole dataset.
    """

    def __init__(self, output_dir: Path, split: str, dtype: np.dtype[Any], shard_size: int) -> None:
        if shard_size < 1:
            raise ValueError(f"shard_size must be positive, got {shard_size}")

        self.output_dir = Path(output_dir)
        self.split = split
        self.dtype = np.dtype(dtype)
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list[int] = []
        self._shards: list[ShardInfo] = []
        self._closed = False

    def append(self, token_ids: Iterable[int]) -> None:
        """Buffer ``token_ids``, flushing whole shards as the buffer fills."""
        if self._closed:
            raise RuntimeError("Cannot append to a closed ShardWriter")
        for token_id in token_ids:
            self._buffer.append(token_id)
            if len(self._buffer) >= self.shard_size:
                self._flush()

    def close(self) -> list[ShardInfo]:
        """Flush any buffered tokens and return the shards written (idempotent)."""
        if not self._closed:
            self._flush()
            self._closed = True
        return list(self._shards)

    def _flush(self) -> None:
        if not self._buffer:
            return
        filename = f"{self.split}_{len(self._shards):05d}.bin"
        count = write_shard(self.output_dir / filename, self._buffer, self.dtype)
        self._shards.append(ShardInfo(filename=filename, split=self.split, num_tokens=count))
        self._buffer.clear()


def pack_tokens(
    tokens: Iterable[int],
    output_dir: Path,
    split: str,
    dtype: np.dtype[Any],
    shard_size: int,
) -> list[ShardInfo]:
    """Pack ``tokens`` into ``shard_size``-token shards under ``output_dir``.

    Thin convenience wrapper over :class:`ShardWriter` for the single-split case.
    An empty stream writes no files and returns an empty list.
    """
    writer = ShardWriter(output_dir, split, dtype, shard_size)
    writer.append(tokens)
    return writer.close()


def write_manifest(
    output_dir: Path,
    dtype: np.dtype[Any],
    vocab_size: int,
    shard_size: int,
    shards: Iterable["ShardInfo"],
) -> Path:
    """Write ``manifest.json`` describing every shard, grouped by split."""
    splits: dict[str, dict[str, Any]] = {}
    for shard in shards:
        entry = splits.setdefault(shard.split, {"shards": [], "num_tokens": 0})
        entry["shards"].append(shard.filename)
        entry["num_tokens"] += shard.num_tokens

    payload = {
        "version": MANIFEST_VERSION,
        "dtype": np.dtype(dtype).name,
        "vocab_size": vocab_size,
        "shard_size": shard_size,
        "splits": splits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_manifest(output_dir: Path) -> dict[str, Any]:
    """Load ``manifest.json`` from a directory (or an explicit file path)."""
    path = output_dir / MANIFEST_NAME if output_dir.is_dir() else output_dir
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


class PackedTokenDataset:
    """A single shard exposed as contiguous ``(x, y)`` next-token examples."""

    def __init__(self, shard_path: Path, context_length: int, dtype: str = "uint16") -> None:
        if context_length < 1:
            raise ValueError(f"context_length must be positive, got {context_length}")

        self.shard_path = Path(shard_path)
        self.context_length = context_length
        self.dtype = np.dtype(dtype)
        self.tokens: np.memmap[Any, np.dtype[Any]] = np.memmap(
            self.shard_path, dtype=self.dtype, mode="r"
        )

        num_tokens = int(self.tokens.shape[0])
        # Each example spans ``context_length + 1`` tokens (inputs plus the final
        # shifted target), and blocks do not overlap.
        self._num_examples = (num_tokens - 1) // context_length
        if self._num_examples < 1:
            raise ValueError(
                f"Shard {self.shard_path} has {num_tokens} tokens; need at least "
                f"{context_length + 1} for one example of context_length={context_length}"
            )

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._num_examples
        if not 0 <= index < self._num_examples:
            raise IndexError(f"Index {index} out of range for {self._num_examples} examples")

        start = index * self.context_length
        block = np.asarray(self.tokens[start : start + self.context_length + 1], dtype=np.int64)
        x = torch.from_numpy(block[:-1])
        y = torch.from_numpy(block[1:])
        return x, y


class PackedSplit:
    """Every shard of one split, indexed as a single sequence of examples.

    Training reads from many shards, so this concatenates the per-shard
    :class:`PackedTokenDataset` views and maps a global example index to the
    shard that holds it. Shards too short to yield one example are skipped
    rather than raising, so a small trailing shard cannot break a run.
    """

    def __init__(self, packed_dir: Path, split: str, context_length: int) -> None:
        self.packed_dir = Path(packed_dir)
        self.split = split
        self.context_length = context_length

        manifest = load_manifest(self.packed_dir)
        entry = manifest.get("splits", {}).get(split)
        if not entry or not entry.get("shards"):
            raise ValueError(f"No {split!r} shards found in manifest under {self.packed_dir}")

        dtype = str(manifest["dtype"])
        self.shards: list[PackedTokenDataset] = []
        # Cumulative example counts, so a global index maps to a shard via bisect.
        self._cumulative: list[int] = []
        total = 0
        for filename in entry["shards"]:
            path = self.packed_dir / filename
            try:
                shard = PackedTokenDataset(path, context_length=context_length, dtype=dtype)
            except ValueError:
                continue
            self.shards.append(shard)
            total += len(shard)
            self._cumulative.append(total)

        if not self.shards:
            raise ValueError(
                f"No {split!r} shard under {self.packed_dir} holds an example of "
                f"context_length={context_length}"
            )
        self._num_examples = total

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._num_examples
        if not 0 <= index < self._num_examples:
            raise IndexError(f"Index {index} out of range for {self._num_examples} examples")

        shard_index = bisect.bisect_right(self._cumulative, index)
        previous = self._cumulative[shard_index - 1] if shard_index else 0
        return self.shards[shard_index][index - previous]

    @property
    def num_tokens(self) -> int:
        """Tokens reachable as examples (excludes each shard's unused remainder)."""
        return self._num_examples * self.context_length
