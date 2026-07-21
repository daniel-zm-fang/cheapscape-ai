"""Phase 3: packed, memory-mapped next-token dataset contract.

These tests pin down the behaviour before the implementation exists: how a token
stream is packed into fixed-size shards, how the memory-mapped dataset turns a
shard into contiguous ``(x, y)`` next-token examples, the id dtype chosen for a
vocabulary, and the manifest that ties shards together.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.packed import (
    PackedTokenDataset,
    ShardInfo,
    ShardWriter,
    dtype_for_vocab,
    load_manifest,
    pack_tokens,
    write_manifest,
    write_shard,
)

# --------------------------------------------------------------- dtype selection


def test_dtype_uint16_for_small_vocab() -> None:
    """A vocabulary that fits in 16 bits is stored as uint16 to halve shard size."""
    assert dtype_for_vocab(256) == np.dtype(np.uint16)
    assert dtype_for_vocab(4096) == np.dtype(np.uint16)
    assert dtype_for_vocab(2**16) == np.dtype(np.uint16)


def test_dtype_uint32_for_large_vocab() -> None:
    """A vocabulary that overflows uint16 falls back to uint32."""
    assert dtype_for_vocab(2**16 + 1) == np.dtype(np.uint32)


def test_dtype_rejects_nonpositive_vocab() -> None:
    with pytest.raises(ValueError):
        dtype_for_vocab(0)


# ------------------------------------------------------------------- write_shard


def test_write_shard_roundtrips_via_numpy(tmp_path: Path) -> None:
    tokens = [0, 1, 255, 256, 4095]
    path = tmp_path / "shard.bin"

    count = write_shard(path, tokens, np.dtype(np.uint16))

    assert count == len(tokens)
    restored = np.fromfile(path, dtype=np.uint16).tolist()
    assert restored == tokens


# ------------------------------------------------------- PackedTokenDataset shape


def _write(tmp_path: Path, tokens: list[int], dtype: np.dtype = np.dtype(np.uint16)) -> Path:
    path = tmp_path / "shard.bin"
    write_shard(path, tokens, dtype)
    return path


def test_dataset_length_uses_nonoverlapping_blocks(tmp_path: Path) -> None:
    """With ``n`` tokens and block ``T`` there are ``(n - 1) // T`` examples."""
    path = _write(tmp_path, list(range(10)))
    dataset = PackedTokenDataset(path, context_length=4)
    assert len(dataset) == 2  # (10 - 1) // 4


def test_dataset_examples_are_shifted_by_one(tmp_path: Path) -> None:
    """Example ``i`` is a contiguous block; ``y`` is ``x`` shifted one token right."""
    path = _write(tmp_path, list(range(10)))
    dataset = PackedTokenDataset(path, context_length=4)

    x0, y0 = dataset[0]
    assert x0.tolist() == [0, 1, 2, 3]
    assert y0.tolist() == [1, 2, 3, 4]

    x1, y1 = dataset[1]
    assert x1.tolist() == [4, 5, 6, 7]
    assert y1.tolist() == [5, 6, 7, 8]


def test_dataset_returns_long_tensors(tmp_path: Path) -> None:
    """Ids come back as ``torch.long`` so they can index embeddings directly."""
    path = _write(tmp_path, list(range(10)))
    x, y = PackedTokenDataset(path, context_length=4)[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.shape == (4,)
    assert torch.equal(x[1:], y[:-1])


def test_dataset_supports_negative_index(tmp_path: Path) -> None:
    path = _write(tmp_path, list(range(10)))
    dataset = PackedTokenDataset(path, context_length=4)
    x_last, y_last = dataset[-1]
    x_pos, y_pos = dataset[len(dataset) - 1]
    assert torch.equal(x_last, x_pos)
    assert torch.equal(y_last, y_pos)


def test_dataset_index_out_of_range_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, list(range(10)))
    dataset = PackedTokenDataset(path, context_length=4)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_dataset_rejects_bad_context_length(tmp_path: Path) -> None:
    path = _write(tmp_path, list(range(10)))
    with pytest.raises(ValueError):
        PackedTokenDataset(path, context_length=0)


def test_dataset_rejects_shard_too_small(tmp_path: Path) -> None:
    """A shard needs at least ``context_length + 1`` tokens for one example."""
    path = _write(tmp_path, [1, 2, 3])
    with pytest.raises(ValueError):
        PackedTokenDataset(path, context_length=4)


def test_dataset_preserves_large_ids_with_uint32(tmp_path: Path) -> None:
    tokens = [0, 70000, 1, 131071]
    path = _write(tmp_path, tokens, np.dtype(np.uint32))
    dataset = PackedTokenDataset(path, context_length=2, dtype="uint32")
    x, y = dataset[0]
    assert x.tolist() == [0, 70000]
    assert y.tolist() == [70000, 1]


# ------------------------------------------------------------------ pack_tokens


def test_pack_tokens_splits_into_fixed_size_shards(tmp_path: Path) -> None:
    shards = pack_tokens(
        iter(range(10)), tmp_path, split="train", dtype=np.dtype(np.uint16), shard_size=4
    )

    assert [s.num_tokens for s in shards] == [4, 4, 2]
    assert [s.filename for s in shards] == [
        "train_00000.bin",
        "train_00001.bin",
        "train_00002.bin",
    ]
    assert all(s.split == "train" for s in shards)

    packed: list[int] = []
    for shard in shards:
        packed.extend(np.fromfile(tmp_path / shard.filename, dtype=np.uint16).tolist())
    assert packed == list(range(10))


def test_pack_tokens_empty_stream_writes_nothing(tmp_path: Path) -> None:
    shards = pack_tokens(iter([]), tmp_path, split="val", dtype=np.dtype(np.uint16), shard_size=4)
    assert shards == []
    assert list(tmp_path.glob("*.bin")) == []


# ------------------------------------------------------------------ ShardWriter


def test_shard_writer_streams_multiple_appends(tmp_path: Path) -> None:
    """Repeated appends accumulate into shards without gaps or overlaps."""
    writer = ShardWriter(tmp_path, "train", np.dtype(np.uint16), shard_size=4)
    writer.append([0, 1, 2])
    writer.append([3, 4, 5, 6, 7])
    shards = writer.close()

    assert [s.num_tokens for s in shards] == [4, 4]
    packed: list[int] = []
    for shard in shards:
        packed.extend(np.fromfile(tmp_path / shard.filename, dtype=np.uint16).tolist())
    assert packed == list(range(8))


def test_shard_writer_close_is_idempotent(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "train", np.dtype(np.uint16), shard_size=4)
    writer.append([0, 1])
    first = writer.close()
    second = writer.close()
    assert first == second
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_shard_writer_rejects_append_after_close(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "train", np.dtype(np.uint16), shard_size=4)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.append([0])


# --------------------------------------------------------------------- manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    shards = [
        ShardInfo(filename="train_00000.bin", split="train", num_tokens=4),
        ShardInfo(filename="val_00000.bin", split="val", num_tokens=2),
    ]

    path = write_manifest(
        tmp_path, dtype=np.dtype(np.uint16), vocab_size=4096, shard_size=4, shards=shards
    )
    assert path == tmp_path / "manifest.json"

    manifest = load_manifest(tmp_path)
    assert manifest["dtype"] == "uint16"
    assert manifest["vocab_size"] == 4096
    assert manifest["shard_size"] == 4
    assert manifest["splits"]["train"]["num_tokens"] == 4
    assert manifest["splits"]["train"]["shards"] == ["train_00000.bin"]
    assert manifest["splits"]["val"]["num_tokens"] == 2
