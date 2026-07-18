"""Phase 3: packed, memory-mapped next-token datasets.

These tests pin down the shard format and indexing before the implementation:
lossless shard round-trips, ``x``/``y`` next-token alignment, shard-boundary
behaviour, and the writer's manifest.
"""

from pathlib import Path

import numpy as np
import pytest

from datasets.packed import PackedTokenDataset, load_manifest, write_shards


def _make_shard(path: Path, tokens: list[int], dtype: str = "uint16") -> None:
    np.save(path, np.array(tokens, dtype=dtype))


def test_dataset_length_is_tokens_minus_context(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npy"
    _make_shard(shard, list(range(10)))
    ds = PackedTokenDataset(shard, context_length=4)
    # Valid start offsets i satisfy i + context_length < len(tokens).
    assert len(ds) == 10 - 4


def test_xy_alignment_is_next_token(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npy"
    tokens = [5, 6, 7, 8, 9, 10, 11, 12]
    _make_shard(shard, tokens)
    ds = PackedTokenDataset(shard, context_length=3)

    x0, y0 = ds[0]
    assert list(x0) == [5, 6, 7]
    assert list(y0) == [6, 7, 8]  # y is x shifted by one token

    x_last, y_last = ds[len(ds) - 1]
    assert list(x_last) == [9, 10, 11]
    assert list(y_last) == [10, 11, 12]


def test_getitem_returns_int64_arrays(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npy"
    _make_shard(shard, list(range(20)))
    ds = PackedTokenDataset(shard, context_length=5)
    x, y = ds[0]
    assert x.dtype == np.int64
    assert y.dtype == np.int64
    assert x.shape == (5,)
    assert y.shape == (5,)


def test_shard_too_small_has_no_examples(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npy"
    _make_shard(shard, [1, 2, 3])
    ds = PackedTokenDataset(shard, context_length=3)
    assert len(ds) == 0


def test_out_of_range_index_raises(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npy"
    _make_shard(shard, list(range(10)))
    ds = PackedTokenDataset(shard, context_length=4)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_negative_index_wraps(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npy"
    tokens = list(range(10))
    _make_shard(shard, tokens)
    ds = PackedTokenDataset(shard, context_length=4)
    assert [int(v) for v in ds[-1][0]] == [int(v) for v in ds[len(ds) - 1][0]]


def test_write_shards_splits_on_boundary(tmp_path: Path) -> None:
    # 25 tokens across two documents, shard_size 10 -> shards of 10, 10, 5.
    docs = [list(range(13)), list(range(13, 25))]
    manifest = write_shards(docs, tmp_path, shard_size=10, dtype="uint16")

    assert manifest["total_tokens"] == 25
    assert [s["num_tokens"] for s in manifest["shards"]] == [10, 10, 5]
    assert manifest["dtype"] == "uint16"

    # Concatenating all shards reproduces the flat token stream in order.
    flat: list[int] = []
    for shard in manifest["shards"]:
        flat.extend(int(v) for v in np.load(tmp_path / shard["path"]))
    assert flat == list(range(25))


def test_write_shards_manifest_roundtrip(tmp_path: Path) -> None:
    write_shards([[1, 2, 3, 4]], tmp_path, shard_size=100, dtype="uint16", extra={"note": "hi"})
    manifest = load_manifest(tmp_path / "manifest.json")
    assert manifest["total_tokens"] == 4
    assert manifest["note"] == "hi"
    assert len(manifest["shards"]) == 1


def test_written_shard_reads_back_through_dataset(tmp_path: Path) -> None:
    manifest = write_shards([list(range(100))], tmp_path, shard_size=1000, dtype="uint16")
    shard_path = tmp_path / manifest["shards"][0]["path"]
    ds = PackedTokenDataset(shard_path, context_length=8)
    x, y = ds[0]
    assert list(x) == list(range(8))
    assert list(y) == list(range(1, 9))
