"""Corpus sizing: converting a token budget into a download size."""

from pathlib import Path

import pytest

from datasets.sizing import (
    bytes_for_tokens,
    measure_vocab_size,
    split_sample,
    take_bytes,
    tokens_for_bytes,
)
from datasets.text import iter_documents

DOCS = [f"the quick brown fox number {index} jumps over the lazy dog\n" * 4 for index in range(30)]


# ------------------------------------------------------------------- splitting


def test_split_is_disjoint_and_covers_every_document() -> None:
    train, holdout = split_sample(DOCS, holdout_fraction=0.1)

    assert len(train) + len(holdout) == len(DOCS)
    assert not set(train) & set(holdout)


def test_split_is_stable_across_calls() -> None:
    assert split_sample(DOCS, 0.2) == split_sample(DOCS, 0.2)


def test_split_drops_empty_documents() -> None:
    train, holdout = split_sample(["a", "", "b", "", "c", "d"], 0.5)

    assert "" not in train and "" not in holdout
    assert len(train) + len(holdout) == 4


def test_split_keeps_both_sides_non_empty_on_a_tiny_sample() -> None:
    train, holdout = split_sample(["a", "b"], 0.5)

    assert train and holdout


def test_split_rejects_impossible_fractions() -> None:
    for fraction in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="holdout_fraction"):
            split_sample(DOCS, fraction)


def test_split_needs_two_documents() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        split_sample(["only one"], 0.5)


# ------------------------------------------------------------------- sampling


def test_take_bytes_stops_near_the_cap() -> None:
    taken = list(take_bytes(DOCS, max_bytes=200))

    assert 0 < len(taken) < len(DOCS)
    assert sum(len(doc.encode()) for doc in taken) >= 200


def test_take_bytes_yields_everything_when_the_cap_is_large() -> None:
    assert list(take_bytes(DOCS, max_bytes=10**9)) == DOCS


def test_take_bytes_rejects_a_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        list(take_bytes(DOCS, 0))


# ----------------------------------------------------------------- projection


def test_bytes_for_tokens_is_the_inverse_of_tokens_for_bytes() -> None:
    assert bytes_for_tokens(4.0, 1_000) == 4_000
    assert tokens_for_bytes(4.0, 4_000) == 1_000


def test_projection_rejects_nonpositive_compression() -> None:
    with pytest.raises(ValueError, match="bytes_per_token"):
        bytes_for_tokens(0.0, 100)


def test_projection_rejects_nonpositive_targets() -> None:
    with pytest.raises(ValueError, match="target_tokens"):
        bytes_for_tokens(4.0, 0)


# ---------------------------------------------------------------- measurement


def test_measurement_reports_compression_above_one_byte_per_token() -> None:
    train, holdout = split_sample(DOCS, 0.2)

    measurement = measure_vocab_size(train, holdout, vocab_size=512)

    assert measurement.vocab_size == 512
    assert measurement.eval_documents == len(holdout)
    assert measurement.eval_bytes == sum(len(doc.encode()) for doc in holdout)
    # Merges must compress, so a token covers more than a single byte.
    assert measurement.bytes_per_token > 1.0
    assert measurement.bytes_per_token == measurement.eval_bytes / measurement.eval_tokens


def test_larger_vocabulary_compresses_at_least_as_well() -> None:
    train, holdout = split_sample(DOCS, 0.2)

    small = measure_vocab_size(train, holdout, vocab_size=300)
    large = measure_vocab_size(train, holdout, vocab_size=600)

    assert large.bytes_per_token >= small.bytes_per_token


def test_measurement_projects_source_and_packed_sizes() -> None:
    train, holdout = split_sample(DOCS, 0.2)
    measurement = measure_vocab_size(train, holdout, vocab_size=512)

    target = 1_000_000
    assert measurement.source_bytes_for(target) == bytes_for_tokens(
        measurement.bytes_per_token, target
    )
    # A 512-token vocabulary fits in uint16, so packing costs 2 bytes per token.
    assert measurement.bytes_per_token_id == 2
    assert measurement.packed_bytes_for(target) == 2 * target


def test_measurement_requires_both_sides() -> None:
    with pytest.raises(ValueError, match="train_texts"):
        measure_vocab_size([], ["x"], vocab_size=300)
    with pytest.raises(ValueError, match="eval_texts"):
        measure_vocab_size(["x"], [], vocab_size=300)


def test_measurement_handles_special_tokens() -> None:
    train, holdout = split_sample(DOCS, 0.2)

    measurement = measure_vocab_size(
        train, holdout, vocab_size=512, special_tokens=["<|bos|>", "<|eos|>"]
    )

    assert measurement.eval_tokens > 0


# --------------------------------------------------------------- text reading


def test_iter_documents_reads_txt_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("nope", encoding="utf-8")

    assert list(iter_documents(tmp_path)) == ["alpha", "beta"]


def test_iter_documents_requires_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input directory not found"):
        list(iter_documents(tmp_path / "absent"))


def test_iter_documents_requires_matching_files(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("nope", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No .txt or .parquet"):
        list(iter_documents(tmp_path))
