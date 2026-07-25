"""Turn a corpus sample into the numbers that size a training run.

Raw bytes do not tell you how many tokens you have: that depends on the
tokenizer, which depends on the corpus. Measuring bytes-per-token on a sample
gives the conversion rate between "gigabytes to download" and "tokens to train
on", so the download can be sized from the compute budget instead of guessed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from datasets.packed import dtype_for_vocab
from tokenizer.bpe import BPETokenizer


@dataclass(frozen=True)
class VocabMeasurement:
    """What one candidate vocabulary size costs and buys."""

    vocab_size: int
    train_bytes: int
    eval_bytes: int
    eval_documents: int
    eval_tokens: int
    bytes_per_token: float
    train_seconds: float

    @property
    def bytes_per_token_id(self) -> int:
        """Width of one packed token id, which sets shard size on disk."""
        return int(dtype_for_vocab(self.vocab_size).itemsize)

    def source_bytes_for(self, target_tokens: int) -> int:
        """Raw corpus bytes needed to yield ``target_tokens``."""
        return bytes_for_tokens(self.bytes_per_token, target_tokens)

    def packed_bytes_for(self, target_tokens: int) -> int:
        """Packed shard bytes that ``target_tokens`` will occupy on disk."""
        if target_tokens < 1:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}")
        return target_tokens * self.bytes_per_token_id


def split_sample(
    documents: Iterable[str], holdout_fraction: float = 0.1
) -> tuple[list[str], list[str]]:
    """Split documents into (train, holdout) deterministically.

    Measuring compression on the text the tokenizer trained on flatters it, so
    the estimate comes from documents the merges never saw. Every ``n``-th
    document is held out rather than a random draw, so the split is stable
    without needing a seed.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")

    docs = [doc for doc in documents if doc]
    if len(docs) < 2:
        raise ValueError(f"Need at least 2 non-empty documents to split, got {len(docs)}")

    stride = max(round(1.0 / holdout_fraction), 2)
    train = [doc for index, doc in enumerate(docs) if index % stride]
    holdout = [doc for index, doc in enumerate(docs) if not index % stride]
    if not train:
        # A tiny sample can put everything in the holdout; keep both non-empty.
        train, holdout = holdout[1:], holdout[:1]
    return train, holdout


def measure_vocab_size(
    train_texts: list[str],
    eval_texts: list[str],
    *,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> VocabMeasurement:
    """Train a tokenizer at ``vocab_size`` and measure its compression."""
    if not train_texts:
        raise ValueError("train_texts must not be empty")
    if not eval_texts:
        raise ValueError("eval_texts must not be empty")

    started = time.perf_counter()
    tokenizer = BPETokenizer(special_tokens=list(special_tokens or []))
    tokenizer.train(train_texts, vocab_size=vocab_size)
    train_seconds = time.perf_counter() - started

    eval_bytes = sum(len(text.encode("utf-8")) for text in eval_texts)
    eval_tokens = sum(len(tokenizer.encode(text)) for text in eval_texts)
    if eval_tokens < 1:
        raise ValueError("Holdout produced no tokens; sample a larger corpus")

    return VocabMeasurement(
        vocab_size=vocab_size,
        train_bytes=sum(len(text.encode("utf-8")) for text in train_texts),
        eval_bytes=eval_bytes,
        eval_documents=len(eval_texts),
        eval_tokens=eval_tokens,
        bytes_per_token=eval_bytes / eval_tokens,
        train_seconds=train_seconds,
    )


def bytes_for_tokens(bytes_per_token: float, target_tokens: int) -> int:
    """Raw corpus bytes needed to produce ``target_tokens``."""
    if bytes_per_token <= 0.0:
        raise ValueError(f"bytes_per_token must be positive, got {bytes_per_token}")
    if target_tokens < 1:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    return round(bytes_per_token * target_tokens)


def tokens_for_bytes(bytes_per_token: float, corpus_bytes: int) -> int:
    """Tokens a corpus of ``corpus_bytes`` will yield."""
    if bytes_per_token <= 0.0:
        raise ValueError(f"bytes_per_token must be positive, got {bytes_per_token}")
    if corpus_bytes < 0:
        raise ValueError(f"corpus_bytes must be non-negative, got {corpus_bytes}")
    return int(corpus_bytes / bytes_per_token)


def take_bytes(documents: Iterable[str], max_bytes: int) -> Iterator[str]:
    """Yield documents until ``max_bytes`` of UTF-8 text has been emitted."""
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")

    used = 0
    for document in documents:
        if used >= max_bytes:
            return
        used += len(document.encode("utf-8"))
        yield document
