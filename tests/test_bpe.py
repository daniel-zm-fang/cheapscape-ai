"""Phase 2: byte-level BPE tokenizer contract.

These tests define the tokenizer's behaviour before the implementation exists:
lossless round-trips, deterministic merge learning, the exact merge rule used to
pick pairs, special-token handling, and artifact save/load.
"""

from pathlib import Path

import pytest

from tokenizer.bpe import BPETokenizer

# A small corpus with obvious repetition so merge behaviour is predictable.
CORPUS = [
    "the quick brown fox",
    "the quick brown fox jumps",
    "the lazy dog sleeps",
    "the the the the",
]


def test_roundtrip_without_training() -> None:
    """A fresh tokenizer must round-trip arbitrary text via raw bytes."""
    tok = BPETokenizer()
    for text in ["hello world", "", "  spaces  ", "tabs\tand\nnewlines"]:
        assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unicode_bytes() -> None:
    """Byte-level encoding must survive multi-byte UTF-8 (accents, emoji, CJK)."""
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    for text in ["café", "naïve résumé", "emoji: 🚀🔥", "日本語のテキスト"]:
        assert tok.decode(tok.encode(text)) == text


def test_roundtrip_after_training() -> None:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=320)
    for text in [*CORPUS, "an unseen sentence about foxes and dogs"]:
        assert tok.decode(tok.encode(text)) == text


def test_base_vocab_is_bytes() -> None:
    """Ids 0..255 always map to the 256 single bytes."""
    tok = BPETokenizer()
    for byte_value in range(256):
        assert tok.decode([byte_value]) == bytes([byte_value]).decode("utf-8", errors="replace")


def test_training_reduces_token_count() -> None:
    """Learned merges must compress repeated text below its raw byte length."""
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=320)
    text = "the quick brown fox"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_merge_rule_picks_most_frequent_pair_first() -> None:
    """The first merge is the most frequent adjacent byte pair (id 256)."""
    tok = BPETokenizer(special_tokens=[])
    # In "ababab", pair (a, b) occurs 3x and (b, a) occurs 2x.
    tok.train(["ababab"], vocab_size=257)
    a, b = ord("a"), ord("b")
    assert tok.merges == {(a, b): 256}
    # "abab" -> (ab)(ab) -> two tokens, both the new merge id.
    assert tok.encode("abab") == [256, 256]


def test_merge_rule_deterministic_tie_break() -> None:
    """Equal-frequency pairs break ties on the lowest (a, b), independent of order."""
    tok1 = BPETokenizer(special_tokens=[])
    tok2 = BPETokenizer(special_tokens=[])
    tok1.train(CORPUS, vocab_size=340)
    tok2.train(list(reversed(CORPUS)), vocab_size=340)
    assert tok1.merges == tok2.merges


def test_vocab_size_bounds_merges() -> None:
    """Merge count equals vocab_size minus bytes minus special tokens."""
    specials = ["<|bos|>", "<|eos|>"]
    tok = BPETokenizer(special_tokens=specials)
    # 278 = 256 bytes + 2 specials + 20 merges, well within what CORPUS supports.
    tok.train(CORPUS, vocab_size=278)
    assert len(tok.merges) == 278 - 256 - len(specials)
    assert tok.vocab_size == 278


def test_merges_never_exceed_budget() -> None:
    """A large target caps at the corpus's available pairs without overflowing."""
    specials = ["<|bos|>", "<|eos|>"]
    tok = BPETokenizer(special_tokens=specials)
    tok.train(CORPUS, vocab_size=400)
    assert len(tok.merges) <= 400 - 256 - len(specials)
    assert tok.vocab_size == 400


def test_vocab_size_too_small_raises() -> None:
    tok = BPETokenizer(special_tokens=["<|bos|>"])
    with pytest.raises(ValueError):
        tok.train(CORPUS, vocab_size=256)  # no room even for the special token


def test_special_tokens_have_reserved_ids() -> None:
    """Special tokens occupy fixed ids right after the byte range."""
    tok = BPETokenizer(special_tokens=["<|bos|>", "<|eos|>"])
    assert tok.special_tokens == {"<|bos|>": 256, "<|eos|>": 257}


def test_special_tokens_are_not_split() -> None:
    """A special-token string encodes to its single reserved id and round-trips."""
    tok = BPETokenizer(special_tokens=["<|bos|>", "<|eos|>"])
    tok.train(CORPUS, vocab_size=400)
    ids = tok.encode("<|bos|>the quick brown fox<|eos|>")
    assert ids[0] == 256
    assert ids[-1] == 257
    assert tok.decode(ids) == "<|bos|>the quick brown fox<|eos|>"


def test_special_token_ids_are_not_reused_by_merges() -> None:
    tok = BPETokenizer(special_tokens=["<|bos|>", "<|eos|>"])
    tok.train(CORPUS, vocab_size=400)
    assert min(tok.merges.values()) >= 258


def test_decode_unknown_id_raises() -> None:
    tok = BPETokenizer(special_tokens=["<|bos|>"])
    tok.train(CORPUS, vocab_size=300)
    with pytest.raises(ValueError):
        tok.decode([999_999])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    tok = BPETokenizer(special_tokens=["<|bos|>", "<|eos|>"])
    tok.train(CORPUS, vocab_size=360)
    artifact = tok.save(tmp_path)

    loaded = BPETokenizer.load(artifact)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.vocab_size == tok.vocab_size

    sample = "<|bos|>the quick brown fox<|eos|>"
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample
