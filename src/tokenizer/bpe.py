"""A byte-level byte-pair encoding (BPE) tokenizer.

The tokenizer is intentionally simple and inspectable, in the spirit of the
project: it operates on raw UTF-8 bytes so that *any* text round-trips
losslessly, learns a deterministic merge table, and reserves fixed ids for
special tokens.

Id layout::

    0 .. 255                      one id per raw byte (the base vocabulary)
    256 .. 256 + S - 1           the S special tokens, in declaration order
    256 + S .. vocab_size - 1    learned merges, in the order they were learned

Because the base is the 256 bytes, decoding is always defined and encoding is
lossless for arbitrary Unicode input.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

_INF = float("inf")

Pair = tuple[int, int]


class BPETokenizer:
    """Learn and apply a deterministic byte-level BPE merge table."""

    def __init__(self, special_tokens: list[str] | None = None) -> None:
        specials = special_tokens if special_tokens is not None else []
        if len(set(specials)) != len(specials):
            raise ValueError("Special tokens must be unique")

        self.special_tokens: dict[str, int] = {
            token: 256 + index for index, token in enumerate(specials)
        }
        self._inverse_special: dict[int, str] = {
            token_id: token for token, token_id in self.special_tokens.items()
        }
        self.num_special: int = len(self.special_tokens)

        # Base vocabulary: one entry per byte. Merges extend this in `train`.
        self.merges: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size: int = 256 + self.num_special

    # ------------------------------------------------------------------ train

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        """Learn a deterministic merge table from a stream of training texts."""
        min_vocab = 256 + self.num_special
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size must be at least {min_vocab} (256 bytes + "
                f"{self.num_special} special tokens), got {vocab_size}"
            )

        num_merges = vocab_size - min_vocab

        # Split special tokens out of the corpus so they never take part in a
        # merge, then work on lists of byte ids.
        sequences: list[list[int]] = []
        for text in texts:
            for is_special, chunk in self._split_special(text):
                if not is_special and chunk:
                    sequences.append(list(chunk.encode("utf-8")))

        merges: dict[Pair, int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_id = min_vocab  # merges start after the reserved special ids

        for _ in range(num_merges):
            counts = self._count_pairs(sequences)
            if not counts:
                break
            pair = self._best_pair(counts)
            merges[pair] = next_id
            vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
            sequences = [self._merge(seq, pair, next_id) for seq in sequences]
            next_id += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = vocab_size

    # ----------------------------------------------------------------- encode

    def encode(self, text: str) -> list[int]:
        """Return token ids for text, emitting reserved ids for special tokens."""
        ids: list[int] = []
        for is_special, chunk in self._split_special(text):
            if is_special:
                ids.append(self.special_tokens[chunk])
            elif chunk:
                ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Reconstruct text from token ids, decoding the byte stream as UTF-8."""
        buffer = bytearray()
        for token_id in token_ids:
            if token_id in self._inverse_special:
                buffer.extend(self._inverse_special[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                buffer.extend(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token id: {token_id}")
        return buffer.decode("utf-8", errors="replace")

    def _encode_chunk(self, data: bytes) -> list[int]:
        """Apply merges to a byte string, always taking the earliest merge first."""
        ids = list(data)
        while len(ids) >= 2:
            pairs = {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}
            # The merge learned earliest has the lowest id; apply it first.
            pair = min(pairs, key=lambda p: self.merges.get(p, _INF))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    # ------------------------------------------------------------- persistence

    def save(self, output_dir: Path) -> Path:
        """Write the merge table and metadata to ``output_dir/tokenizer.json``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            # Ordered by learning order; ids are implied by position.
            "merges": [[a, b] for (a, b) in self.merges],
        }
        path = output_dir / "tokenizer.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Reconstruct a tokenizer from an artifact written by :meth:`save`."""
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        specials = sorted(
            payload["special_tokens"], key=lambda token: payload["special_tokens"][token]
        )
        tokenizer = cls(special_tokens=specials)

        merges: dict[Pair, int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_id = 256 + tokenizer.num_special
        for a, b in payload["merges"]:
            merges[(a, b)] = next_id
            vocab[next_id] = vocab[a] + vocab[b]
            next_id += 1

        tokenizer.merges = merges
        tokenizer.vocab = vocab
        tokenizer.vocab_size = payload["vocab_size"]
        return tokenizer

    # --------------------------------------------------------------- internals

    def _split_special(self, text: str) -> Iterator[tuple[bool, str]]:
        """Yield ``(is_special, chunk)`` segments, isolating special-token strings."""
        if not self.special_tokens:
            yield (False, text)
            return

        specials = sorted(self.special_tokens, key=len, reverse=True)
        remaining = text
        while remaining:
            hit_index = len(remaining)
            hit_token = ""
            for token in specials:
                found = remaining.find(token)
                if found != -1 and found < hit_index:
                    hit_index = found
                    hit_token = token
            if not hit_token:
                yield (False, remaining)
                return
            if hit_index > 0:
                yield (False, remaining[:hit_index])
            yield (True, hit_token)
            remaining = remaining[hit_index + len(hit_token) :]

    @staticmethod
    def _count_pairs(sequences: list[list[int]]) -> dict[Pair, int]:
        counts: dict[Pair, int] = {}
        for seq in sequences:
            for pair in zip(seq, seq[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _best_pair(counts: dict[Pair, int]) -> Pair:
        """Most frequent pair; ties broken by the lexicographically smallest pair."""
        return max(counts, key=lambda pair: (counts[pair], -pair[0], -pair[1]))

    @staticmethod
    def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
        """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
        merged: list[int] = []
        i = 0
        length = len(ids)
        while i < length:
            if i < length - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged
