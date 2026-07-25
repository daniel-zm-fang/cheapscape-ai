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

import heapq
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

_DEAD = -1

Pair = tuple[int, int]


def _link(size: int, boundaries: list[int]) -> tuple[list[int], list[int]]:
    """Build prev/next index arrays, unlinked at each document boundary."""
    previous = [index - 1 for index in range(size)]
    following = [index + 1 for index in range(size)]
    following[size - 1] = _DEAD
    start = 0
    for end in boundaries:
        if end > start:
            previous[start] = _DEAD
            following[end - 1] = _DEAD
        start = end
    return previous, following


def _index_pairs(tokens: list[int], following: list[int]) -> dict[Pair, set[int]]:
    """Map each adjacent pair to the left-hand positions where it occurs."""
    sites: dict[Pair, set[int]] = {}
    for index in range(len(tokens)):
        after = following[index]
        if after != _DEAD:
            sites.setdefault((tokens[index], tokens[after]), set()).add(index)
    return sites


def _apply_merge(
    pair: Pair,
    new_id: int,
    tokens: list[int],
    previous: list[int],
    following: list[int],
    sites: dict[Pair, set[int]],
) -> set[Pair]:
    """Replace every non-overlapping occurrence of ``pair``, left to right.

    Returns the pairs whose occurrence sets changed, so the caller can refresh
    whatever ordering it maintains over them.
    """
    first, second = pair
    touched: set[Pair] = set()

    def forget(key: Pair, position: int) -> None:
        where = sites.get(key)
        if where is not None and position in where:
            where.discard(position)
            if not where:
                del sites[key]
            touched.add(key)

    def remember(key: Pair, position: int) -> None:
        sites.setdefault(key, set()).add(position)
        touched.add(key)

    # A snapshot in ascending order gives the same greedy left-to-right result
    # as a single scan; sites consumed by an earlier merge fail the liveness
    # check below.
    for left in sorted(sites.get(pair, ())):
        if tokens[left] != first:
            continue
        right = following[left]
        if right == _DEAD or tokens[right] != second:
            continue

        before = previous[left]
        after = following[right]
        if before != _DEAD:
            forget((tokens[before], tokens[left]), before)
        forget((tokens[left], tokens[right]), left)
        if after != _DEAD:
            forget((tokens[right], tokens[after]), right)

        tokens[left] = new_id
        tokens[right] = _DEAD
        following[left] = after
        if after != _DEAD:
            previous[after] = left

        if before != _DEAD:
            remember((tokens[before], new_id), before)
        if after != _DEAD:
            remember((new_id, tokens[after]), left)

    sites.pop(pair, None)
    return touched


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
        """Learn a deterministic merge table from a stream of training texts.

        Training memory is proportional to corpus size, because every adjacent
        pair position is indexed. Train on a sample of a large corpus rather
        than the whole thing.
        """
        min_vocab = 256 + self.num_special
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size must be at least {min_vocab} (256 bytes + "
                f"{self.num_special} special tokens), got {vocab_size}"
            )

        num_merges = vocab_size - min_vocab

        # Split special tokens out of the corpus so they never take part in a
        # merge, then flatten the remainder into one byte-id array. Document
        # ends are recorded so no pair is ever counted across a boundary.
        tokens: list[int] = []
        boundaries: list[int] = []
        for text in texts:
            for is_special, chunk in self._split_special(text):
                if not is_special and chunk:
                    tokens.extend(chunk.encode("utf-8"))
                    boundaries.append(len(tokens))

        merges: dict[Pair, int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_id = min_vocab  # merges start after the reserved special ids

        for pair in self._iter_merges(tokens, boundaries, num_merges):
            merges[pair] = next_id
            vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
            next_id += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = vocab_size

    def _iter_merges(
        self, tokens: list[int], boundaries: list[int], num_merges: int
    ) -> Iterator[Pair]:
        """Yield up to ``num_merges`` pairs, most frequent first.

        The corpus is held as a doubly linked list so applying a merge touches
        only the sites where that pair occurs, and an index from pair to those
        sites keeps the counts current. Recounting the whole corpus per merge
        instead makes a realistic vocabulary size take hours.
        """
        if len(tokens) < 2:
            return

        previous, following = _link(len(tokens), boundaries)
        sites = _index_pairs(tokens, following)

        # Ordered by (-count, a, b), which reproduces "most frequent, ties to
        # the lowest pair". Entries are never removed; a popped entry whose
        # count no longer matches is simply stale.
        heap = [(-len(where), pair[0], pair[1]) for pair, where in sites.items()]
        heapq.heapify(heap)

        next_id = 256 + self.num_special
        for _ in range(num_merges):
            pair = self._pop_best(heap, sites)
            if pair is None:
                return
            touched = _apply_merge(pair, next_id, tokens, previous, following, sites)
            for key in touched:
                where = sites.get(key)
                if where:
                    heapq.heappush(heap, (-len(where), key[0], key[1]))
            yield pair
            next_id += 1

    @staticmethod
    def _pop_best(heap: list[tuple[int, int, int]], sites: dict[Pair, set[int]]) -> Pair | None:
        """Return the most frequent live pair, discarding stale heap entries."""
        while heap:
            negative_count, first, second = heapq.heappop(heap)
            pair = (first, second)
            where = sites.get(pair)
            if where and len(where) == -negative_count:
                return pair
        return None

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
        """Apply merges to a byte string, always taking the earliest merge first.

        Same linked-list index as training, ordered by merge id instead of
        frequency: the merge learned earliest has the lowest id and is applied
        everywhere before any later one is considered. Rescanning the whole
        chunk per merge instead is quadratic in document length, which packing
        a real corpus cannot afford.
        """
        tokens = list(data)
        size = len(tokens)
        if size < 2 or not self.merges:
            return tokens

        previous, following = _link(size, [size])
        sites = _index_pairs(tokens, following)
        heap = [
            (rank, pair[0], pair[1])
            for pair in sites
            if (rank := self.merges.get(pair)) is not None
        ]
        heapq.heapify(heap)

        while heap:
            rank, first, second = heapq.heappop(heap)
            pair = (first, second)
            if not sites.get(pair):
                continue
            touched = _apply_merge(pair, rank, tokens, previous, following, sites)
            for key in touched:
                new_rank = self.merges.get(key)
                if new_rank is not None and sites.get(key):
                    heapq.heappush(heap, (new_rank, key[0], key[1]))

        ids: list[int] = []
        index = 0
        while index != _DEAD:
            ids.append(tokens[index])
            index = following[index]
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
