"""Contracts for your byte-pair encoding tokenizer.

Do not fill this in until its round-trip and merge-rule tests exist.
"""

from collections.abc import Iterable


class BPETokenizer:
    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        """Learn a deterministic merge table from a stream of training texts."""
        raise NotImplementedError("Phase 2 exercise: implement BPE vocabulary learning.")

    def encode(self, text: str) -> list[int]:
        """Return token ids for text, including intentional special-token rules."""
        raise NotImplementedError("Phase 2 exercise: implement merge application.")

    def decode(self, token_ids: list[int]) -> str:
        """Reconstruct text from token ids according to the tokenizer contract."""
        raise NotImplementedError("Phase 2 exercise: implement decoding.")
