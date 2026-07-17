"""Phase 2 entry point: learn BPE merges and save tokenizer artifacts.

Reads text from the configured ``input_dir`` (``.txt`` files, or ``.parquet``
files with a ``text`` column when ``pyarrow`` is available), trains the
byte-level BPE tokenizer, and writes ``tokenizer.json`` to ``output_dir``.
"""

import argparse
from pathlib import Path

from config import load_config
from datasets.corpus import iter_documents
from tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "tokenizer.yaml",
        help="Path to tokenizer config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        required_keys=["vocab_size", "special_tokens", "input_dir", "output_dir"],
    )

    input_dir = _resolve(cfg["input_dir"])
    output_dir = _resolve(cfg["output_dir"])
    vocab_size = int(cfg["vocab_size"])
    special_tokens = list(cfg["special_tokens"])

    tokenizer = BPETokenizer(special_tokens=special_tokens)
    tokenizer.train(iter_documents(input_dir), vocab_size=vocab_size)
    artifact = tokenizer.save(output_dir)

    print(
        f"Trained tokenizer: {len(tokenizer.merges)} merges, "
        f"vocab_size={tokenizer.vocab_size}, specials={special_tokens}"
    )
    print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
