"""Phase 2 entry point: learn BPE merges and save tokenizer artifacts.

Reads text from the configured ``input_dir`` (``.txt`` files, or ``.parquet``
files with a ``text`` column when ``pyarrow`` is available), trains the
byte-level BPE tokenizer, and writes ``tokenizer.json`` to ``output_dir``.
"""

# isort: off
import _bootstrap  # noqa: F401 -- must run before any other import

# isort: on

import argparse
from collections.abc import Iterator
from pathlib import Path

from config import load_config
from tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def iter_corpus(input_dir: Path) -> Iterator[str]:
    """Yield documents from every ``.txt`` and ``.parquet`` file under ``input_dir``."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(p for p in input_dir.rglob("*") if p.suffix in {".txt", ".parquet"})
    if not files:
        raise FileNotFoundError(f"No .txt or .parquet files found under {input_dir}")

    for file in files:
        if file.suffix == ".txt":
            yield file.read_text(encoding="utf-8")
        else:
            yield from _iter_parquet_text(file)


def _iter_parquet_text(file: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(f"Reading {file} requires pyarrow; install the 'data' extra.") from exc

    table = pq.read_table(file, columns=["text"])
    for value in table.column("text").to_pylist():
        if value is not None:
            yield str(value)


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
    tokenizer.train(iter_corpus(input_dir), vocab_size=vocab_size)
    artifact = tokenizer.save(output_dir)

    print(
        f"Trained tokenizer: {len(tokenizer.merges)} merges, "
        f"vocab_size={tokenizer.vocab_size}, specials={special_tokens}"
    )
    print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
