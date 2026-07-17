"""Read a training corpus as a stream of documents.

Both the tokenizer trainer and the token packer consume text the same way, so
the file-walking logic lives here. Supports ``.txt`` files (one document per
file) and ``.parquet`` files with a ``text`` column (one document per row).
"""

from collections.abc import Iterator
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".parquet"}


def iter_documents(input_dir: Path) -> Iterator[str]:
    """Yield documents from every supported file under ``input_dir`` (sorted)."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(p for p in input_dir.rglob("*") if p.suffix in _TEXT_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No .txt or .parquet files found under {input_dir}")

    for file in files:
        if file.suffix == ".txt":
            yield file.read_text(encoding="utf-8")
        else:
            yield from _iter_parquet_text(file)


def _iter_parquet_text(file: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(f"Reading {file} requires pyarrow; install the 'data' extra.") from exc

    table = pq.read_table(file, columns=["text"])
    for value in table.column("text").to_pylist():
        if value is not None:
            yield str(value)
