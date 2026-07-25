"""Reading documents out of a corpus directory."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

TEXT_SUFFIXES = frozenset({".txt", ".parquet"})


def iter_documents(input_dir: Path) -> Iterator[str]:
    """Yield documents from every ``.txt`` and ``.parquet`` file under ``input_dir``."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(path for path in input_dir.rglob("*") if path.suffix in TEXT_SUFFIXES)
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
