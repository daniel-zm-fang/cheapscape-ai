"""Corpus document iteration shared by the tokenizer trainer and packer."""

from pathlib import Path

import pytest

from datasets.corpus import iter_documents


def test_iter_documents_reads_txt_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")

    assert list(iter_documents(tmp_path)) == ["first", "second"]


def test_iter_documents_ignores_other_suffixes(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "skip.md").write_text("skip", encoding="utf-8")

    assert list(iter_documents(tmp_path)) == ["keep"]


def test_iter_documents_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "nope"))


def test_iter_documents_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path))
