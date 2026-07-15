"""Phase 1: fixture acquisition."""

from pathlib import Path

import pytest

from datasets.acquire import acquire_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny_corpus.txt"


def test_acquire_fixture_copies_content(tmp_path: Path) -> None:
    source = {
        "type": "fixture",
        "path": str(FIXTURE),
        "id": "tiny_corpus_v0",
    }

    out = acquire_fixture(source, tmp_path / "raw")

    assert out == tmp_path / "raw" / "tiny_corpus_v0.txt"
    assert out.read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_acquire_fixture_missing_file_raises(tmp_path: Path) -> None:
    source = {
        "type": "fixture",
        "path": str(tmp_path / "does_not_exist.txt"),
        "id": "missing",
    }

    with pytest.raises(FileNotFoundError, match="Fixture not found"):
        acquire_fixture(source, tmp_path / "raw")


def test_acquire_fixture_missing_key_raises(tmp_path: Path) -> None:
    source = {"type": "fixture", "path": str(FIXTURE)}

    with pytest.raises(ValueError, match="missing required key"):
        acquire_fixture(source, tmp_path / "raw")
