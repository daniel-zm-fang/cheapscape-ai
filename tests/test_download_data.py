"""Phase 1: source acquisition.

The HTTP tests run against a local server on a loopback port, so they exercise
real sockets, real Range requests, and real streaming without touching the
network.
"""

import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from datasets.acquire import (
    SourceRecord,
    acquire_fixture,
    acquire_http,
    acquire_source,
    load_lock,
    save_lock,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny_corpus.txt"

PAYLOAD = b"".join(f"line {index}\n".encode() for index in range(2000))
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    """Serves PAYLOAD, honouring Range unless the path opts out."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/missing":
            self.send_error(404)
            return

        body = PAYLOAD
        status = 200
        range_header = self.headers.get("Range")
        # /norange simulates a server that ignores Range and resends everything.
        if range_header and self.path != "/norange":
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            if start >= len(PAYLOAD):
                self.send_error(416)
                return
            body = PAYLOAD[start:]
            status = 206

        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            total = len(PAYLOAD)
            self.send_header("Content-Range", f"bytes {total - len(body)}-{total - 1}/{total}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Keep pytest output clean."""


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------- fixture


def test_acquire_fixture_copies_content(tmp_path: Path) -> None:
    source = {"type": "fixture", "path": str(FIXTURE), "id": "tiny_corpus_v0"}

    out = acquire_fixture(source, tmp_path / "raw")

    assert out == tmp_path / "raw" / "tiny_corpus_v0.txt"
    assert out.read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_acquire_fixture_missing_file_raises(tmp_path: Path) -> None:
    source = {"type": "fixture", "path": str(tmp_path / "does_not_exist.txt"), "id": "missing"}

    with pytest.raises(FileNotFoundError, match="Fixture not found"):
        acquire_fixture(source, tmp_path / "raw")


def test_acquire_fixture_missing_key_raises(tmp_path: Path) -> None:
    source = {"type": "fixture", "path": str(FIXTURE)}

    with pytest.raises(ValueError, match="missing required key"):
        acquire_fixture(source, tmp_path / "raw")


# ------------------------------------------------------------------------ http


def test_http_download_writes_full_payload(tmp_path: Path, server_url: str) -> None:
    source = {"type": "http", "id": "corpus", "url": f"{server_url}/corpus.txt"}

    out = acquire_http(source, tmp_path, chunk_size=1024)

    assert out.read_bytes() == PAYLOAD
    assert sha256_file(out) == PAYLOAD_SHA256


def test_http_download_enforces_declared_checksum(tmp_path: Path, server_url: str) -> None:
    source = {
        "type": "http",
        "id": "corpus",
        "url": f"{server_url}/corpus.txt",
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="Checksum mismatch"):
        acquire_http(source, tmp_path)
    # A corrupt download must not be left behind as a usable file.
    assert not (tmp_path / "corpus.txt").exists()
    assert not (tmp_path / "corpus.txt.part").exists()


def test_http_download_resumes_from_partial(tmp_path: Path, server_url: str) -> None:
    """A dropped multi-gigabyte download must continue, not restart."""
    partial = tmp_path / "corpus.txt.part"
    partial.write_bytes(PAYLOAD[:5000])
    source = {"type": "http", "id": "corpus", "url": f"{server_url}/corpus.txt"}

    out = acquire_http(source, tmp_path, chunk_size=1024)

    assert out.read_bytes() == PAYLOAD


def test_http_download_restarts_when_server_ignores_range(tmp_path: Path, server_url: str) -> None:
    partial = tmp_path / "norange.part"
    partial.write_bytes(PAYLOAD[:5000])
    source = {
        "type": "http",
        "id": "corpus",
        "url": f"{server_url}/norange",
        "filename": "norange",
    }

    out = acquire_http(source, tmp_path, chunk_size=1024)

    # Appending would have produced a longer, corrupt file.
    assert out.read_bytes() == PAYLOAD


def test_http_download_restarts_when_partial_is_past_the_end(
    tmp_path: Path, server_url: str
) -> None:
    partial = tmp_path / "corpus.txt.part"
    partial.write_bytes(PAYLOAD + b"extra")
    source = {"type": "http", "id": "corpus", "url": f"{server_url}/corpus.txt"}

    assert acquire_http(source, tmp_path, chunk_size=1024).read_bytes() == PAYLOAD


def test_http_rejects_non_http_scheme(tmp_path: Path) -> None:
    source = {"type": "http", "id": "local", "url": "file:///etc/passwd"}

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        acquire_http(source, tmp_path)


def test_http_rejects_filename_escaping_the_output_dir(tmp_path: Path) -> None:
    source = {
        "type": "http",
        "id": "escape",
        "url": "https://example.com/x",
        "filename": "../escaped.txt",
    }

    with pytest.raises(ValueError, match="Unsafe destination filename"):
        acquire_http(source, tmp_path)


def test_http_requires_a_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required key: url"):
        acquire_http({"type": "http", "id": "nourl"}, tmp_path)


# -------------------------------------------------------------------- lockfile


def test_lock_roundtrip(tmp_path: Path) -> None:
    record = SourceRecord(
        id="corpus",
        type="http",
        filename="corpus.txt",
        size_bytes=len(PAYLOAD),
        sha256=PAYLOAD_SHA256,
        url="https://example.com/corpus.txt",
        acquired_at="2026-07-25T00:00:00+00:00",
    )
    path = save_lock(tmp_path / "sources.lock.json", {"corpus": record})

    assert load_lock(path) == {"corpus": record}


def test_load_lock_without_file_is_empty(tmp_path: Path) -> None:
    assert load_lock(tmp_path / "absent.json") == {}


def test_load_lock_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "sources.lock.json"
    path.write_text('{"version": 99, "sources": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="version 99"):
        load_lock(path)


# -------------------------------------------------------------------- dispatch


def test_acquire_source_records_checksum_on_first_fetch(tmp_path: Path, server_url: str) -> None:
    source = {"type": "http", "id": "corpus", "url": f"{server_url}/corpus.txt"}

    _path, record = acquire_source(source, tmp_path)

    assert record.sha256 == PAYLOAD_SHA256
    assert record.size_bytes == len(PAYLOAD)
    assert record.url == f"{server_url}/corpus.txt"
    assert record.acquired_at


def test_acquire_source_reuses_a_locked_file_without_refetching(
    tmp_path: Path, server_url: str
) -> None:
    source = {"type": "http", "id": "corpus", "url": f"{server_url}/corpus.txt"}
    _path, record = acquire_source(source, tmp_path)

    # Point the source at a dead URL: a second call must not need the network.
    offline = {
        "type": "http",
        "id": "corpus",
        "url": f"{server_url}/missing",
        "filename": "corpus.txt",
    }
    path, reused = acquire_source(offline, tmp_path, locked=record)

    assert reused == record
    assert path.read_bytes() == PAYLOAD


def test_acquire_source_detects_a_source_that_changed_under_the_lock(
    tmp_path: Path, server_url: str
) -> None:
    """A corpus that silently changes would invalidate every run before it."""
    source = {"type": "http", "id": "corpus", "url": f"{server_url}/corpus.txt"}
    _path, record = acquire_source(source, tmp_path)
    stale = SourceRecord(
        id=record.id,
        type=record.type,
        filename=record.filename,
        size_bytes=record.size_bytes,
        sha256="1" * 64,
        url=record.url,
    )

    with pytest.raises(ValueError, match="changed since it was locked"):
        acquire_source(source, tmp_path, locked=stale, verify=True)


def test_acquire_source_handles_fixtures(tmp_path: Path) -> None:
    source = {"type": "fixture", "path": str(FIXTURE), "id": "tiny_corpus_v0"}

    path, record = acquire_source(source, tmp_path)

    assert record.type == "fixture"
    assert record.sha256 == sha256_file(path)
    assert record.url is None


def test_acquire_source_rejects_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported source type"):
        acquire_source({"type": "carrier-pigeon", "id": "x"}, tmp_path)


def test_acquire_source_requires_an_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required key: id"):
        acquire_source({"type": "fixture", "path": str(FIXTURE)}, tmp_path)


# ------------------------------------------------------------- committed lock


def test_committed_lockfile_matches_the_committed_fixture() -> None:
    """The lockfile is the recipe's record; drift between them is a real bug."""
    lock = load_lock(REPO_ROOT / "configs" / "sources.lock.json")
    entry = lock["tiny_corpus_v0"]

    assert entry.sha256 == sha256_file(FIXTURE)
    assert entry.size_bytes == FIXTURE.stat().st_size
