"""Acquire raw text from configured sources into ``data/raw/``.

Corpora never enter version control; the recipe does. ``configs/dataset.yaml``
declares where each source comes from and the lockfile records exactly what came
back, so a corpus can be rebuilt byte-for-byte on a machine that will be
destroyed after the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_VERSION = 1
CHUNK_SIZE = 1 << 20
ALLOWED_SCHEMES = ("http", "https")


@dataclass(frozen=True)
class SourceRecord:
    """What one fetch actually produced, as pinned in the lockfile."""

    id: str
    type: str
    filename: str
    size_bytes: int
    sha256: str
    url: str | None = None
    acquired_at: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SourceRecord:
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            filename=str(data["filename"]),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
            url=data.get("url"),
            acquired_at=str(data.get("acquired_at", "")),
        )


def sha256_file(path: Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Hash a file without holding it in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------- lockfile


def load_lock(path: Path) -> dict[str, SourceRecord]:
    """Read pinned source records, or return empty when no lockfile exists."""
    path = Path(path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != LOCK_VERSION:
        raise ValueError(f"Lockfile {path} has version {version}, expected {LOCK_VERSION}")
    return {
        source_id: SourceRecord.from_mapping(entry)
        for source_id, entry in payload.get("sources", {}).items()
    }


def save_lock(path: Path, records: dict[str, SourceRecord]) -> Path:
    """Write pinned source records, sorted so diffs stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": LOCK_VERSION,
        "sources": {source_id: asdict(records[source_id]) for source_id in sorted(records)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# -------------------------------------------------------------------- fetching


def _safe_filename(source: dict[str, Any]) -> str:
    """Pick a destination filename that cannot escape the output directory."""
    declared = source.get("filename")
    if declared:
        name = str(declared)
    else:
        url = str(source.get("url", ""))
        name = Path(urllib.parse.urlparse(url).path).name or f"{source['id']}.bin"
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe destination filename: {name!r}")
    return name


def _download(url: str, partial: Path, *, timeout: float, chunk_size: int) -> None:
    """Stream ``url`` into ``partial``, resuming an earlier attempt when possible.

    Rented instances lose connections and get replaced, so a multi-gigabyte
    download that cannot resume is a download that may never finish.
    """
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        # 416 means the byte range is past the end: the partial file is stale.
        if error.code == 416 and offset:
            partial.unlink()
            _download(url, partial, timeout=timeout, chunk_size=chunk_size)
            return
        raise

    with response:
        # A server that ignores Range replies 200 with the whole body, so the
        # partial file has to be discarded rather than appended to.
        resuming = offset > 0 and response.status == 206
        with open(partial, "ab" if resuming else "wb") as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)


def acquire_http(
    source: dict[str, Any],
    output_dir: Path,
    *,
    timeout: float = 60.0,
    chunk_size: int = CHUNK_SIZE,
) -> Path:
    """Download one HTTP(S) source into ``output_dir`` and return its path."""
    if source.get("type") != "http":
        raise ValueError(f"Expected type 'http', got {source.get('type')!r}")
    for key in ("id", "url"):
        if not source.get(key):
            raise ValueError(f"HTTP source missing required key: {key}")

    url = str(source["url"])
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme {scheme!r}; allowed: {ALLOWED_SCHEMES}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / _safe_filename(source)
    partial = destination.with_name(destination.name + ".part")

    _download(url, partial, timeout=timeout, chunk_size=chunk_size)

    expected = source.get("sha256")
    if expected:
        actual = sha256_file(partial)
        if actual != expected:
            partial.unlink()
            raise ValueError(
                f"Checksum mismatch for {source['id']}: expected {expected}, got {actual}"
            )

    os.replace(partial, destination)
    return destination


def acquire_fixture(source: dict[str, Any], output_dir: Path) -> Path:
    """Copy a local fixture file into output_dir and return the destination path."""
    if source.get("type") != "fixture":
        raise ValueError(f"Expected type 'fixture', got {source.get('type')!r}")

    try:
        relative_path = Path(source["path"])
        source_id = str(source["id"])
    except KeyError as exc:
        raise ValueError(f"Fixture source missing required key: {exc.args[0]}") from exc

    fixture_path = relative_path if relative_path.is_absolute() else REPO_ROOT / relative_path
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{source_id}.txt"
    shutil.copy2(fixture_path, destination)
    return destination


# ------------------------------------------------------------------ dispatch


def acquire_source(
    source: dict[str, Any],
    output_dir: Path,
    *,
    locked: SourceRecord | None = None,
    verify: bool = False,
    timeout: float = 60.0,
) -> tuple[Path, SourceRecord]:
    """Fetch one source if needed and return its path plus a lockfile record.

    An already-present file is trusted on size alone unless ``verify`` is set,
    because re-hashing tens of gigabytes on every run is its own kind of waste.
    """
    source_type = source.get("type")
    source_id = str(source.get("id", ""))
    if not source_id:
        raise ValueError("Source missing required key: id")

    output_dir = Path(output_dir)
    if source_type == "fixture":
        filename = f"{source_id}.txt"
    elif source_type == "http":
        filename = _safe_filename(source)
    else:
        raise ValueError(f"Unsupported source type: {source_type!r}")

    destination = output_dir / filename
    reusable = (
        locked is not None
        and destination.is_file()
        and destination.stat().st_size == locked.size_bytes
    )
    if reusable and locked is not None and not verify:
        return destination, locked

    if not reusable:
        if source_type == "fixture":
            destination = acquire_fixture(source, output_dir)
        else:
            destination = acquire_http(source, output_dir, timeout=timeout)

    digest = sha256_file(destination)
    declared = source.get("sha256")
    if declared and digest != declared:
        raise ValueError(f"Checksum mismatch for {source_id}: expected {declared}, got {digest}")
    if locked is not None and digest != locked.sha256:
        raise ValueError(
            f"Source {source_id} changed since it was locked: expected {locked.sha256}, "
            f"got {digest}. Delete the lock entry only if the change is intended."
        )

    record = SourceRecord(
        id=source_id,
        type=str(source_type),
        filename=destination.name,
        size_bytes=destination.stat().st_size,
        sha256=digest,
        url=source.get("url"),
        acquired_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return destination, record
