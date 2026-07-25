"""Phase 1 entry point: acquire reviewed sources into data/raw/.

Writes a lockfile pinning the exact bytes each source returned. The corpus stays
out of version control; the lockfile is the record that lets a fresh machine
rebuild the same corpus.
"""

# isort: off
import _bootstrap  # noqa: F401 -- must run before any other import

# isort: on

import argparse
from pathlib import Path

from config import load_config
from datasets.acquire import acquire_source, load_lock, save_lock

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKFILE = REPO_ROOT / "configs" / "sources.lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire raw text into data/raw/.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=DEFAULT_LOCKFILE,
        help="Where to read and write pinned source checksums",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash files that are already present instead of trusting their size",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, required_keys=["sources", "raw_dir"])
    raw_dir = Path(cfg["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = REPO_ROOT / raw_dir

    sources = cfg["sources"]
    if not sources:
        raise ValueError("Config sources list is empty")

    lock = load_lock(args.lockfile)
    total_bytes = 0
    for source in sources:
        source_id = str(source.get("id", ""))
        destination, record = acquire_source(
            source, raw_dir, locked=lock.get(source_id), verify=args.verify
        )
        lock[record.id] = record
        total_bytes += record.size_bytes
        print(
            f"{record.id}: {destination} ({record.size_bytes:,} bytes, sha256 {record.sha256[:12]})"
        )

    save_lock(args.lockfile, lock)
    print(f"\n{len(sources)} source(s), {total_bytes:,} bytes total")
    print(f"Pinned in {args.lockfile}")


if __name__ == "__main__":
    main()
