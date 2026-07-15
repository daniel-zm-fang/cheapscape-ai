"""Phase 1 entry point: acquire one reviewed source into data/raw/."""

import argparse
from pathlib import Path

from config import load_config
from datasets.acquire import acquire_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire raw text into data/raw/.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, required_keys=["sources", "raw_dir"])
    raw_dir = Path(cfg["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = REPO_ROOT / raw_dir

    sources = cfg["sources"]
    if not sources:
        raise ValueError("Config sources list is empty")

    for source in sources:
        source_type = source.get("type")
        if source_type == "fixture":
            destination = acquire_fixture(source, raw_dir)
            print(f"Wrote {destination} from source id={source.get('id')!r}")
        else:
            raise ValueError(f"Unsupported source type: {source_type!r}")


if __name__ == "__main__":
    main()
