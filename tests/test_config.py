"""Phase 0: config loading contract."""

from pathlib import Path

import pytest
import yaml

from config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"


def test_load_dataset_config() -> None:
    cfg = load_config(DATASET_CONFIG, required_keys=["seed", "sources", "output_dir"])

    assert cfg["seed"] == 1337
    assert cfg["output_dir"] == "data/processed"
    assert isinstance(cfg["sources"], list)


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required config keys"):
        load_config(config_path, required_keys=["seed", "output_dir"])
