"""Phase 5: one-batch overfit training contract.

These tests pin down the training loop before full-corpus runs exist: config
validation, device resolution, shard loading, and the core gate that a tiny GPT
can drive loss on a fixed batch toward zero.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from config import load_config
from datasets.packed import write_shard
from model import GPT, GPTConfig
from training import TrainConfig, resolve_device, train
from training.loop import load_train_shard

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"


def _tiny_model_config(**overrides: object) -> GPTConfig:
    data: dict[str, object] = {
        "vocab_size": 64,
        "context_length": 32,
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 32,
        "dropout": 0.0,
        "position_encoding": "absolute",
    }
    data.update(overrides)
    return GPTConfig.from_mapping(data)


def _write_train_shard(
    tmp_path: Path,
    *,
    num_tokens: int,
    vocab_size: int = 64,
    context_length: int = 16,
) -> Path:
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    tokens = [i % vocab_size for i in range(num_tokens)]
    write_shard(packed_dir / "train_00000.bin", tokens, np.dtype(np.uint16))
    manifest = {
        "version": 1,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "shard_size": num_tokens,
        "splits": {
            "train": {"shards": ["train_00000.bin"], "num_tokens": num_tokens},
        },
    }
    (packed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Sanity: dataset must expose enough examples for the tests that follow.
    dataset = load_train_shard(packed_dir, context_length)
    assert len(dataset) >= 1
    return packed_dir


def _tiny_train_config(packed_dir: Path, **overrides: object) -> TrainConfig:
    data: dict[str, object] = {
        "seed": 0,
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": 60,
        "learning_rate": 1.0e-2,
        "context_length": 16,
        "packed_dir": packed_dir,
        "model_config": REPO_ROOT / "configs" / "model.yaml",
        "overfit_one_batch": True,
        "checkpoint_every": None,
        "validation_every": None,
    }
    data.update(overrides)
    return TrainConfig.from_mapping(data, repo_root=REPO_ROOT)


# ------------------------------------------------------------------- TrainConfig


def test_train_config_from_repo_yaml() -> None:
    cfg = TrainConfig.from_mapping(load_config(TRAIN_CONFIG), repo_root=REPO_ROOT)
    assert cfg.batch_size == 4
    assert cfg.max_steps == 100
    assert cfg.learning_rate == pytest.approx(3.0e-4)
    assert cfg.context_length == 512
    assert cfg.overfit_one_batch is True
    assert cfg.packed_dir == REPO_ROOT / "data" / "packed"
    assert cfg.model_config == REPO_ROOT / "configs" / "model.yaml"
    assert cfg.checkpoint_every is None
    assert cfg.validation_every is None


def test_repo_yaml_has_no_null_train_fields() -> None:
    raw = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    for key in (
        "batch_size",
        "max_steps",
        "learning_rate",
        "context_length",
        "packed_dir",
        "model_config",
    ):
        assert raw[key] is not None, f"{key} should be set for Phase 5"


def test_train_config_rejects_non_fp32() -> None:
    with pytest.raises(ValueError, match="fp32"):
        TrainConfig.from_mapping(
            {
                "seed": 1,
                "device": "cpu",
                "precision": "bf16",
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "learning_rate": 1e-3,
                "context_length": 8,
                "packed_dir": "data/packed",
                "model_config": "configs/model.yaml",
            },
            repo_root=REPO_ROOT,
        )


# --------------------------------------------------------------- device / seed


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_torch_device() -> None:
    device = resolve_device("auto")
    assert isinstance(device, torch.device)


# --------------------------------------------------------------- overfit gate


def test_one_batch_overfit_reduces_loss(tmp_path: Path) -> None:
    """The Phase 5 gate: a tiny model must drive fixed-batch loss down sharply."""
    packed_dir = _write_train_shard(tmp_path, num_tokens=128, context_length=16)
    train_cfg = _tiny_train_config(packed_dir)
    model_cfg = _tiny_model_config(context_length=32)

    result = train(train_cfg, model_cfg)

    assert result.steps == train_cfg.max_steps
    assert result.initial_loss > 1.0
    assert result.final_loss < 0.5
    assert result.final_loss < 0.25 * result.initial_loss
    assert result.device == "cpu"


def test_overfit_is_deterministic_with_seed(tmp_path: Path) -> None:
    packed_dir = _write_train_shard(tmp_path, num_tokens=128, context_length=16)
    train_cfg = _tiny_train_config(packed_dir, max_steps=5)
    model_cfg = _tiny_model_config(context_length=32)

    first = train(train_cfg, model_cfg)
    second = train(train_cfg, model_cfg)
    assert first.losses == second.losses


def test_training_context_cannot_exceed_model_context(tmp_path: Path) -> None:
    packed_dir = _write_train_shard(tmp_path, num_tokens=64, context_length=8)
    train_cfg = _tiny_train_config(packed_dir, context_length=16, max_steps=1)
    model_cfg = _tiny_model_config(context_length=8)

    with pytest.raises(ValueError, match="exceeds model"):
        train(train_cfg, model_cfg)


def test_checkpoint_and_validation_deferred(tmp_path: Path) -> None:
    packed_dir = _write_train_shard(tmp_path, num_tokens=64, context_length=8)
    train_cfg = _tiny_train_config(packed_dir, checkpoint_every=10, max_steps=1)
    model_cfg = _tiny_model_config(context_length=32)

    with pytest.raises(NotImplementedError, match="Checkpoints and validation"):
        train(train_cfg, model_cfg)


def test_full_corpus_mode_deferred(tmp_path: Path) -> None:
    packed_dir = _write_train_shard(tmp_path, num_tokens=64, context_length=8)
    train_cfg = _tiny_train_config(packed_dir, overfit_one_batch=False, max_steps=1)
    model_cfg = _tiny_model_config(context_length=32)

    with pytest.raises(NotImplementedError, match="Full-corpus training"):
        train(train_cfg, model_cfg)


def test_injected_model_is_trained(tmp_path: Path) -> None:
    packed_dir = _write_train_shard(tmp_path, num_tokens=128, context_length=16)
    train_cfg = _tiny_train_config(packed_dir, max_steps=20, learning_rate=1.0e-2)
    model_cfg = _tiny_model_config(context_length=32)
    model = GPT(model_cfg)

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    train(train_cfg, model_cfg, model=model)
    changed = any(
        not torch.equal(before[name], param.detach()) for name, param in model.named_parameters()
    )
    assert changed
