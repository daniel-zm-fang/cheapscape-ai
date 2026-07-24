"""Training-loop contract.

Covers the one-batch overfit gate plus the behaviour a rented-GPU run depends
on: config validation, LR schedule, checkpoint/resume, deterministic
validation, spend caps, and preemption.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from config import load_config
from datasets.packed import PackedSplit, write_shard
from model import GPT, GPTConfig
from training import (
    PreemptionSignal,
    TrainConfig,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    lr_at_step,
    resolve_device,
    save_checkpoint,
    train,
)
from training.checkpoint import prune_checkpoints

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
OVERFIT_CONFIG = REPO_ROOT / "configs" / "overfit.yaml"


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


def _write_packed(
    tmp_path: Path,
    *,
    train_tokens: int = 512,
    val_tokens: int = 256,
    vocab_size: int = 64,
) -> Path:
    """Write a two-shard train split plus a val split, with a manifest."""
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    dtype = np.dtype(np.uint16)
    half = train_tokens // 2
    write_shard(packed_dir / "train_00000.bin", [i % vocab_size for i in range(half)], dtype)
    write_shard(
        packed_dir / "train_00001.bin",
        [(i * 3) % vocab_size for i in range(train_tokens - half)],
        dtype,
    )
    write_shard(
        packed_dir / "val_00000.bin", [(i * 5) % vocab_size for i in range(val_tokens)], dtype
    )
    manifest = {
        "version": 1,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "shard_size": max(half, 1),
        "splits": {
            "train": {
                "shards": ["train_00000.bin", "train_00001.bin"],
                "num_tokens": train_tokens,
            },
            "val": {"shards": ["val_00000.bin"], "num_tokens": val_tokens},
        },
    }
    (packed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return packed_dir


def _train_config(packed_dir: Path, **overrides: object) -> TrainConfig:
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
        "lr_schedule": "constant",
        "log_every": 1000,
    }
    data.update(overrides)
    return TrainConfig.from_mapping(data, repo_root=REPO_ROOT)


# ------------------------------------------------------------------- TrainConfig


def test_pretrain_config_from_repo_yaml() -> None:
    cfg = TrainConfig.from_mapping(load_config(TRAIN_CONFIG), repo_root=REPO_ROOT)
    assert cfg.precision == "bf16"
    assert cfg.context_length == 1024
    assert cfg.overfit_one_batch is False
    assert cfg.lr_schedule == "cosine"
    assert cfg.resume is True
    assert cfg.checkpoint_every == 500
    assert cfg.checkpoint_dir == REPO_ROOT / "artifacts" / "checkpoints"
    assert cfg.tokens_per_step == 8 * 8 * 1024


def test_overfit_config_from_repo_yaml() -> None:
    cfg = TrainConfig.from_mapping(load_config(OVERFIT_CONFIG), repo_root=REPO_ROOT)
    assert cfg.overfit_one_batch is True
    assert cfg.precision == "fp32"
    assert cfg.context_length == 512
    assert cfg.checkpoint_dir is None


def test_repo_yaml_has_no_null_required_fields() -> None:
    for path in (TRAIN_CONFIG, OVERFIT_CONFIG):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("batch_size", "max_steps", "learning_rate", "context_length", "packed_dir"):
            assert raw[key] is not None, f"{key} should be set in {path.name}"


def test_config_rejects_unknown_precision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="precision"):
        _train_config(tmp_path, precision="int4")


def test_config_rejects_checkpoint_every_without_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        _train_config(tmp_path, checkpoint_every=5)


def test_config_rejects_budget_without_price(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="price_per_hour_usd"):
        _train_config(tmp_path, budget_usd=10.0)


def test_config_rejects_warmup_beyond_max_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        _train_config(tmp_path, warmup_steps=60, max_steps=60)


# --------------------------------------------------------------------- schedule


def test_warmup_ramps_then_cosine_decays() -> None:
    kwargs = {"base_lr": 1.0, "max_steps": 10, "warmup_steps": 2, "min_lr_ratio": 0.0}
    assert lr_at_step(0, **kwargs) == pytest.approx(0.5)
    assert lr_at_step(1, **kwargs) == pytest.approx(1.0)
    assert lr_at_step(2, **kwargs) == pytest.approx(1.0)
    assert lr_at_step(9, **kwargs) < 0.1
    # Never negative, and clamped at the floor past the horizon.
    assert lr_at_step(50, **kwargs) == pytest.approx(0.0)


def test_constant_schedule_holds_base_lr() -> None:
    assert lr_at_step(7, base_lr=3e-4, max_steps=10, schedule="constant") == pytest.approx(3e-4)


def test_min_lr_ratio_is_the_floor() -> None:
    value = lr_at_step(10, base_lr=1.0, max_steps=10, min_lr_ratio=0.25)
    assert value == pytest.approx(0.25)


# --------------------------------------------------------------- device / runtime


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_torch_device() -> None:
    assert isinstance(resolve_device("auto"), torch.device)


def test_preemption_signal_starts_untriggered() -> None:
    with PreemptionSignal() as preemption:
        assert preemption.triggered is False


# ---------------------------------------------------------------- packed split


def test_packed_split_spans_every_shard(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path, train_tokens=512)
    split = PackedSplit(packed_dir, "train", context_length=16)
    per_shard = (256 - 1) // 16
    assert len(split) == 2 * per_shard
    # An index in the second shard resolves without error and stays in vocabulary.
    x, y = split[per_shard]
    assert x.shape == (16,)
    assert torch.equal(x[1:], y[:-1])


def test_packed_split_skips_shards_too_short(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    write_shard(packed_dir / "train_00000.bin", list(range(40)), np.dtype(np.uint16))
    write_shard(packed_dir / "train_00001.bin", [1, 2], np.dtype(np.uint16))
    (packed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "dtype": "uint16",
                "vocab_size": 64,
                "shard_size": 40,
                "splits": {
                    "train": {
                        "shards": ["train_00000.bin", "train_00001.bin"],
                        "num_tokens": 42,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    split = PackedSplit(packed_dir, "train", context_length=16)
    assert len(split.shards) == 1
    assert len(split) == 2


# --------------------------------------------------------------- overfit gate


def test_one_batch_overfit_reduces_loss(tmp_path: Path) -> None:
    """The Phase 5 gate: a tiny model must drive fixed-batch loss down sharply."""
    packed_dir = _write_packed(tmp_path)
    result = train(_train_config(packed_dir), _tiny_model_config())

    assert result.steps == 60
    assert result.initial_loss > 1.0
    assert result.final_loss < 0.5
    assert result.final_loss < 0.25 * result.initial_loss
    assert result.device == "cpu"
    assert result.stop_reason == "max_steps"
    assert result.tokens_per_second > 0.0


def test_overfit_is_deterministic_with_seed(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, max_steps=5)
    model_cfg = _tiny_model_config()
    assert train(cfg, model_cfg).losses == train(cfg, model_cfg).losses


def test_full_corpus_sampling_is_deterministic(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, overfit_one_batch=False, max_steps=6)
    model_cfg = _tiny_model_config()
    assert train(cfg, model_cfg).losses == train(cfg, model_cfg).losses


def test_gradient_accumulation_runs_micro_batches(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, gradient_accumulation_steps=3, max_steps=4)
    result = train(cfg, _tiny_model_config())
    assert result.steps == 4
    assert cfg.tokens_per_step == 2 * 3 * 16


def test_training_context_cannot_exceed_model_context(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, context_length=64, max_steps=1)
    with pytest.raises(ValueError, match="exceeds model"):
        train(cfg, _tiny_model_config(context_length=32))


def test_injected_model_is_trained(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, max_steps=20)
    model_cfg = _tiny_model_config()
    model = GPT(model_cfg)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    train(cfg, model_cfg, model=model)
    assert any(not torch.equal(before[name], p.detach()) for name, p in model.named_parameters())


# ----------------------------------------------------------------- validation


def test_validation_runs_on_cadence(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, max_steps=6, validation_every=3)
    result = train(cfg, _tiny_model_config())
    assert [step for step, _loss in result.val_losses] == [3, 6]
    assert all(value > 0.0 for _step, value in result.val_losses)


def test_validation_windows_are_fixed(tmp_path: Path) -> None:
    """Validation loss must be comparable across runs, so windows never shuffle."""
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(packed_dir, max_steps=3, validation_every=3)
    model_cfg = _tiny_model_config()
    assert train(cfg, model_cfg).val_losses == train(cfg, model_cfg).val_losses


# ----------------------------------------------------------------- checkpoints


def test_checkpoint_roundtrip_restores_weights(tmp_path: Path) -> None:
    model_cfg = _tiny_model_config()
    model = GPT(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path, step=7, model=model, optimizer=optimizer, model_config=model_cfg
    )

    restored = GPT(model_cfg)
    payload = load_checkpoint(path, model=restored)
    assert payload["step"] == 7
    for (_name, a), (_other, b) in zip(
        model.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    save_checkpoint(tmp_path, step=1, model=model)
    assert [p.name for p in tmp_path.iterdir()] == ["step_0000001.pt"]


def test_prune_keeps_newest_checkpoints(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    for step in (1, 2, 3):
        save_checkpoint(tmp_path, step=step, model=model)
    removed = prune_checkpoints(tmp_path, keep=2)
    assert [p.name for p in removed] == ["step_0000001.pt"]
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "step_0000002.pt",
        "step_0000003.pt",
    ]


def test_training_writes_and_prunes_checkpoints(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    cfg = _train_config(
        packed_dir,
        max_steps=6,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=2,
        keep_last_checkpoints=2,
    )
    result = train(cfg, _tiny_model_config())
    assert [p.name for p in list_checkpoints(checkpoint_dir)] == [
        "step_0000004.pt",
        "step_0000006.pt",
    ]
    assert latest_checkpoint(checkpoint_dir) is not None
    assert result.checkpoints[-1].name == "step_0000006.pt"


def test_resume_continues_from_checkpoint_step(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    model_cfg = _tiny_model_config()
    first = _train_config(
        packed_dir, max_steps=4, checkpoint_dir=checkpoint_dir, checkpoint_every=4
    )
    train(first, model_cfg)

    second = _train_config(
        packed_dir, max_steps=6, checkpoint_dir=checkpoint_dir, checkpoint_every=6
    )
    resumed = train(second, model_cfg)
    assert resumed.start_step == 4
    assert resumed.steps == 2


def test_resume_disabled_starts_from_zero(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    model_cfg = _tiny_model_config()
    cfg = _train_config(packed_dir, max_steps=4, checkpoint_dir=checkpoint_dir, checkpoint_every=4)
    train(cfg, model_cfg)

    fresh = _train_config(
        packed_dir,
        max_steps=4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=4,
        resume=False,
    )
    assert train(fresh, model_cfg).start_step == 0


# --------------------------------------------------------- preemption / budget


def test_preemption_saves_a_checkpoint_and_stops(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    cfg = _train_config(
        packed_dir, max_steps=50, checkpoint_dir=checkpoint_dir, checkpoint_every=1000
    )
    preemption = PreemptionSignal()
    preemption.triggered = True
    preemption.signal_name = "SIGTERM"

    result = train(cfg, _tiny_model_config(), preemption=preemption)
    assert result.stop_reason == "preempted:SIGTERM"
    assert result.steps == 1
    assert latest_checkpoint(checkpoint_dir) is not None


def test_budget_cap_stops_the_run(tmp_path: Path) -> None:
    packed_dir = _write_packed(tmp_path)
    cfg = _train_config(
        packed_dir,
        max_steps=1000,
        price_per_hour_usd=1_000_000.0,  # makes any elapsed time exceed the cap
        budget_usd=0.01,
    )
    result = train(cfg, _tiny_model_config())
    assert result.stop_reason == "budget_exhausted"
    assert result.steps < 1000
    assert result.estimated_cost_usd > 0.0
