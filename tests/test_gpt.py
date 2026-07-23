"""Phase 4: decoder-only GPT contract.

These tests pin down the model before training exists: config validation,
forward shapes, causal masking, context-length enforcement, weight tying, and
a smoke check that logits can drive next-token cross-entropy.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from config import load_config
from model import GPT, GPTConfig
from model.attention import CausalSelfAttention

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPO_ROOT / "configs" / "model.yaml"


def _tiny_config(**overrides: object) -> GPTConfig:
    data: dict[str, object] = {
        "vocab_size": 64,
        "context_length": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 32,
        "dropout": 0.0,
        "position_encoding": "absolute",
    }
    data.update(overrides)
    return GPTConfig.from_mapping(data)


# ------------------------------------------------------------------- GPTConfig


def test_config_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _tiny_config(d_model=30, n_heads=4)


def test_config_rejects_rope_for_now() -> None:
    with pytest.raises(ValueError, match="absolute"):
        _tiny_config(position_encoding="rope")


def test_config_from_repo_yaml() -> None:
    cfg = GPTConfig.from_mapping(load_config(MODEL_CONFIG))
    assert cfg.vocab_size == 4096
    assert cfg.context_length == 1024
    assert cfg.n_layers == 6
    assert cfg.n_heads == 6
    assert cfg.d_model == 384
    assert cfg.position_encoding == "absolute"
    assert cfg.head_dim == 64


def test_repo_yaml_has_no_null_model_fields() -> None:
    raw = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    for key in ("vocab_size", "context_length", "n_layers", "n_heads", "d_model"):
        assert raw[key] is not None, f"{key} should be set for Phase 4"


# -------------------------------------------------------------- forward shapes


def test_forward_shape() -> None:
    model = GPT(_tiny_config())
    token_ids = torch.randint(0, 64, (3, 11))
    logits = model(token_ids)
    assert logits.shape == (3, 11, 64)
    assert torch.isfinite(logits).all()


def test_forward_rejects_rank_other_than_two() -> None:
    model = GPT(_tiny_config())
    with pytest.raises(ValueError, match=r"\[batch, time\]"):
        model(torch.randint(0, 64, (11,)))


def test_forward_rejects_sequence_longer_than_context() -> None:
    model = GPT(_tiny_config(context_length=8))
    with pytest.raises(ValueError, match="exceeds context_length"):
        model(torch.randint(0, 64, (1, 9)))


def test_from_mapping_matches_explicit_config() -> None:
    mapping = {
        "vocab_size": 64,
        "context_length": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 32,
        "dropout": 0.0,
        "position_encoding": "absolute",
    }
    model = GPT.from_mapping(mapping)
    assert model.config == GPTConfig.from_mapping(mapping)
    assert model(torch.randint(0, 64, (2, 5))).shape == (2, 5, 64)


# ------------------------------------------------------------ causal attention


def test_attention_is_causal() -> None:
    """``forward`` must put zero weight on every future position j > i."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=4, context_length=8, dropout=0.0)
    x = torch.randn(1, 6, 32)
    captured: dict[str, torch.Tensor] = {}

    def capture_weights(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        # Dropout's input is the post-softmax attention matrix from ``forward``.
        captured["weights"] = inputs[0].detach().clone()

    handle = attn.attn_dropout.register_forward_hook(capture_weights)
    try:
        output = attn(x)
    finally:
        handle.remove()

    assert output.shape == x.shape
    weights = captured["weights"]
    time = x.shape[1]
    future = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[..., future]) == 0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 4, time), atol=1e-5)


def test_block_preserves_sequence_shape() -> None:
    from model.block import TransformerBlock

    block = TransformerBlock(d_model=32, n_heads=4, context_length=16, dropout=0.0)
    x = torch.randn(2, 7, 32)
    assert block(x).shape == x.shape


def test_absolute_positions_affect_logits() -> None:
    """Zeroing the position table must change outputs — positions are not decorative."""
    torch.manual_seed(0)
    model = GPT(_tiny_config())
    model.eval()
    token_ids = torch.randint(0, 64, (1, 8))

    with torch.no_grad():
        with_positions = model(token_ids)
        model.position_emb.weight.zero_()
        without_positions = model(token_ids)

    assert not torch.allclose(with_positions, without_positions)


# --------------------------------------------------------------- weight tying


def test_lm_head_shares_token_embedding_weights() -> None:
    model = GPT(_tiny_config())
    assert model.lm_head.weight is model.token_emb.weight


def test_parameter_count_matches_tied_architecture() -> None:
    cfg = _tiny_config()
    model = GPT(cfg)
    d_model = cfg.d_model
    # token emb + position emb + L * (2 LayerNorms + qkv + attn proj + mlp) + final LN.
    # lm_head is tied to token_emb, so it contributes no extra parameters.
    per_block = (
        2 * d_model  # ln_1 weight + bias
        + 3 * d_model * d_model  # qkv
        + d_model * d_model  # attn proj
        + 2 * d_model  # ln_2 weight + bias
        + 4 * d_model * d_model  # mlp fc
        + 4 * d_model * d_model  # mlp proj
    )
    expected = (
        cfg.vocab_size * d_model
        + cfg.context_length * d_model
        + cfg.n_layers * per_block
        + 2 * d_model  # ln_f
    )
    assert model.parameter_count() == expected


# --------------------------------------------------------- next-token smoke


def test_logits_support_next_token_cross_entropy() -> None:
    """A random batch must produce a finite mean loss — Phase 5's entry point."""
    torch.manual_seed(0)
    model = GPT(_tiny_config())
    # Packed loader yields x and y of equal length, y shifted by one token.
    x = torch.randint(0, 64, (4, 12))
    y = torch.randint(0, 64, (4, 12))
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, 64), y.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_dropout_inactive_in_eval() -> None:
    model = GPT(_tiny_config(dropout=0.5))
    token_ids = torch.randint(0, 64, (2, 8))
    model.eval()
    with torch.no_grad():
        a = model(token_ids)
        b = model(token_ids)
    assert torch.equal(a, b)


def test_forward_rejects_out_of_vocabulary_ids() -> None:
    model = GPT(_tiny_config(vocab_size=64))
    with pytest.raises(ValueError, match=r"\[0, 64\)"):
        model(torch.tensor([[0, 64]], dtype=torch.long))


def test_forward_rejects_non_integer_token_ids() -> None:
    model = GPT(_tiny_config())
    with pytest.raises(ValueError, match="integer dtype"):
        model(torch.zeros(1, 4))
