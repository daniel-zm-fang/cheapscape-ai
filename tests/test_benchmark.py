"""Benchmark contract: throughput measurement and cost projection."""

import pytest
import torch

from model import GPTConfig
from training.benchmark import benchmark_context, project_cost


def _tiny_model_config(**overrides: object) -> GPTConfig:
    data: dict[str, object] = {
        "vocab_size": 64,
        "context_length": 32,
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 16,
        "dropout": 0.0,
        "position_encoding": "absolute",
    }
    data.update(overrides)
    return GPTConfig.from_mapping(data)


# ------------------------------------------------------------------ projection


def test_project_cost_scales_with_price() -> None:
    hours, usd = project_cost(1_000.0, 3_600_000, 2.0)
    assert hours == pytest.approx(1.0)
    assert usd == pytest.approx(2.0)


def test_project_cost_is_free_without_a_price() -> None:
    _hours, usd = project_cost(500.0, 1_000_000, 0.0)
    assert usd == pytest.approx(0.0)


def test_project_cost_rejects_nonpositive_throughput() -> None:
    with pytest.raises(ValueError, match="tokens_per_second"):
        project_cost(0.0, 1_000, 1.0)


# ------------------------------------------------------------------ measurement


def test_benchmark_reports_positive_throughput() -> None:
    row = benchmark_context(
        _tiny_model_config(),
        context_length=16,
        batch_size=2,
        device=torch.device("cpu"),
        steps=2,
        warmup=1,
    )
    assert row.tokens_per_step == 2 * 16
    assert row.tokens_per_second > 0.0
    assert row.seconds_per_step > 0.0
    assert row.device == "cpu"
    # Peak memory is only tracked on CUDA.
    assert row.peak_memory_bytes == 0


def test_benchmark_counts_accumulated_micro_batches() -> None:
    row = benchmark_context(
        _tiny_model_config(),
        context_length=8,
        batch_size=2,
        gradient_accumulation_steps=3,
        device=torch.device("cpu"),
        steps=1,
        warmup=0,
    )
    assert row.tokens_per_step == 2 * 3 * 8


def test_benchmark_rejects_context_beyond_model_maximum() -> None:
    with pytest.raises(ValueError, match="exceeds model maximum"):
        benchmark_context(
            _tiny_model_config(context_length=16),
            context_length=32,
            batch_size=1,
            device=torch.device("cpu"),
            steps=1,
            warmup=0,
        )
