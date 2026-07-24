"""Phase 5 entry point: launch one reproducible training run."""

import argparse
import os
import sys
from pathlib import Path

# Running as a script puts ``scripts/`` on ``sys.path[0]``. Drop it so local
# module names cannot shadow third-party or stdlib packages.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _script_dir]

# Ensure the src/ layout is importable when the package is not installed.
_src = os.path.join(os.path.dirname(_script_dir), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import load_config
from training.loop import train_from_configs

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = [
    "seed",
    "device",
    "precision",
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "learning_rate",
    "context_length",
    "packed_dir",
    "model_config",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Cheapscape training experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train.yaml",
        help="Path to train config YAML",
    )
    args = parser.parse_args()

    train_cfg = load_config(args.config, required_keys=REQUIRED_KEYS)
    result = train_from_configs(train_cfg, repo_root=REPO_ROOT)

    if result.stop_reason == "already_complete":
        print(f"Nothing to do: checkpoint is already at step {result.start_step}.")
        return

    print(
        f"Ran steps {result.start_step}->{result.start_step + result.steps} on {result.device} "
        f"({result.stop_reason}): loss {result.initial_loss:.4f} -> {result.final_loss:.4f}"
    )
    if result.val_losses:
        step, value = result.val_losses[-1]
        print(f"Validation loss at step {step}: {value:.4f}")
    print(
        f"{result.tokens_per_second:,.0f} tok/s over {result.elapsed_seconds:.1f}s, "
        f"estimated ${result.estimated_cost_usd:.2f}"
    )
    if result.peak_memory_bytes:
        print(f"Peak GPU memory: {result.peak_memory_bytes / 2**30:.2f} GiB")
    if result.checkpoints:
        print(f"Newest checkpoint: {result.checkpoints[-1]}")

    # A preempted run should exit non-zero so a supervising loop relaunches it.
    if result.stop_reason.startswith("preempted"):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
