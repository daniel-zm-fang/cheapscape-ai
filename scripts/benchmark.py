"""Measure throughput and memory before committing GPU budget.

Runs synthetic training steps at each candidate context length and projects what
a token budget would cost on the current machine. Use it to settle the
context-length decision and to sanity-check an instance before a long run.
"""

import argparse
import os
import sys
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _script_dir]
_src = os.path.join(os.path.dirname(_script_dir), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import load_config
from model import GPTConfig
from training.benchmark import benchmark_context, project_cost
from training.runtime import resolve_device, supports_precision

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=500_000_000,
        help="Tokens the real run should train on, used for the cost projection",
    )
    parser.add_argument(
        "--price-per-hour-usd",
        type=float,
        default=0.0,
        help="Hourly rate of the rented instance",
    )
    args = parser.parse_args()

    model_config = GPTConfig.from_mapping(load_config(args.config))
    device = resolve_device(args.device)
    if not supports_precision(device, args.precision):
        raise SystemExit(f"{device.type} cannot run precision {args.precision!r}")

    print(
        f"device={device} precision={args.precision} batch={args.batch_size} "
        f"accum={args.gradient_accumulation_steps} model_max_context={model_config.context_length}"
    )
    header = f"{'ctx':>6} {'tok/step':>10} {'s/step':>9} {'tok/s':>12} {'peak GiB':>9}"
    if args.price_per_hour_usd > 0.0:
        header += f" {'hours':>8} {'USD':>8}"
    print(header)

    for context_length in sorted(args.contexts):
        row = benchmark_context(
            model_config,
            context_length=context_length,
            batch_size=args.batch_size,
            device=device,
            precision=args.precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            steps=args.steps,
            warmup=args.warmup,
        )
        line = (
            f"{row.context_length:>6} {row.tokens_per_step:>10,} {row.seconds_per_step:>9.4f} "
            f"{row.tokens_per_second:>12,.0f} {row.peak_memory_gib:>9.2f}"
        )
        if args.price_per_hour_usd > 0.0:
            hours, usd = project_cost(
                row.tokens_per_second, args.token_budget, args.price_per_hour_usd
            )
            line += f" {hours:>8.1f} {usd:>8.2f}"
        print(line)

    if args.price_per_hour_usd > 0.0:
        print(f"\nProjection assumes a {args.token_budget:,}-token run at 100% utilization.")


if __name__ == "__main__":
    main()
