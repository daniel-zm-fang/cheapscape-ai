"""Measure bytes-per-token on a corpus sample to size the real download.

Trains a tokenizer at each candidate vocabulary size on part of the sample and
measures compression on documents it never saw. The result converts a token
budget into "how much source text to download" and "how large the packed shards
will be", so both can be decided before paying for either.
"""

# isort: off
import _bootstrap  # noqa: F401 -- must run before any other import

# isort: on

import argparse
from pathlib import Path

from datasets.sizing import measure_vocab_size, split_sample, take_bytes
from datasets.text import iter_documents

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Directory of sample .txt/.parquet documents",
    )
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=6_000_000_000,
        help="Tokens the real training run needs",
    )
    parser.add_argument(
        "--max-sample-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="Cap on how much of the sample to read; training memory scales with this",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--special-tokens", nargs="*", default=["<|bos|>", "<|eos|>"])
    args = parser.parse_args()

    documents = list(take_bytes(iter_documents(args.input_dir), args.max_sample_bytes))
    train_texts, eval_texts = split_sample(documents, args.holdout_fraction)
    sample_bytes = sum(len(text.encode("utf-8")) for text in documents)

    print(
        f"sample: {len(documents):,} documents, {sample_bytes:,} bytes "
        f"({len(train_texts):,} train / {len(eval_texts):,} holdout)"
    )
    print(f"target: {args.target_tokens:,} tokens\n")

    header = (
        f"{'vocab':>7} {'bytes/token':>12} {'holdout tok':>12} {'train s':>8} "
        f"{'source GiB':>11} {'packed GiB':>11}"
    )
    print(header)
    for vocab_size in sorted(args.vocab_sizes):
        measurement = measure_vocab_size(
            train_texts,
            eval_texts,
            vocab_size=vocab_size,
            special_tokens=args.special_tokens,
        )
        source_gib = measurement.source_bytes_for(args.target_tokens) / GIB
        packed_gib = measurement.packed_bytes_for(args.target_tokens) / GIB
        print(
            f"{measurement.vocab_size:>7,} {measurement.bytes_per_token:>12.3f} "
            f"{measurement.eval_tokens:>12,} {measurement.train_seconds:>8.1f} "
            f"{source_gib:>11.2f} {packed_gib:>11.2f}"
        )

    print(
        "\nCompression is measured on held-out documents, so it reflects unseen text."
        "\nA bigger vocabulary packs fewer tokens per byte but widens the embedding table;"
        "\nprice both against the GPU benchmark before fixing the model config."
    )


if __name__ == "__main__":
    main()
