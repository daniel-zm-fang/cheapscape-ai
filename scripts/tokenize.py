"""Phase 3 entry point: encode the corpus into packed token shards.

Reads text from ``input_dir`` (``.txt`` files, or ``.parquet`` files with a
``text`` column when ``pyarrow`` is available), encodes each document with a
trained tokenizer, optionally wraps it in ``<|bos|>`` / ``<|eos|>`` markers, and
streams the resulting token ids into fixed-size shards under ``output_dir``. A
deterministic per-document split holds out ``validation_fraction`` of documents
as a ``val`` split. Finally a ``manifest.json`` records the shards, id dtype, and
vocabulary size for the training loop to consume.
"""

# isort: off
import _bootstrap  # noqa: F401 -- must run before any other import

# isort: on

import argparse
import random
from pathlib import Path

from config import load_config
from datasets.packed import ShardWriter, dtype_for_vocab, write_manifest
from datasets.text import iter_documents
from tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = [
    "input_dir",
    "tokenizer_dir",
    "output_dir",
    "shard_size",
    "validation_fraction",
]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _special_id(tokenizer: BPETokenizer, token: str) -> int:
    if token not in tokenizer.special_tokens:
        raise ValueError(
            f"Special token {token!r} is not in the tokenizer "
            f"(known: {sorted(tokenizer.special_tokens)})"
        )
    return tokenizer.special_tokens[token]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a tokenized corpus into shards.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "pack.yaml",
        help="Path to pack config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, required_keys=REQUIRED_KEYS)

    input_dir = _resolve(cfg["input_dir"])
    tokenizer_dir = _resolve(cfg["tokenizer_dir"])
    output_dir = _resolve(cfg["output_dir"])
    shard_size = int(cfg["shard_size"])
    validation_fraction = float(cfg["validation_fraction"])
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction}")

    tokenizer = BPETokenizer.load(tokenizer_dir)
    dtype = dtype_for_vocab(tokenizer.vocab_size)

    prepend_bos = bool(cfg.get("prepend_bos", False))
    append_eos = bool(cfg.get("append_eos", True))
    bos_id = _special_id(tokenizer, str(cfg.get("bos_token", "<|bos|>"))) if prepend_bos else None
    eos_id = _special_id(tokenizer, str(cfg.get("eos_token", "<|eos|>"))) if append_eos else None

    rng = random.Random(int(cfg.get("seed", 1337)))
    writers = {
        "train": ShardWriter(output_dir, "train", dtype, shard_size),
        "val": ShardWriter(output_dir, "val", dtype, shard_size),
    }

    doc_count = 0
    for text in iter_documents(input_dir):
        ids: list[int] = []
        if bos_id is not None:
            ids.append(bos_id)
        ids.extend(tokenizer.encode(text))
        if eos_id is not None:
            ids.append(eos_id)

        split = "val" if rng.random() < validation_fraction else "train"
        writers[split].append(ids)
        doc_count += 1

    shards = writers["train"].close() + writers["val"].close()
    manifest_path = write_manifest(
        output_dir,
        dtype=dtype,
        vocab_size=tokenizer.vocab_size,
        shard_size=shard_size,
        shards=shards,
    )

    totals = {name: sum(s.num_tokens for s in shards if s.split == name) for name in writers}
    print(
        f"Packed {doc_count} documents into {len(shards)} shards "
        f"(train={totals['train']} tokens, val={totals['val']} tokens), dtype={dtype.name}"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
