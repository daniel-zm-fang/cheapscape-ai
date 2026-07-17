"""Phase 3 entry point: create packed token shards and a manifest.

Streams the corpus through a trained tokenizer, wraps each document with
``<|bos|>``/``<|eos|>`` markers, and writes fixed-size ``.npy`` shards plus a
manifest that records the tokenizer and packing settings.
"""

import sys
from pathlib import Path

# This file is named tokenize.py, so running it directly puts scripts/ on
# sys.path[0] and would shadow the standard library's `tokenize` module (imported
# indirectly by numpy). Drop that entry so stdlib imports resolve correctly.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import argparse  # noqa: E402
from collections.abc import Iterable, Iterator  # noqa: E402

from config import load_config  # noqa: E402
from datasets.corpus import iter_documents  # noqa: E402
from datasets.packed import write_shards  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# uint16 covers vocabularies up to 65536 ids; larger vocabularies need uint32.
_UINT16_LIMIT = 1 << 16


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def encode_documents(
    tokenizer: BPETokenizer,
    documents: Iterable[str],
    *,
    bos_id: int | None,
    eos_id: int | None,
) -> Iterator[list[int]]:
    """Encode each document, optionally framing it with BOS/EOS ids."""
    for document in documents:
        tokens = tokenizer.encode(document)
        if bos_id is not None:
            tokens = [bos_id, *tokens]
        if eos_id is not None:
            tokens.append(eos_id)
        yield tokens


def _special_id(tokenizer: BPETokenizer, token: str, *, enabled: bool) -> int | None:
    if not enabled:
        return None
    if token not in tokenizer.special_tokens:
        raise ValueError(f"Tokenizer has no special token {token!r}; cannot frame documents")
    return tokenizer.special_tokens[token]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack tokenized text into shards.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "tokenize.yaml",
        help="Path to tokenize config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        required_keys=["input_dir", "tokenizer_path", "output_dir", "shard_size"],
    )

    input_dir = _resolve(cfg["input_dir"])
    tokenizer_path = _resolve(cfg["tokenizer_path"])
    output_dir = _resolve(cfg["output_dir"])
    shard_size = int(cfg["shard_size"])
    add_bos = bool(cfg.get("add_bos", True))
    add_eos = bool(cfg.get("add_eos", True))
    bos_token = str(cfg.get("bos_token", "<|bos|>"))
    eos_token = str(cfg.get("eos_token", "<|eos|>"))

    tokenizer = BPETokenizer.load(tokenizer_path)
    dtype = "uint16" if tokenizer.vocab_size <= _UINT16_LIMIT else "uint32"

    bos_id = _special_id(tokenizer, bos_token, enabled=add_bos)
    eos_id = _special_id(tokenizer, eos_token, enabled=add_eos)

    token_lists = encode_documents(
        tokenizer, iter_documents(input_dir), bos_id=bos_id, eos_id=eos_id
    )
    manifest = write_shards(
        token_lists,
        output_dir,
        shard_size=shard_size,
        dtype=dtype,
        extra={
            "tokenizer_path": str(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "add_bos": add_bos,
            "add_eos": add_eos,
        },
    )

    print(
        f"Packed {manifest['total_tokens']} tokens into {len(manifest['shards'])} shard(s) "
        f"({dtype}) at {output_dir}"
    )


if __name__ == "__main__":
    main()
