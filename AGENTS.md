# AGENTS.md

## Cursor Cloud specific instructions

Cheapscape.ai is a learning-first, from-scratch language model project. It is a Python
library + CLI, not a web/GUI app. The environment is a Python virtualenv at `.venv`.
The startup update script keeps `.venv` and dependencies in sync; you normally just need
to activate it: `source .venv/bin/activate`.

### Environment notes
- Python 3.12 is used (system `python3`). `.python-version` pins `3.13`, but the project
  only requires `>=3.12` (see `pyproject.toml`), so 3.12 is a valid, supported dev setup.
- PyTorch is installed as the CPU-only build (from the PyTorch CPU wheel index) since the
  VM has no GPU. `torch>=2.3` is satisfied, so a later `pip install` will not replace it.
- `types-PyYAML` is installed for mypy strict mode; it is not declared in `pyproject.toml`
  but is required for `mypy` to pass. The update script installs it.

### Lint / type-check / test (from repo root, venv activated)
- Lint: `ruff check .`
- Format check: `black --check .` (autoformat: `black .`)
- Type check: `mypy` (config in `pyproject.toml`, runs against `src`, strict)
- Tests: `pytest`

### Running the CLI scripts (important gotcha)
Scripts in `scripts/` import top-level modules from `src/` (e.g. `config`, `tokenizer.bpe`,
`datasets.acquire`). `src` is only on the path automatically for pytest (`pythonpath=["src"]`
in `pyproject.toml`). To run a script directly you MUST set `PYTHONPATH=src` and run from the
repo root, e.g.:
```
PYTHONPATH=src python scripts/download_data.py
PYTHONPATH=src python scripts/train_tokenizer.py --config configs/tokenizer.yaml
```
Why this matters: without `PYTHONPATH=src`, `from datasets.acquire ...` resolves to the
installed HuggingFace `datasets` package instead of `src/datasets/`, which in turn imports the
stdlib `tokenize` and collides with `scripts/tokenize.py`, causing a confusing crash. Setting
`PYTHONPATH=src` makes `src/datasets` and `src/tokenizer` win, avoiding the collision.

### Implemented vs. placeholder (phased project)
Only these are implemented: the config loader (`src/config.py`), fixture data acquisition
(`src/datasets/acquire.py` + `scripts/download_data.py`), and the byte-level BPE tokenizer
(`src/tokenizer/bpe.py` + `scripts/train_tokenizer.py`). The model, training loop, packing,
evaluation, preprocess/tokenize/train/posttrain scripts intentionally raise
`NotImplementedError` (later phases). Do not treat those raises as bugs.

### End-to-end sanity flow
`configs/tokenizer.yaml` reads `input_dir: data/processed`, which is produced by the
not-yet-implemented `preprocess` step. To exercise the tokenizer today, acquire the fixture
(`data/raw/`) and train against it by copying the config and pointing `input_dir` at `data/raw`
(the configs are meant to be copied per experiment). `data/` and `artifacts/` are gitignored
runtime outputs.
