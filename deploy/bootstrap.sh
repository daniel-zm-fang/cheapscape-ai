#!/usr/bin/env bash
# Provision a fresh Ubuntu GPU instance to run Cheapscape.
#
# Idempotent: safe to re-run on the same box.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"

log() { printf '\n=== %s\n' "$1"; }

log "System packages"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip git curl
else
  echo "apt-get not found; install python3-venv, pip, git, and curl yourself."
fi

log "Virtual environment at $VENV"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade -q pip

log "Project dependencies"
# Rented images usually ship a CUDA-matched torch; keep it rather than
# downloading several GiB again.
if python -c 'import torch' 2>/dev/null; then
  echo "Reusing preinstalled torch $(python -c 'import torch; print(torch.__version__)')"
  python -m pip install -q --no-deps -e "$REPO_ROOT"
  python -m pip install -q numpy pyyaml
else
  python -m pip install -q -e "$REPO_ROOT"
fi
python -m pip install -q -e "$REPO_ROOT[data]"

log "Environment check"
python - <<'PY'
import torch

print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"cuda  {torch.version.cuda} on {name} ({total:.0f} GiB)")
    print(f"bf16  {'supported' if torch.cuda.is_bf16_supported() else 'unsupported, use fp16'}")
else:
    print("cuda  unavailable - this box cannot train; check the driver or image")
PY

log "Next steps"
cat <<'EOF'
  source .venv/bin/activate
  python3 scripts/benchmark.py --contexts 512 1024 --batch-size 8 --precision bf16
  ./deploy/run_pretrain.sh configs/train.yaml
EOF
