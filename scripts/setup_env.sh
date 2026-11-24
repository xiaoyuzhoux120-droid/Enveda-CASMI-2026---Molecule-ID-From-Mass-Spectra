#!/usr/bin/env bash
# Reproducible setup for the CASMI 2026 research environment.
# Idempotent: re-running reuses .venv and only installs what is missing.
#   bash scripts/setup_env.sh          # CPU (cloud sessions, laptops)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_env.sh   # CUDA GPU
# No credentials are read or written here. Configure Kaggle auth separately
# (KAGGLE_API_TOKEN env var or ~/.kaggle/access_token); see ENVIRONMENT_SETUP.md.
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
export UV_SYSTEM_CERTS=1; unset UV_NATIVE_TLS || true
if ! command -v uv >/dev/null; then python3 -m pip install --user uv; export PATH="$HOME/.local/bin:$PATH"; fi
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
PY=.venv/bin/python
if [ "${USE_LOCK:-1}" = "1" ] && [[ "$TORCH_INDEX" == */cpu ]]; then
  uv pip install --python "$PY" -r requirements-lock.txt --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
else
  uv pip install --python "$PY" -r requirements-research.txt
  uv pip install --python "$PY" torch --index-url "$TORCH_INDEX"
fi
"$PY" -c "import rdkit, matchms, torch; print('rdkit', rdkit.__version__, '| matchms', matchms.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "Done. Verify with: $PY scripts/verify_env.py"
