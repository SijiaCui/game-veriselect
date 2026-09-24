#!/bin/bash
# One-time setup for the GTBench eval track.
#   1. Install GTBench's runtime deps (curated subset; NOT its pinned
#      openai==1.3.5 / langchain — the patched chat backend removes that need,
#      and they would clash with the installed openai 2.x / vllm stack).
#   2. Drop in the patched chat backend + Qwen3 model configs.
#
# GTBench itself is vendored as plain source under bench/GTBench, so there is
# nothing to clone here. Re-run after pulling a new GTBench.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

if [ ! -e bench/GTBench/gamingbench/main.py ]; then
  echo "!! bench/GTBench is missing — restore the vendored tree before running this" >&2
  exit 1
fi

# --- 1. deps ---
echo ">> installing GTBench runtime deps"
python3 -m pip install "open_spiel==1.4" ml_collections jsonlines absl-py "gymnasium>=0.29" pyyaml retrying python-box

# --- 2. patch + configs ---
echo ">> applying patch + model configs"
cp "$HERE/gtbench_patch/chat.py" bench/GTBench/gamingbench/chat/chat.py
cp "$HERE/gtbench_model_configs/"*.yaml bench/GTBench/gamingbench/configs/model_configs/

echo ">> verifying imports"
python3 -c "import pyspiel; print('pyspiel', pyspiel.__version__ if hasattr(pyspiel,'__version__') else 'ok')"
( cd bench/GTBench && python3 -c "import gamingbench.chat.chat as c; import inspect; assert 'OpenAI' in inspect.getsource(c); print('gamingbench chat backend patched OK')" )
echo "SETUP COMPLETE"
