#!/usr/bin/env bash
# Install only the local navigation simulation environment; never launch a robot.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"
if [[ ! -x .venv-go2/bin/python ]]; then
    python3 -m venv .venv-go2
fi
.venv-go2/bin/python -m pip install uv
export UV_CACHE_DIR="$repo_dir/.cache-sim"
export UV_PYTHON_INSTALL_DIR="$repo_dir/.python-sim"
.venv-go2/bin/uv python install 3.11 --no-bin
if [[ ! -x .venv-sim/bin/python ]]; then
    .venv-go2/bin/uv venv --python 3.11 .venv-sim
fi
.venv-go2/bin/uv pip install --python .venv-sim/bin/python -e '.[sim,test]'
.venv-sim/bin/topomap-sim --help
