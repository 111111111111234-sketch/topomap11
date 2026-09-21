#!/usr/bin/env bash
# Prepare original HGR's native CPU dependencies, not a substitute algorithm.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo 'This setup targets Apple Silicon macOS.' >&2
    exit 1
fi
if [[ ! -x .venv-go2/bin/uv ]]; then
    echo 'Run scripts/setup_navigation_sim.sh first to prepare project-local uv/Python.' >&2
    exit 1
fi
export UV_CACHE_DIR="$repo_dir/.cache-sim"
export UV_PYTHON_INSTALL_DIR="$repo_dir/.python-sim"
.venv-go2/bin/uv python install 3.11 --no-bin
if [[ ! -x .venv-hgr-mac/bin/python ]]; then
    .venv-go2/bin/uv venv --python 3.11 .venv-hgr-mac
fi
.venv-go2/bin/uv pip install --python .venv-hgr-mac/bin/python -e '.[hgr-mac,sim,test]'
echo 'Native dependencies installed. This does not verify models, PyTorch3D, Habitat, or the complete navigation loop.'
