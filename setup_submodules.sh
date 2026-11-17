#!/usr/bin/env bash
set -euo pipefail

# godot-cpp is tracked in .gitmodules and pinned to the godot-4.4.1-stable tag.
# This just fetches it; it no longer adds submodules (that only ever worked locally).
cd "$(dirname "$0")"
git submodule update --init --recursive
echo "[Prism RP] Submodules ready! Run 'scons shaders' and 'scons platform=linux target=template_debug' to build."
