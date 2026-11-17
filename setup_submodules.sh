#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".git" ]; then
    echo "[Prism RP] Initializing Git repository..."
    git init
fi

if [ ! -f "thirdparty/godot-cpp/SConstruct" ]; then
    echo "[Prism RP] Adding godot-cpp (branch 4.4) submodule..."
    git submodule add -f -b 4.4 https://github.com/godotengine/godot-cpp.git thirdparty/godot-cpp
fi

if [ ! -f "thirdparty/meshoptimizer/CMakeLists.txt" ]; then
    echo "[Prism RP] Adding meshoptimizer submodule..."
    git submodule add -f https://github.com/zeux/meshoptimizer.git thirdparty/meshoptimizer
fi

git submodule update --init --recursive
echo "[Prism RP] Submodules ready! Run 'scons shaders' and 'scons platform=linux target=template_debug' to build."
