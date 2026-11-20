#!/usr/bin/env bash
# Runs shade_materials.gd on the custom editor with the Vulkan validation layers on and prints PASS
# or FAIL. The Track B material shading shader indexes a sampled-image array with nonuniformEXT, so a
# device that did not enable the descriptor-indexing features fails here with a validation error.
# Any validation error or engine ERROR line fails the run.
#
# Usage: tests/engine/run_shade_materials.sh <custom-godot-editor-binary>
# Point VK_DRIVER_FILES at lavapipe to run without a GPU.
set -uo pipefail

GODOT="${1:?usage: run_shade_materials.sh <custom-godot-editor-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SHADERS="$DIR/../../addons/prism_rp/shaders"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS-VK_LAYER_KHRONOS_validation}"

# CI has Xvfb; a developer machine without it gets a real window.
WRAP=()
if command -v xvfb-run > /dev/null; then
  WRAP=(xvfb-run -a -s "-screen 0 1280x720x24")
fi

# Not --headless: headless forces the dummy renderer, so there would be no RenderingDevice.
"${WRAP[@]}" "$GODOT" --rendering-driver vulkan --script "$DIR/shade_materials.gd" -- "--shader-dir=$(cd "$SHADERS" && pwd)" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ] || ! grep -q "SHADE_MATERIALS_TEST: PASS" "$LOG" \
  || grep -qE "VUID-|SYNC-HAZARD|^ERROR: |SCRIPT ERROR|FAILED:" "$LOG"; then
  echo "FAIL"
  exit 1
fi
echo "PASS"
