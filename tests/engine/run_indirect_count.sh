#!/usr/bin/env bash
# Runs indirect_count.gd on the custom editor with the Vulkan validation layers on and prints PASS
# or FAIL. Any validation error or engine ERROR line fails the run.
#
# Usage: tests/engine/run_indirect_count.sh <custom-godot-editor-binary>
# Point VK_DRIVER_FILES at lavapipe to run without a GPU.
set -uo pipefail

GODOT="${1:?usage: run_indirect_count.sh <custom-godot-editor-binary>}"
SCRIPT="$(dirname "$0")/indirect_count.gd"
LOG="$(mktemp)"
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS:-VK_LAYER_KHRONOS_validation}"

# CI has Xvfb; a developer machine without it gets a real window.
WRAP=()
if command -v xvfb-run > /dev/null; then
  WRAP=(xvfb-run -a -s "-screen 0 1280x720x24")
fi

# Not --headless: headless forces the dummy renderer, so there would be no RenderingDevice.
"${WRAP[@]}" "$GODOT" --rendering-driver vulkan --script "$SCRIPT" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ] || ! grep -q "INDIRECT_COUNT_TEST: PASS" "$LOG" \
  || grep -qE "VUID-|SYNC-HAZARD|^ERROR: |SCRIPT ERROR" "$LOG"; then
  echo "FAIL"
  rm -f "$LOG"
  exit 1
fi
echo "PASS"
rm -f "$LOG"
