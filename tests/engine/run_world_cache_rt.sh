#!/usr/bin/env bash
# Runs world_cache_rt.gd (Lumina's ray-traced world cache update lighting the view from off-screen lights and
# emission) on the custom editor with the Vulkan validation layers and synchronization validation on, and prints PASS
# or FAIL. Any validation error, sync hazard, leaked RID, engine ERROR line or failed check fails it.
#
# Usage: tests/engine/run_world_cache_rt.sh <custom-godot-editor-binary>
# Run from the repository root with the addon linked into project/ and imported (see tests/run_vulkan_qa.sh).
# Point VK_DRIVER_FILES at lavapipe to run without a GPU. An empty VK_INSTANCE_LAYERS turns the validation layer off.
set -uo pipefail
GODOT="${1:?usage: run_world_cache_rt.sh <custom-godot-editor-binary>}"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/world_cache_rt.gd"
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS-VK_LAYER_KHRONOS_validation}"
export VK_LAYER_ENABLES="${VK_LAYER_ENABLES:-VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT}"
# CI has Xvfb; a developer machine without it gets a real window.
WRAP=()
if command -v xvfb-run > /dev/null; then
  WRAP=(xvfb-run -a -s "-screen 0 1280x720x24")
fi
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
failed=0
fail() {
  echo "FAIL: $1"
  failed=1
}
# Not --headless: headless forces the dummy renderer, so there would be no RenderingDevice.
"${WRAP[@]}" "$GODOT" --rendering-driver vulkan --audio-driver Dummy --path project --resolution 320x180 --script "$SCRIPT" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
[ "$status" -eq 0 ] || fail "the test exited with status $status"
grep -q "WORLD_CACHE_RT_TEST: PASS" "$LOG" || fail "the test did not print PASS"
grep -qE "VUID-|SYNC-HAZARD|SCRIPT ERROR|FAILED:" "$LOG" && fail "the log has a validation error, sync hazard or failed check"
grep -q "^ERROR: " "$LOG" && fail "the log has an engine error"
# RenderingDevice frees leaked RIDs itself at teardown, so only its warning shows them.
grep -qE 'RIDs? of type ".*" (was|were) leaked' "$LOG" && fail "the test leaked RIDs"
if [ "$failed" -ne 0 ]; then
  echo "FAIL"
  exit 1
fi
echo "PASS"
