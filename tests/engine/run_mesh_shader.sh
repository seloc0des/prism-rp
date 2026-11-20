#!/usr/bin/env bash
# Runs mesh_shader.gd (mesh and task shaders) and mesh_indirect.gd (indirect mesh draws) on the
# custom editor with the Vulkan validation layers and synchronization validation on, and prints
# PASS or FAIL. Each script runs two passes:
#   1. The default pass draws. Any validation error, sync hazard or engine ERROR line fails it.
#   2. The "negative" pass misuses the API on purpose. Each misuse must produce its engine error
#      message, and nothing may crash or trip a validation error.
#
# Usage: tests/engine/run_mesh_shader.sh <custom-godot-editor-binary>
# Point VK_DRIVER_FILES at lavapipe to run without a GPU. An empty VK_INSTANCE_LAYERS turns the
# validation layer off (the Khronos layers older than 1.3.226 cannot validate mesh shaders).
set -uo pipefail

GODOT="${1:?usage: run_mesh_shader.sh <custom-godot-editor-binary>}"
DIR="$(dirname "$0")"
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
run() {  # $1 = marker the script prints on success, $2 = script, remaining = script user arguments
  local marker="$1" script="$DIR/$2"
  shift 2
  "${WRAP[@]}" "$GODOT" --rendering-driver vulkan --script "$script" -- "$@" 2>&1 | tee "$LOG"
  local status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || fail "$marker exited with status $status"
  grep -q "$marker: PASS" "$LOG" || fail "$marker did not print PASS"
  grep -qE "VUID-|SYNC-HAZARD|SCRIPT ERROR|FAILED:" "$LOG" && fail "$marker log has a validation error, sync hazard or failed check"
}

expect() {  # $1 = message fragment, $2 = minimum number of occurrences
  local count
  count=$(grep -cF "$1" "$LOG")
  [ "$count" -ge "$2" ] || fail "expected the error '$1' at least $2 time(s), saw $count"
}

run MESH_SHADER_TEST mesh_shader.gd
grep -q "^ERROR: " "$LOG" && fail "the default pass printed an engine error"

run MESH_SHADER_NEGATIVE_TEST mesh_shader.gd negative
expect "A mesh shader cannot be combined with vertex or tessellation shaders." 1
expect "Mesh shader pipelines take no vertex format." 1
expect "A task shader requires a mesh shader." 1
expect "Pre-raster shader (vertex or mesh shader) is not provided for pipeline creation." 1
expect "The bound render pipeline uses a mesh shader, use draw_list_draw_mesh_tasks." 3
expect "The bound render pipeline has no mesh shader." 1
expect "is larger than device limit" 2

run MESH_INDIRECT_TEST mesh_indirect.gd
grep -q "^ERROR: " "$LOG" && fail "the indirect pass printed an engine error"

run MESH_INDIRECT_NEGATIVE_TEST mesh_indirect.gd negative
expect "Buffer provided was not created to do indirect dispatch." 1
expect "Count buffer provided was not created to do indirect dispatch." 1
expect "Mesh tasks commands extend past the end of the buffer." 3
expect "Count buffer offset provided (+4) is past the end of buffer." 1
expect "Stride must be at least 12 bytes" 1
expect "Offsets and stride must be multiples of 4." 2
expect "The bound render pipeline has no mesh shader." 2

if [ "$failed" -ne 0 ]; then
  echo "FAIL"
  exit 1
fi
echo "PASS"
