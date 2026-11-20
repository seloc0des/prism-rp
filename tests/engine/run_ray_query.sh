#!/usr/bin/env bash
# Runs ray_query.gd (ray queries against a one-triangle TLAS, patch 0006) on the custom editor with the Vulkan
# validation layers and synchronization validation on, and prints PASS or FAIL. Two passes:
#   1. The default pass traces. Any validation error, sync hazard, leaked RID or engine ERROR line fails it.
#   2. The "negative" pass builds a TLAS over a BLAS that was never built and over a freed one. Each must produce
#      its engine error message, and nothing may crash, leak or trip a validation error.
#
# Usage: tests/engine/run_ray_query.sh <custom-godot-editor-binary>
# Point VK_DRIVER_FILES at lavapipe to run without a GPU. An empty VK_INSTANCE_LAYERS turns the validation layer off.
set -uo pipefail

GODOT="${1:?usage: run_ray_query.sh <custom-godot-editor-binary>}"
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
run() {  # $1 = marker the script prints on success, remaining = script user arguments
  local marker="$1"
  shift
  "${WRAP[@]}" "$GODOT" --rendering-driver vulkan --script "$DIR/ray_query.gd" -- "$@" 2>&1 | tee "$LOG"
  local status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || fail "$marker exited with status $status"
  grep -q "$marker: PASS" "$LOG" || fail "$marker did not print PASS"
  grep -qE "VUID-|SYNC-HAZARD|SCRIPT ERROR|FAILED:" "$LOG" && fail "$marker log has a validation error, sync hazard or failed check"
  # RenderingDevice frees leaked RIDs itself at teardown, so only its warning shows them.
  grep -qE 'RIDs? of type ".*" (was|were) leaked' "$LOG" && fail "$marker leaked RIDs"
}

expect() {  # $1 = message fragment, $2 = minimum number of occurrences
  local count
  count=$(grep -cF "$1" "$LOG")
  [ "$count" -ge "$2" ] || fail "expected the error '$1' at least $2 time(s), saw $count"
}

run RAY_QUERY_TEST
grep -q "^ERROR: " "$LOG" && fail "the default pass printed an engine error"

run RAY_QUERY_NEGATIVE_TEST negative
expect "BLAS either has not been built yet, or has been invalidated by an operation and needs to be rebuilt." 1
expect 'Parameter "blas" is null.' 1

if [ "$failed" -ne 0 ]; then
  echo "FAIL"
  exit 1
fi
echo "PASS"
