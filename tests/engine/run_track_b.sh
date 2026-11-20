#!/usr/bin/env bash
# Checks the Track B switch of the custom editor and prints PASS or FAIL:
#   - the project setting exists with the default Off (headless, no GPU needed);
#   - on Vulkan, the engine logs exactly one "[Prism RP] Track B" line, with the setting as its source
#     by default and the command line when --prism-trackb=auto is given, and always reports the mesh
#     GPU path on lavapipe;
#   - the stock Forward+ renderer is the one running in both cases.
#
# Usage: tests/engine/run_track_b.sh <custom-godot-editor-binary>
# Point VK_DRIVER_FILES at lavapipe to run without a GPU.
set -uo pipefail

GODOT="${1:?usage: run_track_b.sh <custom-godot-editor-binary>}"
DIR="$(dirname "$0")"

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

expect_once() {
  local count
  count=$(grep -cF "$1" "$LOG")
  [ "$count" -eq 1 ] || fail "expected '$1' exactly once, saw $count"
}

"$GODOT" --headless --script "$DIR/track_b_setting.gd" 2>&1 | tee "$LOG"
grep -q "TRACK_B_SETTING_TEST: PASS" "$LOG" || fail "the Track B setting check did not pass"

# Not --headless: headless forces the dummy renderer, which never creates the scene renderer.
run_vulkan() {  # remaining = script user arguments
  "${WRAP[@]}" "$GODOT" --rendering-driver vulkan --quit -- "$@" 2>&1 | tee "$LOG"
  expect_once "[Prism RP] Track B:"
  grep -q "Forward+ - Using Device" "$LOG" || fail "the Vulkan Forward+ renderer did not start"
  grep -qE "SCRIPT ERROR|VUID-" "$LOG" && fail "the run printed a script error or validation error"
}

run_vulkan
expect_once "[Prism RP] Track B: off (setting) | GPU path: mesh"

run_vulkan --prism-trackb=auto
expect_once "[Prism RP] Track B: auto (command line) | GPU path: mesh"

if [ "$failed" -ne 0 ]; then
  echo "FAIL"
  exit 1
fi
echo "PASS"
