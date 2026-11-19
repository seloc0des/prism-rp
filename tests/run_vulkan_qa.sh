#!/usr/bin/env bash
# Boot every benchmark scene on Vulkan with the validation layers on, once with Prism RP effects
# off (baseline) and once with them on, and fail on anything Prism RP adds.
#
# Usage: tests/run_vulkan_qa.sh <godot-binary>
# Run from the repository root with the addon already linked into project/ and imported.
# Writes godot_baseline.log, godot_vulkan_qa.log and the *_findings.txt files to the current directory.
# No `set -e`: the grep checks below inspect exit codes on purpose.
set -o pipefail

GODOT="${1:?usage: run_vulkan_qa.sh <godot-binary>}"
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS:-VK_LAYER_KHRONOS_validation}"
SCENES="01_cornell_box 02_foliage_overdraw 03_thin_geometry_aliasing 04_dynamic_sponza 05_volumetrics_wboit"

# CI has Xvfb; a developer machine without it gets real windows.
if command -v xvfb-run > /dev/null; then
  DISPLAY_WRAP=(xvfb-run -a -s "-screen 0 1280x720x24")
else
  echo "xvfb-run not found: running with real windows"
  DISPLAY_WRAP=()
fi

run_scenes() {  # $1 = log file, remaining = rig user args
  local log="$1"; shift
  : > "$log"
  for scene in $SCENES; do
    echo "=== ${scene}" | tee -a "$log"
    # Not --headless: headless forces the dummy renderer, so Vulkan would never start.
    "${DISPLAY_WRAP[@]}" \
      "$GODOT" --rendering-driver vulkan --path project --fixed-fps 60 \
        "res://benchmarks/${scene}.tscn" --quit-after 30 -- "$@" 2>&1 | tee -a "$log"
  done
}

# Baseline first: same scenes with every Prism RP effect off. Anything it reports comes
# from Godot, Mesa or the validation layers themselves, not from this repository.
run_scenes godot_baseline.log --prism-chronos=off --prism-lumina=off
run_scenes godot_vulkan_qa.log

# Godot silently falls back to OpenGL when Vulkan fails; that must not pass as a Vulkan run.
if grep -q "switching to OpenGL" godot_vulkan_qa.log || ! grep -q "Forward+ - Using Device" godot_vulkan_qa.log; then
  echo "::error::Godot did not run on Vulkan Forward+."
  exit 1
fi
if ! grep -q "\[Prism RP\] Registered" godot_vulkan_qa.log; then
  echo "::error::Prism RP GDExtension did not register (extension not loaded)."
  exit 1
fi

# Always fatal, baseline or not.
if grep -E "SCRIPT ERROR|Parse Error|Can't open dynamic library|Failed to retrieve non-existent singleton|\[Prism RP\].*([Ff]ailed|FAILED|not found)" godot_vulkan_qa.log; then
  echo "::error::Script, GDExtension or Prism RP errors detected!"
  exit 1
fi

# Validation IDs and engine ERROR lines: fatal only if Prism RP introduced them.
extract() { { grep -oE "VUID-[A-Za-z0-9_-]+|SYNC-HAZARD-[A-Z_-]+" "$1"; grep -E "^ERROR: " "$1"; } | LC_ALL=C sort -u; }
extract godot_baseline.log > baseline_findings.txt || true
extract godot_vulkan_qa.log > prism_findings.txt || true
echo "--- present in baseline too (not ours):"; LC_ALL=C comm -12 baseline_findings.txt prism_findings.txt
LC_ALL=C comm -13 baseline_findings.txt prism_findings.txt > new_findings.txt
if [ -s new_findings.txt ]; then
  echo "::error::Vulkan validation / engine errors introduced by Prism RP:"
  cat new_findings.txt
  exit 1
fi
