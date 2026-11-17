#!/usr/bin/env python3
"""Static check that GDScript in the repo only uses names that exist in Godot's API.

Compares property assignments, method calls, signal connections and ALL_CAPS class constants in
every .gd file against godot-cpp's extension_api.json (Godot 4.4.1). It's a name check, not a type
checker: it catches typos and removed/renamed APIs (like DirectionalLight3D.volumetric_fog_energy)
before a Godot run does. Locally defined functions are allowed as method names.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
API_JSON = REPO / "thirdparty" / "godot-cpp" / "gdextension" / "extension_api.json"
SCRIPT_GLOBS = ("project/**/*.gd", "addons/**/*.gd")

# Prism RP's own classes/properties are bound from C++ and aren't in extension_api.json.
PRISM_CPP_SOURCES = ("addons/prism_rp/src/*.cpp",)


def load_api_names() -> dict[str, set[str]]:
    api = json.loads(API_JSON.read_text(encoding="utf-8"))
    names: dict[str, set[str]] = {k: set() for k in ("methods", "properties", "signals", "constants")}
    for c in api["classes"] + api["builtin_classes"]:
        names["methods"].update(m["name"] for m in c.get("methods", []))
        names["properties"].update(p["name"] for p in c.get("properties", []) + c.get("members", []))
        names["signals"].update(s["name"] for s in c.get("signals", []))
        names["constants"].update(k["name"] for k in c.get("constants", []))
        for e in c.get("enums", []):
            names["constants"].update(v["name"] for v in e["values"])
    names["methods"].update(u["name"] for u in api["utility_functions"])
    for pattern in PRISM_CPP_SOURCES:
        for cpp in REPO.glob(pattern):
            src = cpp.read_text(encoding="utf-8")
            names["methods"].update(re.findall(r'D_METHOD\("(\w+)"', src))
            names["properties"].update(re.findall(r'PropertyInfo\([^,]+,\s*"(\w+)"', src))
    return names


def check_file(path: Path, names: dict[str, set[str]]) -> list[str]:
    src = path.read_text(encoding="utf-8")
    local_funcs = set(re.findall(r"^func (\w+)", src, re.M))
    local_vars = set(re.findall(r"^(?:@export\S*\s+)?var (\w+)", src, re.M))
    problems: list[str] = []
    try:
        rel = path.relative_to(REPO)
    except ValueError:
        rel = path
    for n, line in enumerate(src.splitlines(), 1):
        # Drop string literals first, so text like "effect.property=value" or a "#" in a string is ignored.
        code = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', line).split("#")[0]
        for m in re.findall(r"\.([a-z_][a-z0-9_]*)\(", code):
            if m not in names["methods"] and m not in local_funcs:
                problems.append(f"{rel}:{n}: unknown method .{m}()")
        for p in re.findall(r"\b[a-z_][a-z0-9_]*\.([a-z_][a-z0-9_]*)\s*[+\-*/]?=(?!=)", code):
            if p not in names["properties"] and p not in local_vars:
                problems.append(f"{rel}:{n}: unknown property .{p}")
        for s in re.findall(r"\b([a-z_]+)\.connect\(", code):
            if s not in names["signals"] and s not in names["properties"] and s not in local_vars:
                problems.append(f"{rel}:{n}: unknown signal {s}")
        for c in re.findall(r"\b[A-Z][A-Za-z0-9]+\.([A-Z][A-Z0-9_]+)\b", code):
            if c not in names["constants"]:
                problems.append(f"{rel}:{n}: unknown constant {c}")
    return problems


def run() -> list[str]:
    names = load_api_names()
    problems: list[str] = []
    for pattern in SCRIPT_GLOBS:
        for gd in sorted(REPO.glob(pattern)):
            problems += check_file(gd, names)
    return problems


if __name__ == "__main__":
    found = run()
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
