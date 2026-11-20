from pathlib import Path
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]

# The GDExtension builds against the repo's submodule (src/ layout). The engine module builds the same
# builder against Godot's own copy (a flat directory, older), so the tests run against both.
MESHOPT_DIRS = {
    "submodule": REPO / "thirdparty" / "meshoptimizer" / "src",
    "godot": REPO / "engine" / "godot" / "thirdparty" / "meshoptimizer",
}


@pytest.mark.parametrize("name", MESHOPT_DIRS)
def test_meshlet_builder(tmp_path, name):
    meshopt = MESHOPT_DIRS[name]
    if shutil.which("g++") is None or not (meshopt / "meshoptimizer.h").exists():
        pytest.skip(f"needs g++ and the {name} meshoptimizer sources")
    exe = tmp_path / "test_meshlet_builder"
    sources = [str(REPO / "tests/meshlet/test_meshlet_builder.cpp"), str(REPO / "addons/prism_rp/src/meshlet/meshlet_builder.cpp")]
    sources += [str(p) for p in sorted(meshopt.glob("*.cpp"))]
    subprocess.run(["g++", "-std=c++20", "-O1", f"-I{REPO / 'addons/prism_rp/include'}", f"-I{meshopt}", *sources, "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    print(run.stdout)
    assert run.returncode == 0, run.stderr
