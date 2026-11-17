from pathlib import Path
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]

MESHOPT = REPO / "thirdparty" / "meshoptimizer" / "src"


@pytest.mark.skipif(shutil.which("g++") is None or not MESHOPT.exists(), reason="needs g++ and the meshoptimizer submodule")
def test_meshlet_builder(tmp_path):
    exe = tmp_path / "test_meshlet_builder"
    sources = [str(REPO / "tests/meshlet/test_meshlet_builder.cpp"), str(REPO / "addons/prism_rp/src/meshlet/meshlet_builder.cpp")]
    sources += [str(p) for p in sorted(MESHOPT.glob("*.cpp"))]
    subprocess.run(["g++", "-std=c++20", "-O1", f"-I{REPO / 'addons/prism_rp/include'}", f"-I{MESHOPT}", *sources, "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    print(run.stdout)
    assert run.returncode == 0, run.stderr
