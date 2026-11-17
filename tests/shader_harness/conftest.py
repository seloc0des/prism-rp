import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402


def _vulkan_device_available() -> bool:
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=60)
        return out.returncode == 0 and "deviceName" in out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def gpu():
    if not harness.toolchain_available():
        pytest.skip("needs glslangValidator, g++, libvulkan-dev and nlohmann-json")
    if not _vulkan_device_available():
        pytest.skip("no Vulkan device (install mesa-vulkan-drivers for lavapipe)")
    return harness.Harness(validation=True)
