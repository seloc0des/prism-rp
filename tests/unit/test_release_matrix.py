import re

import yaml

from conftest import REPO


def test_release_builds_every_gdextension_library():
    gdext = (REPO / "addons/prism_rp/prism_rp.gdextension").read_text()
    wanted = set(re.findall(r"bin/libprism_rp\.(\w+)\.(template_\w+)\.(\w+)\.", gdext))
    workflow = yaml.safe_load((REPO / ".github/workflows/release.yml").read_text())
    built = {(r["platform"], r["target"], r["arch"]) for r in workflow["jobs"]["build"]["strategy"]["matrix"]["include"]}
    assert wanted - built == set(), f"no release build for: {sorted(wanted - built)}"
