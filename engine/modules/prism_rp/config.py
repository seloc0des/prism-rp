def can_build(env, platform):
    # The Godot editor is Linux only; shipped games target Linux and Windows.
    return platform in ("linuxbsd", "windows")


def configure(env):
    # Makes the version 4.7.2.stable.prism, so the editor reads export templates from their own folder and the binaries
    # carry a .prism suffix. Stock 4.7.2 templates and ours then sit side by side.
    env.add_module_version_string("prism")
