def can_build(env, platform):
    # The Godot editor is Linux only; shipped games target Linux and Windows.
    return platform in ("linuxbsd", "windows")


def configure(env):
    pass
