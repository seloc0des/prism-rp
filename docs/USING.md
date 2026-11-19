# Using Prism RP in your own project

Prism RP is a GDExtension addon for **Godot 4.7, Forward+, Vulkan**, on Linux and Windows (x86_64).
What to expect today:

| Effect | State |
| :--- | :--- |
| **Chronos** (temporal anti-aliasing, FFT bloom, ACES look) | Ready. Verified on an RTX 4060 Ti: about 0.3 ms per frame at Tier 2, image-quality gate passed in every test scene. |
| **Lumina** (screen-space global illumination) | Usable, but not yet better than Godot's SDFGI (see `docs/NEXT_STEPS.md`). Fast (under 2 ms) and settles within a frame of a lighting change. |

## 1. Install it into your project

From this repository:

```bash
python3 tools/install_addon.py ~/path/to/your_game
```

It builds the extension (debug for the editor and debug exports, release for release exports) for
Linux and Windows and copies `addons/prism_rp` into your project, replacing an older copy. The Windows
build is cross-compiled on Linux with MinGW-w64:

```bash
sudo apt install g++-mingw-w64-x86-64-posix   # once; Debian / Ubuntu / Pop!_OS
```

Use `--platforms linux` if you only want Linux. Re-run the script whenever you pull a newer Prism RP.

Then in Godot: **Project Settings > Plugins > enable Prism RP.** Your project must use the
**Forward+** renderer (Project Settings > Rendering > Renderer).

## 2. Add the effects

Both effects are `CompositorEffect`s on a `Compositor`, set on your `WorldEnvironment` (or a camera):

```gdscript
var compositor := Compositor.new()
compositor.compositor_effects = [
    LuminaCascadesCompositorEffect.new(),   # optional: global illumination
    ChronosOpaqueCaptureEffect.new(),       # keeps particles and glass from smearing
    ChronosVPUCompositorEffect.new(),       # anti-aliasing, bloom, look
]
$WorldEnvironment.compositor = compositor
```

You can also build the same `Compositor` resource in the Inspector and save it with the scene.

**Chronos**
- Use `ChronosJitterCamera` (`res://addons/prism_rp/scripts/chronos_jitter_camera.gd`) instead of a
  plain `Camera3D`: Chronos needs the sub-pixel jitter it adds. You set `fov` as usual.
- Leave Godot's own TAA off (`Viewport.use_taa`), and don't use FSR2 as `scaling_3d_mode`: they would
  filter the image a second time, so Chronos switches its TAA off and warns if it sees them.
- With `aces_odt_enabled` on (Chronos' own look), set the Environment's tonemapper to Linear,
  exposure 1. With it off, keep whichever tonemapper you like.

**Lumina**
- Turn SDFGI, VoxelGI and LightmapGI off in the same Environment, or the bounce light counts twice.
- Lumina is diffuse only: for metals and mirrors, turn on screen-space reflections
  (`Environment.ssr_enabled`) or add ReflectionProbes.
- It only sees what's on screen, so light from surfaces behind the camera or off the frame edges is
  missing. `world_cascades_enabled` (off by default) remembers light seen in earlier frames.

The README has the details of each setting; the editor dock (bottom panel, once the plugin is
enabled) shows per-pass GPU timings and debug views.

## 3. Export your game

Exports work like any Godot project once Godot's **export templates** are installed (Editor > Manage
Export Templates). Godot copies the right Prism RP library next to the exported executable.

In each export preset, add `*.glsl` to **Resources > Filters to export non-resource files**: Prism RP
compiles its shaders from those files at runtime.

- **Linux:** the release export runs Prism RP from `libprism_rp.linux.template_release.x86_64.so`.
- **Windows:** the release export uses `libprism_rp.windows.template_release.x86_64.dll`. It needs only
  system DLLs (KERNEL32, msvcrt), so it runs on any 64-bit Windows with a Vulkan driver. You can test a
  Windows export on Linux with Wine (or Proton), which passes Vulkan through to your GPU.

GPU timings are off in release builds; start the game with `-- --prism-telemetry=on` to profile one.

`tests/check_export.py` exports a test scene for both platforms and runs it (Windows through Wine if
installed); the GPU session runs it as its `export` stage, and CI runs the Linux export on every change.
