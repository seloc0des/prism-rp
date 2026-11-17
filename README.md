# Prism RP: Hybrid Visibility & Radiance Pipeline for Godot 4

**Prism RP** is a modular, GPU-driven rendering and temporal post-processing project built on Godot's `RenderingDevice`. The long-term goal is to close the visual and scalability gap between Godot 4 Forward+ and high-end engines (Unreal Engine 5's Nanite/Lumen/Substrate, Unity HDRP) while staying lightweight, cross-platform and MIT-licensed.

Most of that is still ahead. Today the repository ships one working piece, the **Chronos VPU** compositor effect, plus the build, CI and QA scaffolding the rest will be measured against. The table marks what exists and what is planned; `ROADMAP.md` has the schedule.

## Status

| Subsystem | Prism RP target | State |
| :--- | :--- | :--- |
| **Video & Temporal Post** | **Chronos VPU**: dilated motion vectors, reactive mask, compute TAA, FFT diffraction bloom with red halation, SDR filmic look | Implemented (GDExtension `CompositorEffect`) |
| Upscaler backends | FSR 3.1 / XeSS / DLSS behind one interface | Planned (Phase 1) |
| **Global Illumination** | **Lumina-Cascades**: screen-space + world-space radiance cascades with ReSTIR and SVGF | Planned (Phase 2) |
| **Geometry & Culling** | **GPU Meshlet Pipeline** (`VK_EXT_mesh_shader` + compute fallback) into a 64-bit Visibility Buffer | Planned (Phase 3) |
| **Material Shading** | **Tile-Sorted Compute Shading** into a Thin G-Buffer | Planned (Phase 4) |
| **Shadows** | **Virtualized Clipmap Shadow Pages (VCSP)** driven by visibility-buffer page faults | Planned (Phase 4) |

## How Chronos fits into Godot 4.4

Godot 4.4 runs compositor effects at `POST_TRANSPARENT`. That is before Godot's own TAA/FSR2, its upscale to the output size, and its tonemap pass. A few things follow from that, and Chronos is built around them:

* **Everything resolves at the render (internal) resolution.** Godot then upscales to the window size with whatever `scaling_3d_mode` you set. The compositor API has no way to hand Godot a frame at a different resolution, so Chronos is a temporal anti-aliaser today, not an upscaler.
* **Output is linear Rec.709.** With `aces_odt_enabled` off, Chronos writes scene-referred HDR and your Environment's tonemapper takes it from there. With it on, Chronos applies its own SDR look (ACES 1.3 reference gamut compression plus a filmic tonescale; this is *not* the ACES 2.0 ODT) and writes display-referred linear values. In that case set the Environment to `TONE_MAPPER_LINEAR` with exposure 1.0 so Godot only does the sRGB encode.
* **Display P3 and HDR10 aren't supported yet.** Godot 4.4 has no HDR or wide-gamut swapchain. The `output_display_mode` values exist for later, warn, and render as SDR sRGB.
* **Jitter comes from the camera, not from Godot.** Godot only jitters when its own TAA or FSR2 is enabled, and those would run after Chronos and filter the image a second time. Use `ChronosJitterCamera` (`addons/prism_rp/scripts/chronos_jitter_camera.gd`) in place of `Camera3D`. You set `fov` as usual; at runtime it switches to a frustum projection and moves `frustum_offset` along a Halton(2,3) sequence. Chronos reads the offset back from each eye's projection and removes it from Godot's motion vectors.
* **Godot's own temporal modes switch Chronos TAA off.** If `Viewport.use_taa` is on, or `scaling_3d_mode` is FSR2 or MetalFX Temporal, Chronos warns once and bypasses its TAA (bloom and the SDR look still run) rather than resolve twice.

### Particles and transparent surfaces

These write neither depth nor motion vectors, so a TAA history reprojects the background under them and smears them. Add a `ChronosOpaqueCaptureEffect` to the same `Compositor` as Chronos. It copies the opaque-only frame at `PRE_TRANSPARENT`, and Chronos flags every pixel the transparent pass changed in its reactive mask (the same opaque-vs-final comparison FSR 2 uses to generate its reactive mask). `transparency_reactive_scale` sets how hard those pixels drop their history.

```gdscript
var compositor := Compositor.new()
compositor.compositor_effects = [ChronosOpaqueCaptureEffect.new(), ChronosVPUCompositorEffect.new()]
```

### Telemetry

`PrismTelemetry` records per-pass GPU timestamps once per frame, on the render thread. It is on by default in debug and editor builds and off in `template_release`. Pass `--prism-telemetry=on` (after `--`) or call `PrismTelemetry.set_enabled(true)` to profile a release build. The bloom shows up as three stages (`chronos_fft_bloom_forward_rows`, `chronos_fft_bloom_columns_convolve`, `chronos_fft_bloom_inverse_rows`), so the Chronos budget can be broken down per dispatch.

## Quick-Start Setup

1. **Fetch submodules** (godot-cpp, pinned to `godot-4.4.1-stable`):
   ```bash
   ./setup_submodules.sh
   ```

2. **Validate GLSL 4.60 shaders as Vulkan 1.3 SPIR-V** (needs `glslangValidator`):
   ```bash
   scons shaders
   ```

3. **Build the C++20 GDExtension**:
   * **SCons**:
     ```bash
     scons platform=linux target=template_debug -j$(nproc)
     ```
   * **CMake + Ninja**:
     ```bash
     cmake -B build_cmake -G Ninja -DGODOT_TARGET=template_debug -DPRISM_RP_VALIDATE_SHADERS=ON
     cmake --build build_cmake -j$(nproc)
     ```

4. **Link the addon into the QA project and import it once.** The import writes `.godot/extension_list.cfg`; Godot won't load the extension until it exists.
   ```bash
   ln -s "$(pwd)/addons/prism_rp" project/addons/prism_rp
   ```
   ```bash
   godot --headless --path project --import
   ```

5. **Run the visual and frame-budget QA harness** (Godot 4.4.1 on `PATH`, real Vulkan; on headless Linux it wraps itself in `xvfb-run`):
   ```bash
   pip install flip-evaluator numpy pillow
   ```
   ```bash
   python3 tests/run_visual_regression.py --tier tier2 --update-golden
   ```
   ```bash
   python3 tests/run_visual_regression.py --tier tier2 --max-flip 0.015 --max-pass-delta-ms 0.25
   ```
   The benchmark rig saves the displayed frame and the PrismTelemetry GPU timings at the capture frame. Add `--renderdoc` to run under `renderdoccmd`: the rig triggers a capture through RenderDoc's in-app API, and if RenderDoc's Python module can be imported (`RENDERDOC_PYTHON_PATH`), per-pass durations come from the capture instead.

## Exporting projects

The `.glsl` files are marked "Keep File (exported as is)" through their committed `.import` files, so Chronos can read them at runtime in exported builds. Keep those `.import` files next to the shaders when you copy the addon.

See `ROADMAP.md` for the full 24-week plan, hardware tier budgets and QA gates.

## License

MIT; see `LICENSE`. `thirdparty/renderdoc/renderdoc_app.h` is RenderDoc's in-app API header, MIT-licensed by Baldur Karlsson.
