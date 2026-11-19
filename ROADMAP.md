# Prism RP: 24-Week Engineering Execution and Verification Plan

Building **Prism RP** for Godot 4.x uses a dual-track engineering strategy:

* **Track A: Drop-In `CompositorEffect` Plugin (Weeks 1-11):** Ships **Chronos VPU** (SVGF denoising, reactive-mask temporal upscaling, ACEScg optics) and **Lumina-Cascades** (screen-space and SDF radiance cascades) as a pure C++20 GDExtension usable immediately in existing Godot Forward+ projects.
* **Track B: Native `RendererSceneRenderRD` Backend (Weeks 12-24):** Implements the full **64-bit Visibility Buffer**, tile-sorted compute material shading, and **Virtual Shadow Pages** as a selectable rendering driver (`prism_rp`), built as a Godot engine module on a custom Godot 4.7.2 (decisions below).

## Engine and Platform Decisions (2026-11-19)
* **Custom Godot build:** Prism RP becomes a Godot engine module, plus a small, separate patch set on Godot 4.7.2 that adds a mesh-shader stage and indirect-count draws to `RenderingDevice`. This unblocks Gates 3 and 4.
* **Track A stays portable:** Chronos and Lumina remain a GDExtension that also works in stock Godot.
* **Editor:** Linux only.
* **Shipped games:** Linux and Windows only, so the custom build provides export templates for those two (Windows cross-compiled with MinGW). No macOS, iOS, Android or web.
* **Fallbacks:** every feature checks GPU support at startup and falls back to stock Godot rendering.

## Hardware Tier Frame Budgets
* **Tier 1 (Steam Deck / RDNA2):** 720p internal -> 1080p output, **16.6 ms** total frame budget (60 FPS, **13.50 ms** Prism RP target).
* **Tier 2 (Mid-Range Desktop RTX 3060 / RX 6700):** 1080p internal -> 1440p output, **11.1 ms** total frame budget (90 FPS, **9.15 ms** Prism RP target).
* **Tier 3 (Enthusiast RT GPU):** 1440p internal -> 4K output, **8.3 ms** total frame budget (120 FPS, **6.80 ms** Prism RP target).

---

## Phased Milestones & Definition of Done (DoD)

### Phase 0: Architecture Spike, Repo Scaffold & Automated Visual QA Harness (Weeks 1–2)
* Stand up C++20 GDExtension build (`SCons` and `CMake`) with `glslangValidator` SPIR-V verification.
* Build 5 canonical test scenes (`01_cornell_box`, `02_foliage_overdraw`, `03_thin_geometry_aliasing`, `04_dynamic_sponza`, `05_volumetrics_wboit`).
* **Gate 0 DoD:** CI builds cleanly on Windows and Linux (macOS isn't a target); a lavapipe Vulkan boot of all 5 scenes (under Xvfb, not `--headless`, which uses the dummy renderer) passes with zero `VK_LAYER_KHRONOS_validation` errors.

### Phase 1: Chronos VPU Core & ACEScg Color Pipeline (Weeks 3–6 — Track A)
* Implement `ChronosVPUCompositorEffect` on `RenderSceneBuffersRD`.
* Deliver scene-referred **ACEScg (AP1)** color management, 3x3 closest-depth dilated velocity vectors, `R8_UNORM` Reactive Mask generation, native compute TAA (Godot's compositor effects can't output an upscaled frame, through 4.8) + FSR 3.1 / XeSS / DLSS backend abstraction, 256x256 Radix-2 Stockham FFT diffraction bloom with red halation, and an SDR filmic look (ACES 1.3 gamut compression). Godot 4.7 added HDR output; HDR-aware Chronos output is planned (`docs/NEXT_STEPS.md`).
* **Gate 1 DoD:** NVIDIA FLIP perceptual error **< 0.08** at 50% render scale vs. 4K supersampled reference; zero particle ghosting; Chronos VPU pass time **< 1.65 ms** on Tier 2 at 1440p.

### Phase 2: Lumina-Cascades GI & SVGF Denoiser (Weeks 7–11 — Track A)
* Build hierarchical **6D Radiance Cascades** (screen-space Levels 0–2 merged into world-space probes Levels 3–5) with `VK_KHR_ray_query` and SDF cone-tracing fallback.
* Pair with **ReSTIR** reservoir resampling and multi-pass **SVGF** spatiotemporal denoising + 3-iteration À-Trous filter.
* **Status:** v1 is in (`LuminaCascadesCompositorEffect`): screen-space cascade levels 0-3 with octahedral probes, SVGF temporal + 3-iteration a-trous, applied through an albedo proxy because Forward+ has no albedo buffer. A world-space radiance cache (16^3 probes, off by default) covers recently seen off-screen light. `VK_KHR_ray_query` / SDF fallback for never-seen light and ReSTIR are still to do; the ray-query fallback and a real albedo buffer (replacing the proxy) come with the custom Godot build. Gate 2's path-traced reference runs through Blender Cycles (`tests/run_gate2.py`).
* **Gate 2 DoD** (revised after the first GPU measurement; the reasons are in `docs/NEXT_STEPS.md`). `tests/run_gate2.py` compares Lumina's indirect light with an offline path-traced reference of the scene lights' bounce light (Cycles, one diffuse bounce), as a viewer sees it (times each surface's albedo) in a matte version of each scene, and measures stock Godot's SDFGI the same way:
  * Lumina's brightness-fitted PSNR **beats SDFGI's** on every Gate 2 scene;
  * Lumina's fitted PSNR is **above the interim floor** in `docs/NEXT_STEPS.md` (screen-space Lumina); **36 dB** stays the target for the ray-query fallback, which sees off-screen and hidden surfaces;
  * no visible artifacts in the GPU test plan's Lumina checks (colour smears, darkened screen edges, blotches);
  * dynamic lights settle within **6 frames**: within 10% of the settled irradiance (`tests/check_gi_convergence.py`);
  * combined GI + SVGF budget **< 2.40 ms** on Tier 2.

### Phase 3: Meshlet Asset Pipeline & 64-Bit Visibility Buffer (Weeks 12–16 — Track B)
* Partition imported meshes via `meshoptimizer` into 64-vertex, 124-triangle meshlets with bounding spheres and backface normal cones.
* Implement two-phase GPU Hi-Z occlusion culling and rasterize into `R32G32_UINT` **Visibility Buffer** (`InstanceID`, `MeshletID`, `PrimitiveID`; barycentrics are rebuilt analytically in shading rather than stored).
* **Status:** the meshlet builder is in (`PrismMeshletBuilder`, meshoptimizer v1.3, Godot winding handled, unit-tested). Shaders for Hi-Z culling, the visibility buffer and its rasterization (mesh-shader and vertex-pulling paths, opaque and alpha-tested) are in the tree, with lavapipe tests for the compute passes, but not wired up: running them needs the custom Godot build (Godot's `RenderingDevice` has no mesh-shader stage or indirect-count draws, through 4.8). See `docs/TRACK_B_SHADERS.md`.
* **Gate 3 DoD:** 15M-triangle stress scene renders at **60+ FPS on Tier 2** in **< 0.65 ms** for culling + visibility rasterization with **1.00x** fragment invocation per visible pixel.

### Phase 4: Tile-Sorted Material Shading & Virtual Shadow Pages (Weeks 17–20 — Track B)
* Reconstruct world positions, tangent frames, and analytic `ddx`/`ddy` UV partials in compute.
* Bin 8x8 screen tiles by `MaterialID` into indirect compute kernels outputting the **Thin G-Buffer** (Standard PBR, Clear-Coat, Subsurface, Thin-Film).
* Implement 16k x 16k **Virtual Shadow Pages (VCSP)** driven by visibility-buffer page-fault feedback.
* **Status:** shaders for tile classification, material shading and virtual shadow pages (16k x 16k virtual per clipmap level, 4096x4096 R32 atlas = 64 MiB, age-based eviction, alpha-tested casters) are in the tree with lavapipe tests, but not wired up. See `docs/TRACK_B_SHADERS.md`.
* **Gate 4 DoD:** Zero foliage shadow shimmer during camera rotation; physical shadow pool capped at **64 MB**.

### Phase 5: Froxel Volumetrics, Frame Generation & Production Hardening (Weeks 21–24)
* Integrate 160x90x128 temporal froxel volumetrics, Weighted Blended Order-Independent Translucency (WBOIT), and async-compute optical-flow frame interpolation.
* Pre-warm PSOs on scene load and audit all fallback toggles (Hardware RT off, Mesh Shader off, Frame Gen off).
* **Gate 5 DoD:** Zero frame spikes **> 16.6 ms** during traversal; zero Vulkan validation errors across a 30-minute soak test.
