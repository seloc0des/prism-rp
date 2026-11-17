# Prism RP — Hybrid Visibility & Radiance Pipeline for Godot 4

**Prism RP** is a modular, GPU-driven rendering and temporal video-processing architecture built for Godot's `RenderingDevice` abstraction and Render Graph. It bridges the visual and scalability gap between Godot 4 Forward+ and high-end engines (Unreal Engine 5 Nanite/Lumen/Substrate, Unity HDRP) while remaining lightweight, cross-platform, and MIT-friendly.

## Core Architecture

| Subsystem | Godot 4.6 (Forward+) | Unreal Engine 5.7 | Prism RP |
| :--- | :--- | :--- | :--- |
| **Geometry & Culling** | CPU frustum culling + traditional LOD meshes | **Nanite** virtualized micropolygon clusters | **GPU Meshlet Pipeline** (`VK_EXT_mesh_shader` + compute fallback) writing a 64-bit Visibility Buffer |
| **Material Shading** | Uber-shader permutations in forward pass | **Substrate** multi-lobe layered material graph | **Tile-Sorted Compute Shading** reading a Thin G-Buffer; decoupled from rasterization |
| **Global Illumination** | Lightmapper, coarse VoxelGI, or SDFGI | **Lumen** hybrid screen/SDF/hardware RT GI | **Lumina-Cascades**: hierarchical 6D screen-space + world-space radiance probes with ReSTIR |
| **Shadows** | Cascaded Shadow Maps (CSM) + Omni/Spot atlases | **Virtual Shadow Maps (VSM)** per-pixel pages | **Virtualized Clipmap Shadow Pages (VCSP)** allocated via visibility-buffer page faults |
| **Video & Temporal Post** | FSR 2.2, TAA, basic linear-to-sRGB/AgX tonemapping | TSR, hardware DLSS/FSR3, physical camera post | **Chronos VPU**: unified dilated motion vectors, SVGF denoising, native Compute TAAU / FSR 3.1 / XeSS / DLSS, and ACEScg 2.0 optics |

---

## Quick-Start Setup

1. **Initialize Git & Submodules**:
   ```bash
   chmod +x setup_submodules.sh
   ./setup_submodules.sh
   ```

2. **Validate GLSL 4.60 Shaders to Vulkan 1.3 SPIR-V**:
   ```bash
   scons shaders
   ```

3. **Build the C++20 GDExtension**:
   * **Via SCons**:
     ```bash
     scons platform=linux target=template_debug -j$(nproc)
     ```
   * **Via CMake + Ninja**:
     ```bash
     cmake -B build_cmake -G Ninja -DGODOT_TARGET=template_debug -DPRISM_RP_VALIDATE_SHADERS=ON
     cmake --build build_cmake -j$(nproc)
     ```

4. **Run the Automated Visual & Frame-Budget QA Harness**:
   ```bash
   pip install flip-evaluator numpy pillow
   python3 tests/run_visual_regression.py --tier tier2 --update-golden
   python3 tests/run_visual_regression.py --tier tier2 --max-flip 0.015 --max-pass-delta-ms 0.25
   ```

See `ROADMAP.md` for the full 24-week dual-track execution plan, hardware tier budgets, and QA verification gates.
