# Prism RP — Hybrid Visibility & Radiance Pipeline for Godot 4

**Prism RP** is a modular, GPU-driven rendering and temporal video-processing architecture built for Godot's `RenderingDevice` abstraction and Render Graph. It bridges the visual and scalability gap between Godot 4 Forward+ and high-end engines (Unreal Engine 5 Nanite/Lumen/Substrate, Unity HDRP) while remaining lightweight, cross-platform, and MIT-friendly.

## Core Architecture

| Subsystem | Godot 4.6 (Forward+) | Unreal Engine 5.7 | Prism RP |
| :--- | :--- | :--- | :--- |
| **Geometry & Culling** | CPU frustum culling + traditional LOD meshes | **Nanite** virtualized micropolygon clusters | **GPU Meshlet Pipeline** (`VK_EXT_mesh_shader` + compute fallback) writing a 64-bit Visibility Buffer |
| **Material Shading** | Uber-shader permutations in forward pass | **Substrate** multi-lobe layered material graph | **Tile-Sorted Compute Shading** reading a Thin G-Buffer; decoupled from rasterization |
| **Global Illumination** | Lightmapper, coarse VoxelGI, or SDFGI | **Lumen** hybrid screen/SDF/hardware RT GI | **Lumina-Cascades**: hierarchical 6D screen-space + world-space radiance probes with ReSTIR |
| **Shadows** | Cascaded Shadow Maps (CSM) + Omni/Spot atlases | **Virtual Shadow Maps (VSM)** per-pixel pages | **Virtualized Clipmap Shadow Pages (VCSP)** allocated via visibility-buffer page faults |
| **Video & Temporal Post** | FSR 2.2, TAA, basic linear-to-sRGB/AgX tonemapping | TSR, hardware DLSS/FSR3, physical camera post | **Chronos VPU**: unified dilated motion vectors, SVGF denoising, FSR 3.1/XeSS/DLSS + optical-flow frame gen, and ACEScg optics |

---

## Quick-Start Setup

1. **Initialize Git & Submodules** (or run `./setup_submodules.sh`):
   ```bash
   git init
   git submodule add -b 4.4 https://github.com/godotengine/godot-cpp.git thirdparty/godot-cpp
   git submodule add https://github.com/zeux/meshoptimizer.git thirdparty/meshoptimizer
   git submodule update --init --recursive
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

4. **Test in Godot 4**:
   Open `project/project.godot` in Godot 4.4+ (or run headless with Vulkan validation layers enabled):
   ```bash
   VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation godot --headless --rendering-driver vulkan --path project --quit-after 5
   ```

See `ROADMAP.md` for the full 24-week dual-track execution plan, hardware tier budgets, and QA verification gates.
