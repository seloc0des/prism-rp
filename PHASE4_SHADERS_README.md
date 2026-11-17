# Prism RP — Phase 4 Visibility Buffer & Virtual Shadow Pages Shaders

This package contains all 10 GLSL 4.60 / Vulkan 1.3 shader files generated for Phase 4, structured to merge directly into the root of your `prism-rp` repository:

## Included Files

### Shared Shader Headers (`addons/prism_rp/shaders/include/`)
- `visibility_common.glsl`: `std430`/`std140` buffer contracts (`GpuMeshlet`, `GpuInstance`, `VisibleMeshlet`, `FalseNegativeItem`, `VisibilityCullUBO`), 64-bit `R32G32_UINT` visibility payload packing/unpacking, frustum/normal-cone culling, and Reverse-Z Hi-Z sphere projection.
- `vis_shading_common.glsl`: `GpuVertexAttributes`, `GpuPbrMaterial`, `MaterialTileWorkItem`, analytic screen-space barycentric/UV gradient reconstruction (`compute_barycentric_gradients`), and octahedral normal encoding.
- `vsp_common.glsl`: 8-level toroidal Virtual Shadow Pages clipmap structures (`16x16` virtual pages per level, `128x128` physical pages in an `8192x8192` atlas), Page Table Entry bitflags, and ordered float-to-uint depth helpers.

### Visibility Buffer & 8x8 Tile Shading (`addons/prism_rp/shaders/visibility/`)
- `vis_hi_z_build.comp.glsl`: Reverse-Z `R32_SFLOAT` Hi-Z mip pyramid generator (`MODE_COPY_DEPTH` and odd-dimension-safe `MODE_DOWNSAMPLE_MIN`).
- `vis_cull_meshlets.comp.glsl`: Two-phase GPU meshlet culler (`PHASE1_MAIN` with previous-frame Hi-Z + false-negative queueing; `PHASE2_DISOCCLUSION` with post-Phase-1 Hi-Z).
- `vis_raster.mesh.glsl`: Hardware `VK_EXT_mesh_shader` Visibility Buffer rasterizer (`64` vertices / `124` primitives per workgroup).
- `vis_raster.vert.glsl`: Programmable vertex-pulling fallback shader for platforms without `VK_EXT_mesh_shader`.
- `vis_raster.frag.glsl`: Early-Z `R32G32_UINT` payload fragment writer using `GL_EXT_fragment_shader_barycentric`.
- `vis_tile_classify.comp.glsl`: `8x8` tile material deduplication pass emitting compact `(tile_coord, material_id)` work items and `VkDispatchIndirectCommand`.
- `vis_shade_materials.comp.glsl`: Indirect per-material PBR compute shader outputting HDR radiance, albedo/roughness, octahedral normals/metallic/AO, and Godot-sign motion vectors.

### Virtual Shadow Pages (`addons/prism_rp/shaders/shadows/`)
- `vsp_page_request_and_invalidate.comp.glsl`: Toroidal scroll / dynamic AABB / sun-angle invalidation pass (`Mode 0`) and screen-space depth page request & physical pool allocator (`Mode 1`).
- `vsp_raster_pages.comp.glsl`: Compute meshlet shadow rasterizer writing ordered depth via `imageAtomicMin` into `128x128` physical atlas tiles.
- `vsp_sample_shadows.comp.glsl`: Geometric-normal-biased Vogel-disk blocker search + contact-hardening PCSS sampler outputting `R8_UNORM` directional shadow visibility.

## Merging Into Your Repository
1. Extract this `.zip` directly into the root of your `prism-rp` repository (`addons/prism_rp/shaders/...` aligns 1:1 with your existing directory tree).
2. Remove the obsolete `addons/prism_rp/shaders/visibility/.gitkeep` placeholder if desired.
3. Ensure `CMakeLists.txt` and `SConstruct` recursively glob `addons/prism_rp/shaders/**/*.glsl` so `shaders/shadows/*.comp.glsl` compiles to `.spv` alongside `shaders/visibility/`.
