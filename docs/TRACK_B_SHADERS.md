# Track B draft shaders: visibility buffer and virtual shadow pages

These shaders are drafts for Phases 3 and 4. They compile to Vulkan 1.3 SPIR-V and pass `spirv-val` in CI, and that is all
that has been checked: **no C++ dispatches them, nothing tests their output, and they have known bugs (listed below).**
Treat them as a starting point to review and fix before writing host code, not as working passes.

## Files

Shared headers (`addons/prism_rp/shaders/include/`):

- `visibility_common.glsl`: buffer contracts (`GpuMeshlet`, `GpuInstance`, `VisibleMeshlet`, `FalseNegativeItem`,
  `VisibilityCullUBO`), 64-bit `R32G32_UINT` visibility payload packing, frustum and normal-cone culling, reverse-Z sphere
  projection and Hi-Z test.
- `vis_shading_common.glsl`: `GpuVertexAttributes`, `GpuPbrMaterial`, `MaterialTileWorkItem`, analytic barycentrics and UV
  derivatives (`compute_barycentric_gradients`), octahedral normal encoding.
- `vsp_common.glsl`: 8-level toroidal clipmap (16x16 virtual pages of 128x128 texels per level), page-table bits, the
  8192x8192 physical atlas layout, depth-as-uint helpers.

Visibility buffer and tile shading (`addons/prism_rp/shaders/visibility/`):

- `vis_hi_z_build.comp.glsl`: reverse-Z min-reduction Hi-Z pyramid (copy mode and odd-size-aware downsample).
- `vis_cull_meshlets.comp.glsl`: two-phase meshlet culling (phase 1 against last frame's Hi-Z, queueing rejects; phase 2
  re-tests them against the Hi-Z rebuilt after phase 1).
- `vis_raster.mesh.glsl`: `VK_EXT_mesh_shader` rasterizer, one workgroup per meshlet (64 vertices / 124 triangles).
- `vis_raster.vert.glsl`: vertex-pulling alternative to the mesh shader.
- `vis_raster.frag.glsl`: writes the 64-bit payload, using `GL_EXT_fragment_shader_barycentric`.
- `vis_tile_classify.comp.glsl`: per 8x8 tile, finds the materials present and emits `(tile, material)` work items plus an
  indirect dispatch.
- `vis_shade_materials.comp.glsl`: one workgroup per work item; PBR with a directional sun, writes HDR radiance, albedo +
  roughness, octahedral normal + metallic + AO, and motion vectors (Godot sign, `prev_uv - curr_uv`).

Virtual shadow pages (`addons/prism_rp/shaders/shadows/`):

- `vsp_page_request_and_invalidate.comp.glsl`: mode 0 invalidates pages (clipmap scroll, moving-object AABBs, sun change);
  mode 1 marks the page each screen pixel needs and allocates physical pages.
- `vsp_raster_pages.comp.glsl`: compute rasterizer, one workgroup per dirty page, `imageAtomicMin` into the atlas.
- `vsp_sample_shadows.comp.glsl`: normal-offset bias, blocker search and PCSS filter, writes an `R8` shadow mask.

## Why nothing runs them yet

They need the Track B decision (engine module or GDExtension) first. Against Godot 4.4.1's `RenderingDevice`
(`thirdparty/godot-cpp/gdextension/extension_api.json`):

- There is no mesh or task shader stage, so `vis_raster.mesh.glsl` can't be loaded from a GDExtension.
- There is `draw_list_draw_indirect` but no indirect-count draw. The culler writes one draw command per visible meshlet and a
  GPU-side count, which needs `vkCmdDrawIndexedIndirectCount` or a different layout.
- `vis_raster.frag.glsl` (used by both raster paths) needs fragment-shader barycentrics, and `vis_shade_materials` needs a
  runtime-sized `sampler2D[]` indexed with `nonuniformEXT`. Check that the device has those features enabled before relying
  on them.

## Contracts the host code must honour

Not enforced anywhere yet; write them down next to the C++ structs when they exist.

- `GpuMeshlet` (64 bytes) matches `PrismMeshletData` exactly; triangles are one packed `uint` each (`a | b << 8 | c << 16`),
  `vertices[]` maps meshlet-local to surface vertex index.
- `GpuInstance` is **160 bytes** (two `mat4` + eight `uint`), not the 112 its comment says.
- `VisibilityCullUBO` / `VisibilityShadeUBO` / `VspUniformUBO` are std140; `VspClipmapLevel` is 96 bytes.
- Hi-Z must be sampled with NEAREST filtering (or a min-reduction sampler). The shadow atlas must be sampled NEAREST, and it
  is written through an `r32ui` view of an `R32_SFLOAT` image (mutable format / `texture_create_shared`).
- Counters to reset each frame: cull `counters[0]` before each phase and `counters[1]` before phase 1, the mesh-task
  `groupCountX` before each phase, the classify dispatch `x`, and the VSP dirty-queue count before mode 1.
- Clamp every GPU-written count to its buffer capacity before using it as a dispatch or draw count.
- `vis_raster.vert.glsl` reads `gl_VertexIndex` as `0 .. 3 * triangle_count - 1`, so an indexed draw needs an identity index
  buffer (or switch to `VkDrawIndirectCommand`). `gl_InstanceIndex` is the visible-meshlet slot.

## Known issues

Found in review; none fixed yet. **H** wrong image or data race, **M** wrong in common cases or a hard limit, **L** minor.

Visibility buffer:

- **H** `vis_raster.mesh.glsl` culls `twice_area >= 0`, which under Godot's Y-flipped projection is the *front* face. The
  Hi-Z UV mapping (`visibility_common.glsl`) and pixel reconstruction (`vis_shade_materials`) assume that projection, so
  either the cull or those lookups are wrong for any one convention. Mirrored instances (negative determinant) aren't handled.
- **H** Alpha test is never applied: the flag is packed into the payload but nothing discards, in either raster path or in
  `vsp_raster_pages`. Foliage draws, occludes and shadows as solid cards.
- **M** Background is `uvec2(0, 0)`, which is also a valid payload (instance 0, meshlet 0, primitive 0 at vertex 0). Clear
  to `0xFFFFFFFF` or offset instance IDs.
- **M** When the false-negative queue is full, `vis_cull_meshlets` drops the meshlet instead of drawing it.
- **M** The Hi-Z test isn't conservative for non-power-of-two sizes: floor-sized mips sampled with normalized UVs leave up to
  ~8 px of the footprint uncovered at 1080p. Use a power-of-two pyramid or scale UVs per mip.
- **M** The payload's meshlet field is 12 bits: at most 4096 meshlets (~507k triangles) per surface, unchecked.
- **M** `vis_tile_classify` puts all work in dispatch X, which can exceed `maxComputeWorkGroupCount[0]` (spec minimum 65,535;
  1440p has 57,600 tiles).
- **M** `compute_barycentric_gradients` clamps clip `w` to 1e-6, so triangles crossing the near plane shade wrong.
- **M** Colour spaces: sun and emissive are treated as ACEScg, albedo textures are Rec.709, and Chronos expects linear Rec.709.
- **L** The octahedral world-space normal output does not match Lumina's input (Godot's view-space best-fit RGB normals).
- **L** Phase 1 tests the near plane with the current view but previous-frame transforms; cone culling isn't conservative
  under non-uniform scale; the double-sided flip happens after normal mapping; `inverse(mat3)` runs per pixel / per meshlet.

Virtual shadow pages:

- **H** `vsp_raster_pages` clears the page with `imageStore` and then `barrier()`, which doesn't order image memory; another
  lane's `imageAtomicMin` can be overwritten. Needs `memoryBarrierImage()` before the barrier, or a separate clear.
- **H** `vsp_sample_shadows` falls back to coarser levels with the start level's `clip_uv`; each level needs its own projection.
- **H** On dirty-queue overflow, `vsp_page_request_and_invalidate` has already marked the page VALID, so it keeps stale depth
  indefinitely.
- **M** Each dirty page loops over every meshlet in the scene with three workgroup barriers per meshlet; a sun change at Gate 3
  scale will likely time out. Needs a per-page or per-level cull pass first.
- **M** Sizing: the 8192x8192 R32 atlas is 256 MiB against Gate 4's 64 MB cap; the virtual space is 2048x2048 per level
  (ROADMAP: 16k x 16k); the 4096-page pool is twice the whole virtual space, and pages are never evicted.
- **M** Only the page under each pixel is requested, but PCSS taps reach neighbouring pages, which count as lit.
- **L** PCSS uses the perspective-light penumbra formula for an orthographic sun, and its search radius doubles at each level
  change. Back-facing double-sided receivers get the wrong normal.

## Bringing a shader online

1. Fix its known issues above.
2. Write the host code and a C++ push-constant struct with a `static_assert(sizeof)`.
3. Move it from `UNHOSTED` to `SHADER_TO_STRUCT` in `tests/unit/test_push_constants.py` (the test fails until you do).
4. Add numeric tests to `tests/shader_harness` for the compute passes, as the Chronos and Lumina shaders have.
