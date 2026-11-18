# Track B shaders: visibility buffer and virtual shadow pages

These shaders implement the Phase 3 and 4 GPU passes. They compile to Vulkan 1.3 SPIR-V in CI, the compute passes
are tested numerically on lavapipe with validation layers on (`tests/shader_harness/test_track_b_visibility.py`,
`test_track_b_vsp.py`), and the raster pipelines' stage interfaces are checked in `tests/unit/test_raster_interfaces.py`.
**No C++ dispatches them yet**: that needs the Track B decision below.

## Files

Shared headers (`addons/prism_rp/shaders/include/`):

- `visibility_common.glsl`: conventions, buffer contracts (`GpuMeshlet`, `GpuInstance`, `VisibleMeshlet`,
  `VisibilityCullUBO`, indirect commands), the 64-bit payload, transform helpers (`prism_normal_matrix`), triangle facing
  (`vis_triangle_culled`), frustum / cone culling, sphere projection and the conservative Hi-Z test.
- `vis_shading_common.glsl`: `GpuVertexAttributes`, `GpuPbrMaterial`, tile work items, `ShadeDispatch`,
  `VisibilityShadeUBO`, homogeneous barycentrics with screen-space derivatives, octahedral normal encoding.
- `vis_raster_{mesh,vert,frag}_body.glsl`: the raster stages' code, shared by the entry points below.
- `vsp_common.glsl`: clipmap and atlas layout, page-table bits, `VspFrameState`, `VspUniformUBO`, coordinate helpers.

Visibility buffer (`addons/prism_rp/shaders/visibility/`):

| Shader | Role |
| :--- | :--- |
| `vis_hi_z_build.comp.glsl` | Reverse-Z min pyramid with Vulkan mip sizes; odd sizes fold the leftover texel into the last one. |
| `vis_cull_meshlets.comp.glsl` | Two-phase culling: frustum, normal cone, last frame's Hi-Z (phase 1), this frame's (phase 2). Writes opaque and alpha-tested ("masked") lists. |
| `vis_raster.mesh.glsl` + `vis_raster_ms.frag.glsl` | Mesh-shader raster, opaque list. |
| `vis_raster_masked.mesh.glsl` + `vis_raster_ms_masked.frag.glsl` | Mesh-shader raster, masked list (alpha test, no early depth test). |
| `vis_raster.vert.glsl` + `vis_raster.frag.glsl` | Vertex-pulling raster, opaque list. |
| `vis_raster_masked.vert.glsl` + `vis_raster_masked.frag.glsl` | Vertex-pulling raster, masked list. |
| `vis_tile_classify.comp.glsl` | Per 8x8 tile, the material bins present; emits work items and sizes the shade dispatch. |
| `vis_shade_materials.comp.glsl` | One workgroup per (tile, material): PBR sun + emissive, G-buffer, motion vectors. |

Virtual shadow pages (`addons/prism_rp/shaders/shadows/`), in frame order:

| Shader | Role |
| :--- | :--- |
| `vsp_page_request_and_invalidate.comp.glsl` mode 0 | Per virtual page: age, evict, invalidate (scroll, moved AABBs, sun change). |
| `vsp_page_request_and_invalidate.comp.glsl` mode 1 | Per pixel: request the pages the sampler will read, allocate, queue for rendering. |
| `vsp_clear_pages.comp.glsl` | Clear each queued page (indirect). |
| `vsp_cull_meshlets.comp.glsl` | Pair each shadow-casting meshlet with the queued pages it overlaps. |
| `vsp_raster_pages.comp.glsl` | Rasterize each (meshlet, page) pair with `imageAtomicMin`, alpha test included (indirect). |
| `vsp_sample_shadows.comp.glsl` | Contact-hardening PCSS into an `R8` shadow mask. |

## Why nothing runs them yet

Against Godot 4.4.1's `RenderingDevice` (`thirdparty/godot-cpp/gdextension/extension_api.json`):

- There is no mesh or task shader stage: the `*.mesh.glsl` path needs an engine module (or a later Godot). The
  vertex-pulling path is the GDExtension-compatible one.
- `draw_list_draw_indirect` has no count variant. The culler writes one draw per visible meshlet plus a GPU count, so a
  GDExtension host would either draw all `max_visible_meshlets` commands (zeroing the buffer each frame so unused ones
  are empty) or read the count back a frame late.
- `vis_shade_materials`, the masked fragment shaders and `vsp_raster_pages` index a runtime-sized `sampler2D[]` with
  `nonuniformEXT`: the device needs descriptor indexing (`runtimeDescriptorArray`,
  `shaderSampledImageArrayNonUniformIndexing`) enabled.

## Host contracts

Conventions: Godot's depth-corrected projections (reverse-Z, Y flipped for Vulkan, so NDC y = -1 is the top row and
uv = ndc * 0.5 + 0.5); front faces are clockwise on screen; colours are linear Rec.709 throughout (albedo and emissive
textures through sRGB views), which is what Chronos expects.

Struct sizes (std430 unless noted): `GpuMeshlet` 64, `GpuInstance` 160, `GpuVertexAttributes` 48, `GpuPbrMaterial` 80,
`VisibilityCullUBO` 272 (std140), `VisibilityShadeUBO` 256 (std140), `VspClipmapLevel` 96 (std140), `VspUniformUBO` 992
(std140), `VspFrameState` 176. `tests/shader_harness/track_b.py` packs every one of them and is the reference.

Geometry:

- `GpuMeshlet` is byte-identical to `PrismMeshletData`'s records, but its `vertex_offset` / `triangle_offset` index the
  scene-wide buffers: rebase them when packing several surfaces. `GpuInstance.vertex_base` is added to the surface vertex
  indices. Set `prev_world_from_local = world_from_local` on an instance's first frame.
- Set `PRISM_INSTANCE_FLAG_ALPHA_TEST` exactly on instances whose material has `PRISM_MAT_FLAG_ALPHA_TEST`: it routes
  them to the masked list and makes them alpha-tested shadow casters.

Visibility buffer:

- `R32G32_UINT`, cleared to `0xFFFFFFFF` (`PRISM_VIS_PAYLOAD_EMPTY`). Payload: X = instance, Y = meshlet << 8 |
  primitive << 1 | alpha-tested. Barycentrics are rebuilt in shading, not stored.
- Culling: before each phase reset `counters[0]` and `counters[2]` to 0 and both mesh-task commands to `{0, 1, 1}`;
  before phase 1 also `counters[1]`. List L entry i is at `L * max_visible_meshlets + i`. Size `max_visible_meshlets`
  for the worst case: a full list drops meshlets.
- Vertex path: draw each list's `VkDrawIndirectCommand`s non-indexed, no vertex buffers. Pipeline: cull back faces,
  front face CLOCKWISE; mirrored instances (negative determinant) need the opposite front face (a separate pass) or the
  two-sided flag. The mesh path handles mirroring itself.
- Hi-Z: mip L is `max(1, size >> L)`, built from mip L-1 by `vis_hi_z_build`; bind every mip. It's read with
  `texelFetch`, so the sampler's filter doesn't matter. Pass the mip count in `hiz_mip_count`.
- Shading: reset `ShadeDispatch` to `{256, 0, 1, 0}` each frame, run `vis_tile_classify`, then
  `vkCmdDispatchIndirect` `vis_shade_materials` on it.

Virtual shadow pages:

- Page table: `R32_UINT` 2D array, 128 x 128 x 8. Atlas: 4096 x 4096 `R32_UINT` (64 MiB, Gate 4's cap): 1024 pages.
- Initialize once: free stack = 0..1023, `VspFrameState.free_page_count` = 1024. Each frame, before mode 1, reset the
  rest of `VspFrameState`: counts 0, `clear_dispatch` / `raster_dispatch` `{0, 1, 1}`, dirty rects `{~0u, ~0u, 0, 0}`.
  Indirect args: `clear_dispatch` at byte 16, `raster_dispatch` at byte 32.
- Each clipmap level: orthographic `light_view_proj` snapped to the page grid, mapping the level's full width
  (`world_origin_scale.w`) to NDC [-1, 1] and depth to [0, 1] with smaller = nearer the sun; `page_offset_and_pad.xy`
  is the toroidal offset, `.zw` this frame's scroll in pages.
- `sun_dir_and_bias.w` is a depth bias in meters; `pcss_params.x` is tan(sun angular radius); `evict_after_frames` >= 1.

## What the review found, and what changed

| Issue | Fix | Test |
| :--- | :--- | :--- |
| Mesh shader culled Godot's front faces; mirrored instances unhandled | `vis_triangle_culled`: area sign for Godot's projection, mirrored flips it | `test_triangle_cull_keeps_godot_front_faces` |
| Alpha test never applied | Masked list + raster variants without early depth test; alpha-tested shadow casters | `test_clear_cull_raster_page_with_alpha_tested_caster`, `test_raster_interfaces` |
| Mesh-path fragment input wasn't `perprimitiveEXT` (one frag served both paths) | Separate mesh-path fragment shaders | `test_raster_interfaces` |
| Background `uvec2(0, 0)` was a valid payload | Clear to `0xFFFFFFFF`; primitive 127 never occurs | `test_payload_round_trip_and_empty_sentinel`, `test_classify_*` |
| 12-bit meshlet IDs; fragment barycentric extension required | Payload without barycentrics: 32-bit instance, 24-bit meshlet; no extension | `test_payload_*` |
| False-negative overflow dropped meshlets | Drawn in phase 1 instead | `test_cull_false_negative_overflow_draws_instead_of_dropping` |
| Hi-Z not conservative for non-power-of-two sizes | Integer texel mapping matching the pyramid's coverage | `test_hiz_pyramid_covers_every_pixel`, `test_hiz_occlusion_is_conservative` |
| Shade dispatch could exceed `maxComputeWorkGroupCount[0]` | 2D dispatch, 256 wide | `test_classify_bins_materials_and_sizes_the_2d_dispatch` |
| Barycentrics wrong for corners behind the camera | Homogeneous formulation, no `w` clamp; edge-on detected | `test_barycentrics_*` |
| ACEScg vs Rec.709 mix | Linear Rec.709 throughout | (contract) |
| Cone culling unsafe under non-uniform scale; per-pixel `inverse()` | Cone test only for uniform scale; cofactor normal matrix | `test_cull_frustum_cone_and_lists`, `test_normal_matrix_matches_inverse_transpose` |
| Phase 1 near-plane test mixed current view and previous transforms | Clip-space test with the matching matrix | `test_cull_*` |
| Double-sided flip after normal mapping | Flip before, keep the tangent frame (Godot's behaviour) | `test_shade_materials_end_to_end` |
| VSP page clear raced `imageAtomicMin` | Separate clear pass | `test_clear_cull_raster_*` |
| Coarser-level fallback sampled the wrong place | Taps are world positions, reprojected per level | `test_sampler_falls_back_to_the_right_place_in_a_coarser_level` |
| Dirty-queue overflow left stale pages VALID | VALID only once queued | `test_full_render_queue_leaves_pages_invalid_until_queued` |
| Raster cost was every page x every meshlet | `vsp_cull_meshlets` builds (meshlet, page) items inside per-level dirty rects | `test_clear_cull_raster_*`, `test_raster_queue_overflow_invalidates_the_page` |
| 256 MiB atlas; 2k virtual per level; no eviction | 64 MiB atlas, 16k virtual per level, age-based eviction | `test_invalidate_ages_evicts_and_invalidates` |
| Only the centre page requested | Filter footprint + coarsest level requested | `test_request_allocates_and_queues_every_page_shading_reads` |
| PCSS used a perspective-light penumbra in UV units | World-space taps, orthographic penumbra in meters | `test_sampler_*` |
| Mirrored / two-sided receivers got the wrong normal | Determinant sign; two-sided faces the camera | `test_sampler_mirrored_receiver_keeps_its_normal` |

## Still open

- The platform items above (mesh stage, indirect count, descriptor indexing).
- The graphics raster stages are compile- and interface-checked only; the harness runs compute shaders.
- `vis_shade_materials` lighting is a single sun plus a flat ambient term, a placeholder for the real G-buffer design;
  albedo goes to a linear `RGBA8` target, which bands in dark tones (an sRGB format would fix it).
- One directional light. Pages that can't be allocated (1024-page pool) or rendered (queues full) fall back to coarser
  levels until a later frame; tune level extents and `evict_after_frames` against real scenes.
- `vsp_raster_pages` rasterizes each triangle on one thread, so very large casters at fine levels are slow; binning big
  triangles is a later optimization.
- lavapipe returns wrong sizes from `textureSize()` with a lod that differs between invocations; `is_hiz_occluded`
  computes mip sizes instead (noted in the code).

## Bringing a shader online

1. Write the host code (and, for push constants, a C++ struct with a `static_assert(sizeof)`).
2. Move the shader from `UNHOSTED` to `SHADER_TO_STRUCT` in `tests/unit/test_push_constants.py` (the test fails until
   you do).
3. Keep its lavapipe tests passing; build the host-side packing on `tests/shader_harness/track_b.py`'s layouts.
