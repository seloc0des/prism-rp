# Next steps: from here to the first GPU test

**Where things stand (2026-11-18).** `main` is green, the Godot Vulkan smoke test included. The Track B shaders are
fixed and their compute passes tested on lavapipe. Chronos and Lumina have still never run on a real GPU, which is
the gap this plan closes. Phases 1–3 are done: what remains is the GPU session (Phase 4) and its follow-up.

**Target engine: Godot 4.7.2**, the current stable release. The reasons are in "Godot 4.5 → 4.8" below.

How it runs:

- Phases 1–3 happen remotely: in the cloud, plus CI on lavapipe (Mesa's CPU Vulkan driver).
- Phase 4 is one sitting at a GPU machine.
- Work lands through pull requests whose CI must be green before merging. Everything goes through one working
  branch, so consecutive phases sometimes share a PR.

## Phase 1: move to Godot 4.7 (remote) — done (#3)

1. Move the `thirdparty/godot-cpp` submodule from the 4.4 branch to `10.0.0-stable`. Build against
   `api_version=4.7` in SCons and `GODOTCPP_API_VERSION=4.7` in CMake.
2. Fix whatever v10 breaks, keeping zero warnings in our code.
3. Set `compatibility_minimum = "4.7"` in `prism_rp.gdextension`. Move the CI and release workflows, README and QA
   scripts to Godot 4.7.2.
4. Point the tests that read Godot's API dump (the GDScript API-name check) at the 4.7 file.

**Done when:** every CI job is green on Godot 4.7.2, the Vulkan smoke test shows no new validation findings against
its baseline, and the lavapipe suites pass.

**Risks:** the size of the v10 migration is unknown until it builds. Godot 4.6 reportedly changed motion vectors,
which Chronos depends on; Phase 3 adds a lavapipe check and Phase 4 confirms it on a GPU.

## Phase 2: Chronos for 4.7, HDR-aware output (remote) — done (#3)

Godot 4.7 can drive HDR displays, but Chronos's SDR look clamps to [0, 1] and would throw HDR away.

1. Each frame, the main thread reads whether the window outputs HDR and its `get_output_max_linear_value()`. The
   render thread uses that copy, the same way Chronos already handles its other settings.
2. With HDR output on, the SDR look maps highlights up to that maximum instead of 1.0. With it off, output stays
   bit-identical to today's.
3. Retire the placeholder `output_display_mode` values (Display P3, HDR10) in favour of Godot's own HDR output. Old
   scenes get a warning and a mapping, and the README's HDR section is rewritten.
4. Add lavapipe tests of the tone mapping at maximum linear values 1.0 (unchanged) and 4.0 (midtones unchanged,
   highlights extended).

**Done when:** the new and existing Chronos tests pass and CI is green.

## Phase 3: a one-command GPU session (remote) — done (#4)

1. **`tests/gpu_session.py`.** It:
   - checks for a Vulkan GPU and Godot 4.7.2, downloading Godot if it's missing;
   - builds the extension, links (or copies) the addon and imports the project;
   - runs `run_gpu_validation.py` (every scene and configuration, plus Gate 1), and Gate 2 if Blender is installed;
   - packs report, images and logs into one zip and prints its path.

   It works on Linux and Windows.
2. **New validation configurations:**
   - alpha-hashed foliage in scene 02 (Godot's alpha-hash transparency resolved by Chronos);
   - translucency ghosting in scene 05 (opaque capture feeding the reactive mask);
   - an HDR on/off pair, run when the display supports HDR.
3. **Motion-vector check on 4.7:** a CI capture on lavapipe of a known camera move, checking the sign and size of
   Godot's motion vectors as Chronos reads them. This catches a convention change before the GPU session.
4. **A CI job running `gpu_session.py --software` on lavapipe,** so the session script itself is tested on every PR
   and doesn't fail on your machine.
5. **`docs/GPU_TEST_PLAN.md` refreshed:** `main` instead of the old branch, Godot 4.7.2, the new checks, and how
   to run it.

**Done when:** CI runs the whole session end to end on lavapipe and uploads the zip.

## Phase 4: the GPU session (about an hour at a GPU machine)

On the GPU machine: `git pull`, then `python3 tests/gpu_session.py --tier <tier1|tier2|tier3>`. It writes one zip with
the report, images and logs (details in `GPU_TEST_PLAN.md`).

What gets judged (from `ROADMAP.md`):

| Check | Pass |
| :--- | :--- |
| Vulkan validation | No errors that Prism RP introduces |
| Gate 1 (Chronos) | FLIP < 0.08 at 50% render scale against the supersampled reference; Chronos < 1.65 ms on Tier 2 |
| Gate 2 (Lumina) | PSNR > 36 dB against the Cycles reference (needs Blender); GI + SVGF < 2.40 ms on Tier 2 |
| By eye (`GPU_TEST_PLAN.md`) | No particle ghosting, foliage shimmer under control, motion vectors right, HDR highlights correct on an HDR display |

**Optional extras:** an HDR display for the HDR check (on Linux it also needs a Wayland session), and Blender 4.x or
5.x for Gate 2.

## Phase 5: after the session (remote)

1. Fix what the session finds. Where possible, each finding becomes a failing test first, then a fix.
2. Commit the reviewed reference images so visual regressions can be caught. Optionally add a self-hosted runner on
   a GPU machine, but only for manual runs and pushes to `main`: the repository is public, and a self-hosted runner
   must never run pull requests from forks.
3. Next plan: Lumina's ray-query fallback. Godot 4.7 exposes ray queries, and lavapipe supports them, so it can be
   built and tested remotely. After that comes the Track B decision.

## Not in this plan, and why

- **Track B host code:** a GDExtension can't replace Godot's opaque pass, and Godot (up to 4.8 beta) has no mesh
  shaders and no indirect-count draws. It needs the engine-module decision first.
- **Cluster LOD, software rasterizer, FSR / XeSS / DLSS backends:** cluster LOD is useful once Track B renders
  meshlets. The software rasterizer is blocked on stock Godot, which has no 64-bit image atomics. The compositor has
  no hook after upscaling, so the backends can't upscale.

## Godot 4.5 → 4.8: what changed for Prism RP

Diffed from the API dumps `godot-cpp` ships for each version.

| Version | Relevant changes |
| :--- | :--- |
| 4.5 | 32-bit image atomics feature flag, MetalFX upscalers, SMAA |
| 4.6 | Dynamic-offset uniform / storage buffers, depth-resolve attachments; reported motion-vector changes |
| 4.7 | Ray tracing (BLAS/TLAS, ray-tracing pipelines, `SUPPORTS_RAY_QUERY`, acceleration-structure uniforms); HDR output (`Window.hdr_output_requested`, `get_output_max_linear_value()`, per-window luminance) |
| 4.8 beta | Godot meshes' vertex and index buffers exposed as raw GPU buffers, optionally as storage buffers |

Unchanged from 4.4 through 4.8: the compositor API (five callback points, the last before Godot's TAA, upscaling and
tonemapping), and the absence of mesh / task shader stages and indirect-count draws.
