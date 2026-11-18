# Next steps: from here to the first GPU test

**Where things stand (2026-11-18).** `main` is green, the Godot Vulkan smoke test included. The Track B shaders are
fixed and their compute passes tested on lavapipe. Phases 1–3 are done, and the first GPU session (Phase 4) ran
on Tier 2 hardware: its findings are under Phase 5, which is in progress.

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

## Phase 4: the GPU session (about an hour at a GPU machine) — first run done

First run (Linux, Tier 2, X11):

- **Vulkan validation:** nothing from Prism RP. Every message also occurred in the `baseline_godot` runs.
- **Motion vectors:** 0.037 px median error for about 11 px of motion (0.035 px with jitter).
- **Timings, all within budget:** Chronos 0.25–0.40 ms (0.79 ms with alpha-hashed foliage), optics about 0.04 ms,
  Lumina 0.5–1.7 ms (2.1 ms in one debug view).
- **Gate 1:** 4 of 5 scenes pass (mean FLIP 0.019–0.047). Scene 05 failed at 0.158 because its sparks were unseeded
  and differed between the reference and test captures.
- **Not run:** Gate 2 (the installed Blender was 3.0, too old) and HDR (an X11 session).

On the GPU machine: `git pull`, then `python3 tests/gpu_session.py --tier <tier1|tier2|tier3>`. It writes one zip with
the report, images and logs (details in `GPU_TEST_PLAN.md`).

What gets judged (from `ROADMAP.md`):

| Check | Pass |
| :--- | :--- |
| Vulkan validation | No errors that Prism RP introduces |
| Gate 1 (Chronos) | FLIP < 0.08 at 50% render scale against the supersampled reference; Chronos < 1.65 ms on Tier 2 |
| Gate 2 (Lumina) | As revised below and in `ROADMAP.md`: beats SDFGI against the Cycles reference (needs Blender), above the interim floor, settles within 6 frames, GI + SVGF < 2.40 ms on Tier 2 |
| By eye (`GPU_TEST_PLAN.md`) | No particle ghosting, foliage shimmer under control, motion vectors right, HDR highlights correct on an HDR display |

**Optional extras:** an HDR display for the HDR check (on Linux it also needs a Wayland session), and Blender 4.x or
5.x for Gate 2.

## Phase 5: after the session (remote)

1. Fix what the session finds. Where possible, each finding becomes a failing test first, then a fix. From the first
   run:
   - **Bloom streaks:** Chronos' FFT bloom drew starburst spikes across the whole frame, wrapping around its edges.
     The spikes are now finite strips in frequency space (lavapipe tests for locality, positivity and visible spikes).
   - **Scene 05:** the sparks use a fixed seed, so Gate 1 compares the same particles.
   - **Lumina is diffuse only:** with SDFGI off, metals reflected nothing. The rig turns on screen-space reflections
     when Lumina runs (not for Gate 2), and the README says so.
   - **Session script:** an old Blender gets a clear skip instead of a traceback, and validation counts only errors
     that stock Godot doesn't also produce.

   The second run (Blender 5.3) confirmed all of these and scored Gate 2 for the first time: 14.2 dB fitted in the
   Cornell box and 21.4 dB in Sponza, against 36.

### Gate 2, revised

36 dB of PSNR against a path tracer means an average error of about 1.6% of full brightness everywhere in the frame.
Screen-space GI can't get there: it never sees light from surfaces off screen or hidden behind others, and it is
blurrier than per-pixel path tracing. What this project needs from Lumina is to look clearly better than stock
Godot's GI, without artifacts and within budget, so Gate 2 now asks for that (`ROADMAP.md`):

- **Beat SDFGI** against the same Cycles reference, measured the same way, on every Gate 2 scene.
- **Above an interim floor** of fitted PSNR (below), with 36 dB kept as the target for the ray-query fallback.
- **No visible artifacts** in the test plan's Lumina checks.
- **Settle within 6 frames** after a light switches on (`tests/check_gi_convergence.py`).
- **GI + SVGF under 2.40 ms** on Tier 2 (0.5–1.7 ms on the first run).

Gate 2 compares the bounce light of the scene's lights with emission off: the Cornell box's ceiling panel is drawn next
to the point light that actually lights it, and counting its light as Lumina's would light the box twice. CI measures
all of this on lavapipe (report only) and renders the reference twice to bound the best possible score.

What the first fair comparison shows Lumina needs, in order: no dark band where screen-space rays leave the screen
(edges and the floor near the camera), much weaker colour bleeding next to bright, saturated lights (red and blue
smears on Sponza's columns), and no bright blobs from individual probes.

**Interim floor:** set from this PR's CI measurement.
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
