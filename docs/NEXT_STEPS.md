# Next steps: from here to the first GPU test

**Where things stand (2026-11-19).** `main` is green, the Godot Vulkan smoke test included. The Track B shaders are
fixed and their compute passes tested on lavapipe. Phases 1–3 are done, and four GPU sessions (Phase 4) have run
on Tier 2 hardware: their findings are under Phase 5, which is in progress. Gates 0 and 1 pass. Gate 2 doesn't yet:
Lumina beats SDFGI in the Cornell box but not in Sponza. Track B waits on a custom Godot build, decided below.

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

**First measurement (CI, lavapipe, 128 samples, emission off):**

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Best possible (reference noise) |
| :--- | :--- | :--- | :--- |
| Cornell box | 13.5 dB (11.5) | 16.6 dB (14.6), 6.6x too dark before the fit | 27.7 dB |
| Sponza | 22.4 dB (23.8) | 27.3 dB (26.0) | 36.1 dB |

Settling: 37% away from the settled result after 6 frames, not settled after 12 (needs under 10% by 6).

So Lumina doesn't pass the revised Gate 2 yet: SDFGI is ahead by 3 dB in the Cornell box and 5 dB in Sponza, and
Lumina settles far too slowly. The Cornell reference is noisy enough at 128 samples that nothing can score above about
28 dB (about 34 dB at 512): the old 36 dB was out of reach there whatever the GI.

**Interim floor: 20 dB fitted** on every Gate 2 scene. Beating SDFGI is the real bar; the floor catches output that
is badly wrong in a scene where SDFGI also does poorly (it sits above SDFGI's Cornell score, below Lumina's Sponza
score and well below the reference's noise ceiling).

**Next for Lumina, in order:** settle within 6 frames (react to lighting changes instead of averaging over many
frames); no dark band where screen-space rays leave the screen; much weaker colour bleeding next to bright,
saturated lights; no bright blobs from individual probes. CI's Gate 2 and settling numbers measure each step.

**Third GPU session (Tier 2, after #8–#10, 512 samples):** every stage ran. Gate 1 5/5, no validation errors
caused by Prism RP, Lumina settles after 1 frame (5% at 6 frames). The release exports load and run Prism RP on
Linux and, through Wine, on Windows (Chronos 0.15 ms, Lumina 0.6–0.7 ms at 1280x720).

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Lumina's mean vs the reference |
| :--- | :--- | :--- | :--- |
| Cornell box | 15.2 dB (13.3) | 17.3 dB (15.6) | 0.50x |
| Sponza | 21.8 dB (23.4) | 30.1 dB (28.8), 2.8x too bright before the fit | 0.38x |

Lumina was too dark, not misplaced: 87% (Cornell) and 96% (Sponza) of its squared error is where it is darker than
the reference, mostly on walls and floor near the camera. Each interval's length in screen pixels was turned into a
world length at the probe's own depth, so a probe could see about as far as it was from the camera (four levels,
8 px spacing, 1080p): the near floor and the walls beside the camera never reached the lit far side of the room.
Intervals are now measured in pixels along the ray's projection on screen, and the top level runs on until the
ray leaves the screen. A lavapipe test (a corridor whose far half is lit) gets 1.07x the analytically integrated
light on the near wall, against 0.007x before.

**Fourth GPU session (Tier 2, after #11, 512 samples):** every stage ran again. Gate 1 5/5, no validation errors
caused by Prism RP, Lumina within 10% of its settled light on the first frame (2% at 6 frames), motion vectors 0.037 px
off, and the release exports run on Linux and on Windows through Wine.

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Lumina's mean vs the reference | Lumina before #11 |
| :--- | :--- | :--- | :--- | :--- |
| Cornell box | 20.05 dB (17.9) | 17.3 dB (15.6) | 0.79x | 15.2 dB, 0.50x |
| Sponza | 25.2 dB (25.3) | 30.1 dB (28.8), 2.8x too bright before the fit | 0.97x | 21.8 dB, 0.38x |

The Cornell box now passes Gate 2's image checks, by 0.05 dB over the floor. Sponza doesn't: SDFGI is 4.9 dB ahead
on the GPU (8.3 dB before #11). On lavapipe the gap is only 2.1 dB because SDFGI scores 2.8 dB lower there, while
Lumina scores the same on both (25.2 dB), so the GPU numbers are the ones to beat.

What changed, and what's left:

- **Near the camera:** measured the same way on both sessions (after the brightness fit), the share of Lumina's
  squared error where it is darker than the reference fell from 85% to 61% (Cornell) and from 90% to 75% (Sponza).
  In the validation screenshots the Cornell box's near floor is now nearly as bright as stock Godot's, and so is
  Sponza's near left wall. Still too dark: surfaces whose light comes from off
  screen or behind the camera, such as the camera-facing side of the Cornell box's small block (about half as bright
  as with SDFGI), the Sponza pillar face along the right edge of the frame and the undersides of Sponza's upper
  gallery. The ray-query fallback is the fix for light that was never on screen.
- **Colour:** 16% of Lumina's Sponza error is in colour rather than brightness (12% before). The red and blue smears
  on Sponza's columns are now its most visible artifact.
- **Blobs:** bright, round probe blobs are back in the Cornell box. Before #11 a few faint ones sat on the ceiling;
  now there is a strong one on the right wall, a fainter one beside it and one on the ceiling.
- **Thin geometry (scene 03):** the louvre slats and upper wire grid are still black with Lumina, so the short reach
  wasn't the cause.
- **Translucency (scene 05):** with Lumina, the red and green translucent panes come out grey; with SDFGI they're
  tinted. Not investigated yet.
- **Cost:** Lumina takes 1.1–1.8 ms in each scene's default view, 1.3–2.0 ms in the full pipeline and at most 2.3 ms
  in a debug view, all under the 2.40 ms budget. Foliage takes 1.67 ms (2.02 ms in the full pipeline). Last session's
  2.40–2.42 ms foliage frames had a running average of 2.20–2.23 ms. The budget check reads a single frame, and in
  these two sessions a single frame sat up to 0.66 ms above or 0.40 ms below the running average. Running the top
  level's rays until they leave the screen added about 0.1 ms to the trace pass.
2. Commit the reviewed reference images so visual regressions can be caught. Optionally add a self-hosted runner on
   a GPU machine, but only for manual runs and pushes to `main`: the repository is public, and a self-hosted runner
   must never run pull requests from forks.
3. Next, in order:
   - thin geometry that goes black with Lumina (scene 03), reproduced in the shader harness first;
   - the colour tint next to bright, saturated surfaces (Sponza's columns) and the Cornell box's probe blobs, and
     whether the integrate pass's firefly clamp (8x the mean of the other directions) throws away real light when
     the rest of the hemisphere is dark;
   - keep Lumina under the 2.40 ms Tier 2 budget in the foliage scene;
   - the custom Godot build (next section), with CI that builds the Linux editor and the Linux and Windows export
     templates;
   - on the custom build, Lumina's real albedo instead of the albedo proxy, and the ray-query fallback for off-screen
     and hidden light, to finish Gate 2;
   - Track B wired up for Gates 3 and 4, then Gate 5.

## Custom Godot build (decided 2026-11-19)

Track B can't run as a GDExtension: it has to replace Godot's opaque pass, and Godot's `RenderingDevice` has no
mesh-shader stage and no indirect-count draws (through 4.8). So:

- Prism RP becomes a Godot engine module, plus a small, separate patch set on Godot 4.7.2 that adds the mesh-shader
  stage and indirect-count draws. Keeping the patches separate keeps them small enough to carry to later Godot
  releases.
- Chronos and Lumina stay a GDExtension that also works in stock Godot.
- The editor is supported on Linux only. Shipped games run on Linux and Windows only, so the build makes export
  templates for those two, with Windows cross-compiled using MinGW. No macOS, iOS, Android or web.
- Every feature checks GPU support at startup and falls back to stock Godot rendering when the GPU lacks it.

## Not in this plan, and why

- **Track B host code:** a GDExtension can't replace Godot's opaque pass, and Godot (up to 4.8 beta) has no mesh
  shaders and no indirect-count draws. It waits on the custom Godot build above.
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
