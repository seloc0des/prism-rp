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
- **Above an interim floor** of fitted PSNR (below), with a higher target for the ray-query fallback, still to be re-derived (`ROADMAP.md`).
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

**Interim floor: 22 dB fitted** on every Gate 2 scene since Gate 2 measures multi-bounce light (below); 25 dB from #22 and 20 dB before that. Beating SDFGI is the real bar; the floor catches output that
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
  wasn't the cause. Black turns out to be nearly right (below).
- **Translucency (scene 05):** with Lumina, the red and green translucent panes come out grey; with SDFGI they're
  tinted. Not investigated yet.
- **Cost:** Lumina takes 1.1–1.8 ms in each scene's default view, 1.3–2.0 ms in the full pipeline and at most 2.3 ms
  in a debug view, all under the 2.40 ms budget. Foliage takes 1.67 ms (2.02 ms in the full pipeline). Last session's
  2.40–2.42 ms foliage frames had a running average of 2.20–2.23 ms. The budget check reads a single frame, and in
  these two sessions a single frame sat up to 0.66 ms above or 0.40 ms below the running average. Running the top
  level's rays until they leave the screen added about 0.1 ms to the trace pass.

**The Gate 2 reference rendered Godot's colours undecoded (fixed in #13).** Godot treats every authored colour
(albedo, emission, light, ambient and background colours) as sRGB and decodes it to linear before lighting. The scene
export handed Blender the authored numbers, which Blender reads as linear, so the reference's surfaces were brighter
and less saturated than the ones Godot draws: a 0.5 grey reflected 50% of the light instead of 21%, and a saturated
red (0.8, 0.1, 0.1) bounced 8 parts red to 1 green instead of 60 to 1. The ambient and background were too bright in
the same way. Every Gate 2 number above was measured against that reference, SDFGI's as much as Lumina's. The export
now writes colours as authored with their energies kept separate, `godot_to_blender.py` decodes them the way Godot
does, and a reference test renders an authored 0.5 background and ambient and expects 0.214.

Gate 2 against the corrected reference (same build as the fourth session, Tier 2, 512 samples):

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Lumina's mean vs the reference |
| :--- | :--- | :--- | :--- |
| Cornell box | 21.5 dB (19.3) | 18.7 dB (16.9) | 1.07x |
| Sponza | 32.4 dB (32.5) | 34.2 dB (33.6), 6.7x too bright before the fit | 2.36x |

The Cornell box passes Gate 2's image checks, 1.5 dB over the floor and 2.9 dB ahead of SDFGI. In Sponza SDFGI leads
by 1.7 dB, not 4.9. Colour is now a larger share of Lumina's Sponza error (28%, up from 16%), so the smears on the
columns are real and are the next Lumina fix.

**Thin geometry (scene 03) isn't a Lumina bug.** A Cycles render of scene 03 with its real materials (copper and steel,
metallic 1.0, roughness 0.06–0.10) shows the slats and wires nearly black: they're smooth metals reflecting a dark
sky, and the sun lights only their backs. Lumina's wires match that render and its slats are a little too dark, while
SDFGI makes the slats 85 times too bright. What Lumina misses on the slats is a faint copper reflection, because it has
no specular term for metals. That needs each metal's real colour, which stock Godot doesn't give a compositor effect
(`needs_separate_specular` does give it the metallic value), so it belongs with the custom build's real albedo.

**The reference's lights didn't match Godot's either (fixed in #14).** A white floor under one light, rendered by both
Godot 4.7.2 and the reference, showed two differences. Godot multiplies a directional light's energy by pi and divides
its Lambert term by pi, so a sun of energy 1 shows 1.0 on a white surface facing it; the reference's sun showed 1/pi.
And Godot's lamps fade as `max(1 - (d / range)^4, 0)^2 * d^-attenuation`, where Blender's fade with the inverse square: with
Godot's default attenuation of 1 and a long range, a surface 2 m from a lamp gets twice the inverse square's light. The reference's sun is now pi times stronger, and each lamp scales its
emission by the distance to the lit point so it fades exactly as Godot's does (within 0.1% from 1.0 to 1.8 m). Two
reference tests hold both to Godot's numbers. The brightness fit couldn't have hidden this: it removes one overall
factor, and Sponza's sun was off against its lamps.

Gate 2 against the reference with both fixes (same build, Tier 2, 512 samples):

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Lumina's mean vs the reference | Best possible |
| :--- | :--- | :--- | :--- | :--- |
| Cornell box | 21.5 dB (19.5) | 18.9 dB (17.5) | 0.97x | 35.5 dB |
| Sponza | 22.7 dB (22.9) | 24.4 dB (23.9), 2.3x too bright before the fit | 0.81x | 33.8 dB |

Sponza scores lower for both than in the table above because the reference now carries the full sunlight, so it is
about three times brighter and every error weighs more against PSNR's fixed peak. What matters is the comparison:
Lumina's light level is now right in both scenes (the fit scales it by 0.89 and 1.03), and SDFGI is 1.6 dB ahead in
Sponza. Three quarters of Lumina's Sponza error is where it is too dark: the undersides of the upper galleries, the near
left wall and the pillar along the right edge of the frame. It is too bright on the far wall, and 30% of the error is
colour (the smears on the columns).

**Where Sponza's dark areas come from.** All three are surfaces seen at a grazing angle, and the integrate pass did
reject a grazing surface's own probes: it compared depths, which change fast across a probe spacing on such a surface,
and a pixel with no accepted probe got no light (8% of a floor seen from just above it, in a shader test). It now
measures a probe's distance from the pixel's tangent plane instead (#15). That fixes 0.3–0.6% of the pixels in the two
Gate 2 scenes and leaves their scores where they were, so it isn't the main cause. The main cause is light that never
reaches the screen: the sunlit floor that lights the gallery undersides is under the galleries or behind the camera.
The world-space radiance cache is meant to supply it, and it does light those undersides, but it also brightens the
rest of the frame (1.56x the default's irradiance at the median, up to 6x in streaks on the columns), so Sponza scores
21.9 dB with it on against 22.7 dB without. That cache is the next fix.

**The world-space cache leaked light from behind surfaces (fixed in #16).** Two leaks, each with a shader test that
fails without the fix:

- A cache cell whose centre sat less than half a cell behind a visible surface wasn't marked buried. Its rays started
  hidden behind the surface, and their first step read the surface's lit front as a hit, so the cell took that colour
  in all 64 directions and lit its surroundings like a lamp. Any cell behind the visible surface by more than 1% of its
  depth now counts as buried.
- A probe blended the 8 cells around it, including cells on the far side of its own surface (outside the room, inside a
  pillar). Classify can't always bury those, because from the camera they can line up with empty background, so a
  wall could light itself. Cells behind the probe's surface no longer count.

With both, turning the cache on improves Lumina in both Gate 2 scenes (Tier 2 GPU, same reference):

| Scene | Lumina, cache off | Lumina, cache on | SDFGI |
| :--- | :--- | :--- | :--- |
| Cornell box | 21.5 dB, 0.97x the reference's level | 22.1 dB, 1.34x | 18.9 dB |
| Sponza | 22.7 dB, 0.81x | 23.2 dB, 1.22x | 24.4 dB |

It still adds a quarter to a third too much light, and the streaks on Sponza's pillars are still there. One likely cause:
each cache texel always traces the centre of its patch of directions, so a small bright spot can stand in for the
whole patch. Spreading the rays over the patch helped Sponza (23.9 dB) but hurt the Cornell box (18.0 dB, 1.81x),
probably because the cache stores what the camera sees, highlights included, with no clamp against them as the
screen-space path has. The cache stayed off by default until the fixes further down.

**Sponza's extra light was the cache's hit thickness (fixed in #20).** The cache's update counted a ray as hitting a
surface when it was up to 25% of the view depth behind it (2 m at 8 m away), two and a half times the screen-space
trace's 10% (the effect's `thickness` setting). A cache ray passing behind a sunlit pillar half a metre thick therefore
recorded the pillar's lit face. The cache now uses the same thickness as the screen-space trace. With the cache on,
Sponza scores 23.75 dB at 1.05x the reference's light level (SDFGI 24.4 dB), and the Cornell box 22.2 dB, still 1.35x.
The cache costs about 0.07 ms on Tier 2.

The colour smears on Sponza's columns turn out to be the same missing light: the worst column's camera-facing side is
lit mostly from behind the camera (the large red drape and the warm floor), so without the cache it is half as bright
as the reference and greyer, and the drapes in view dominate what is left. What remains for the cache is the Cornell
box, where it adds light evenly to every surface. Coarse directions aren't the cause: doubling the cache's direction
resolution made the Cornell box worse. In an analytic open box run through the whole pipeline in the shader harness,
the cache lands within 8% of the exactly computed answer (0.95x with a bright hotspot on the ceiling, where the
screen-space path alone gets 0.60x), so the excess comes from something particular to the real scene and is still open.
The same test answers the firefly-clamp question: the clamp does throw away real light from a small, bright hotspot
when the rest of the hemisphere is dark (removed; see **Lumina no longer clamps fireflies** below).

**Gate 2 measured SDFGI in a colourless scene (fixed in #22).** Gate 2 took SDFGI's light from Godot's lighting-only
debug view, which swaps every material for Godot's grey default (albedo 0.6, roughness 0.8, metallic 0.2), and does so
in every pass except shadows: also where SDFGI captures the scene's colours. So SDFGI computed its bounce light as if
every wall were grey, with no red or green in the Cornell box and no coloured drapes in Sponza, while the reference had
them. Now every Gate 2 capture renders the reference's scene: matte (`--prism-diffuse-only`), no environment light
(ambient off and a black background, which GI rays that miss would otherwise pick up), and one bounce of GI
(`--prism-gi-single-bounce`: SDFGI's default bounce feedback computes several, while the reference and Lumina compute
one). SDFGI's light is the final image with SDFGI on minus off, Godot's unshaded view gives each surface's albedo, and
Lumina's irradiance and the reference's are multiplied by it, so Gate 2 compares bounce light as a viewer sees it. The
export now carries each material's specular level, so the reference's matte scene is matte too.

Gate 2 on the corrected measurement (Tier 2 GPU, 512 samples; Lumina's world cache off, the default then):

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Lumina's mean vs the reference | Best possible |
| :--- | :--- | :--- | :--- | :--- |
| Cornell box | 28.4 dB (26.9) | 22.6 dB (22.0) | 1.07x | 41.7 dB |
| Sponza | 29.3 dB (29.5) | 34.9 dB (33.3) | 0.77x | 41.2 dB |

Seen fairly, SDFGI is good in Sponza: 5.6 dB ahead of Lumina, not the 0.6–1.6 dB the old measurement suggested, while
Lumina leads the Cornell box by 5.9 dB. SDFGI's light level is right in both (the fit scales it by 1.02 and 0.89). Most of the Sponza gap is light Lumina never sees from screen space (surfaces lit
from behind the camera, the gallery undersides): the world cache recovers some of it today (about 1.5 dB in a first
measurement), and the ray-query fallback on the custom build is what closes it.

Weighting by albedo lowers the error on dark surfaces, so these numbers sit higher than the old ones. The interim floor
moves from 20 to 25 dB by the same rule as before: above SDFGI's Cornell score and below Lumina's Sponza score.

**Lumina no longer clamps fireflies, and small bright patches don't flicker or leave blobs.** The clamp in the
integrate pass cut any direction much brighter than the rest of the probe, so it also cut the real light from a bright
wall when the rest of the room was dark. In an analytic black room lit by one wall, Lumina got 0.50x of the one-bounce
light computed by brute force, and a 4 px hotspot 0.03x. The clamp was there because one ray per direction hit a small
hotspot at full strength or missed it, and the probes that hit it turned into blobs. Every cascade level now traces a
radiance mip chain and samples each hit at the mip as wide as its ray's blur cone. Each texel of the chain also keeps
the nearest depth it covers, and a hit steps to finer mips while its texel takes in something well in front of it;
without that, the floor behind a bright object picked up the object's lit front through the blur. Near silhouettes
that check falls back to sharp mips, where one ray misses the hotspot again, so each direction is now traced by 2 x 2
sub-rays. In the harness the wall gets 0.69x and the hotspot 0.64x, spread evenly over the room, light still doesn't
leak behind objects, and a moving hotspot flickers a little more than before (0.047 relative, from 0.034).

| Lumina, Tier 2 | Cornell box, fitted (centre), level | Sponza, fitted (centre), level |
| :--- | :--- | :--- |
| Before | 28.4 dB (26.9), 1.07x | 29.3 dB (29.5), 0.77x |
| No clamp, mip chain, one ray per direction | 28.1 dB (26.7), 1.02x | 29.2 dB (29.5), 0.80x |
| The same with 2 x 2 sub-rays | 28.9 dB (27.5), 1.01x | 29.3 dB (29.6), 0.79x |

The last two rows share their references (SDFGI scored 22.5 and 34.9 dB in both); the first row's run had slightly
different ones (22.6 and 34.9). The Cornell box's blobs are gone: the dark smudge at the foot of the red wall and the
bright patch under the light. Sponza doesn't change, because its error is light from behind the camera. The price is
frame time: Lumina takes 2.0 to 3.9 ms across the five benchmark scenes on Tier 2, up from 1.1 to 1.8 ms, over the
2.40 ms budget in four of them. That is the chosen order: the best image first, then the cost comes back down.

**The trace is a fifth cheaper and the image holds.** Each cascade level now has its own timer
(`lumina_rc_trace_l0` to `_l3`). Before this change level 0 took about 0.10 ms on Tier 2, level 1 0.19, level 2 0.33
and the top level 0.59. Every level has the same number of texels and the levels below the top the same steps, but
each level's rays cover four times more screen than the one below, and the top level takes twice the steps. Level 0
now traces one ray per direction instead of 2 x 2 (unless it is also the top level), and the default steps per
interval drop from 8 to 6 (12 on the top level). The harness tests all pass. Gate 2 barely moves: 0.02 dB down in the
Cornell box (28.86) and 0.04 dB in Sponza (29.25), less than any other change tried here. Over four runs of the old
settings and three of the new, the trace went from 1.21 to 0.96 ms. Cornell, foliage and thin geometry are now under
the budget, volumetrics sits at 2.45 ms and Sponza at 2.76. Identical runs differ by up to 0.4 ms on a scene, so any
budget call needs several runs.

Some things were tried and left out. One ray on levels 1 or 2 fails the hotspot test (0.48x and 0.51x against its 0.6x
floor). One ray on the top level saves 0.5 ms but costs the Cornell box 0.5 dB and makes the thin-geometry scene's
wire mesh sparkle. Cutting the top level to 8 steps instead of 12 saves another 0.2 ms for 0.15 dB, and it darkens
Sponza, which already reads too dark. The harness's 128 x 96 scenes can't judge any of the top-level changes: its
interval starts 168 px from the probe, past their edge. Run at 512 x 384, the shipping trace gives a 4 px hotspot
0.55x of its light and lets 0.57x of the card's light through behind it at thickness 0.2, both outside the limits the
small scenes pass.

**The world cache is on by default, and it still adds too much light to a small open room.** Turned on after the
trace changes above, it brought the Cornell box to 1.52x the reference's light level (27.18 dB) and Sponza to 0.97x
(30.82 dB), against 1.00x and 0.77x with it off. More than half of the Cornell box's extra came through directions
that leave the box by its open front, where there is nothing to see. The cache was missing two of the screen-space
trace's rules. It read each hit at a single pixel, so the one direction that pointed at a bright spot held the spot's
peak across its whole cone. It now samples the radiance mip chain at its cone's width. It also recorded the lit front
of any surface it passed just behind, which the trace counts as blocked. It now checks that the surface faces the ray.
Each fix has a harness test that fails without it. Together they bring the Cornell box to 28.96 dB at 1.38x, 0.1 dB above
the cache off, and Sponza to 30.45 dB at 0.92x, 1.2 dB above it. The cache's two passes cost 0.04 to 0.18 ms on Tier 2.

What's left is built into how the cache learns. It only sees the screen, so a ray that leaves the screen can't tell
empty space from something out of view, and the merge fills that direction from whichever neighbouring cells did see
something. That filling is what lights Sponza's camera-facing surfaces. In a box open toward the camera it is also
what adds the extra light. Weighting it by how many cells saw the direction brought the Cornell box to 1.21x but gave
back most of Sponza's gain (29.82 dB). The ray-query fallback can tell the two cases apart.

**On the custom build, the world cache traces the scene mirror.** Each of the cache's 16^3 x 64 directions traces a
ray through Lumina's scene mirror every frame (`lumina_ws_update_rt`), so no direction is ever unknown. A hit the
camera sees takes its radiance from the screen, as the march did. Any other hit is lit from the mirror's copy of the
scene's lights, with a shadow ray for each light that casts shadows, plus its emission. The mirror reads lights the
way Godot's GI does: bake mode Disabled leaves a light out, indirect energy scales it, and a directional light set to
Sky Only lights nothing. A one-sided surface seen from behind blocks the ray, so a probe inside a column sees nothing
instead of the room beyond it. Every hit also adds its albedo times the cache's own irradiance there, last frame's,
which a new pass averages from each cell's directions (`lumina_ws_irradiance`). That is a second bounce, then a
third, and it is what lights surfaces that only other surfaces light. Without ray queries the cache marches as before.

For a single-bounce comparison, `run_gate2.py --diffuse-bounces 1` holds Lumina to one bounce as well
(`world_multi_bounce_enabled`, turned off by the rig's `--prism-gi-single-bounce`), as it already did SDFGI. Each
outcome has a shader harness test on lavapipe, for which the harness learned to build acceleration structures,
and `tests/engine/run_world_cache_rt.sh` checks the whole pipeline: a light behind the camera lights the view, stops
when its bake mode is Disabled, and the wall's emission takes over; light that bounced twice off screen reaches the
view, and goes when `world_multi_bounce_enabled` is off. Not read yet: physical light units (the mirror warns),
AreaLight3D, a light's distance fade and cull mask, and the camera's exposure normalization, which Godot applies to
the on-screen light the cache takes but the mirror doesn't.

Single-bounce Gate 2 on the RTX 4060 Ti, the measure up to this change (512 samples, one session; fitted PSNR,
centre in brackets, then the mean light level against the reference):

| Scene | Cache marched | Cache traced | SDFGI |
| :--- | :--- | :--- | :--- |
| Cornell box | 28.97 dB (28.80), 1.38x | 29.25 dB (27.98), 1.11x | 22.46 dB (21.97), 0.78x |
| Sponza | 30.45 dB (30.60), 0.92x | 33.40 dB (31.71), 1.14x | 34.87 dB (33.36), 1.03x |

Sponza gains 2.9 dB and its gap to SDFGI drops from 4.4 to 1.5 dB, so Gate 2 still doesn't pass there. The Cornell
box goes from 1.38x to 1.11x the reference's light, but its fitted score barely moves and its centre loses 0.8 dB:
what's left there isn't the open front any more. Both scenes now come out brighter than the reference, by 11 and 14
percent. The compare images show no leaks or dark patches near the columns.

**Gate 2 now measures multi-bounce light (decided 2026-11-20).** Real light keeps bouncing, Cycles renders four
diffuse bounces by default, and both Lumina's traced cache and SDFGI bounce more than once as shipped, so the gate
now scores what a player sees: the reference renders four bounces, and Lumina and SDFGI keep their own.
`run_gate2.py` does this by default.

| Scene | Lumina, fitted (centre) | SDFGI, fitted (centre) | Mean light, Lumina / SDFGI | Best possible |
| :--- | :--- | :--- | :--- | :--- |
| Cornell box | 24.24 dB (22.81) | 19.77 dB (19.04) | 0.76x / 0.61x | 35.7 dB |
| Sponza | 32.52 dB (30.42) | 33.61 dB (31.97) | 0.93x / 0.89x | 40.1 dB |

Lumina leads the Cornell box by 4.5 dB and trails in Sponza by 1.1 dB, so Gate 2 still fails there. The interim
floor moves to 22 dB by the old rule, above SDFGI's Cornell score and below Lumina's Sponza score, about as far
above SDFGI as before. Scored this way, Lumina is too dark, most of all in the closed Cornell box: only the world
cache bounces more than once, while the screen-space levels near each pixel still see a single bounce. Bouncing
the screen-space levels' light too is the next step for the image.

The traced update and the mirror's GPU work cost 0.16 to 0.45 ms a frame on the 4060 Ti, against 0.02 to 0.12 ms
for the march. This session's whole-frame timings swung by 0.6 ms between runs and put even the marched cache over
2.40 ms in four scenes, so the budget needs the several-run timing that settled it in #33. The mirror also gathers
the scene on the main thread every frame now that the cache uses it.

**The floor behind an object took the light of its lit front (fixed after #36).** The harness's card test (a bright
red card standing on a grey floor) passed at 96 x 64 and leaked at the other sizes tried: 7.3x the exact light on the
floor behind the card at 128 x 96 and 3.5x at 256 x 192. Neither the integrate pass nor the trace's thickness test
was the cause. On a floor seen at a grazing angle one probe spacing on screen is most of a metre, so at 128 x 96 the
level-1 probes sat 0.1 m in front of the card and saw its lit face, while the level-0 probes they served were 0.6 m
behind it. The merge's depth weights rated every parent there a poor match (0.19 at best). Then it normalised the
least poor one to full weight. Now a probe that no parent can stand in for (bilinear times depth weight under 0.3)
traces on to the edge of the screen by itself, with twice its level's steps and 4 x 4 sub-rays per direction, and takes
no parent light at all. The floor behind the card gets 0.7–1.2x the exact light at every size from 96 x 64 to
512 x 384, and a harness test holds 128 x 96 and 256 x 192 to 0.5–1.5x of it. To confirm on the GPU:

- **Cost:** in the analytic scenes at 1080p, under 1% of level-0 probes and up to 4% of an upper level's take this
  path, but each traces 16 rays across the whole screen where it used to trace one ray over its interval. Time the
  trace passes over several runs.
- **Gate 2:** those probes also stop receiving the world cache, which reaches the screen levels through the merge, so
  silhouettes and grazing floors may come out a little darker. Check both scenes' scores, and look at the compare
  images around Sponza's pillars and the Cornell box's blocks.

**The screen-space levels now bounce light more than once.** Godot's colour buffer holds direct light only, so
every screen level traced a single bounce while Gate 2 scores four. The radiance copy now adds last frame's
denoised GI times the albedo proxy (and the effect's intensity) to every pixel it can reproject onto the same
surface, the light the viewer saw that pixel reflect, so each frame adds a bounce. In a harness room with grey
walls of albedo 0.5, Lumina gets 0.78x of the exact light with every bounce counted (`irradiance_truth` now
path-traces later bounces), against 0.49x before, and it settles by the fourth frame. The proxy reads a saturated
colour as albedo 1 in its channel, and in a closed red box that ran up to 2.8x the walls' own light, so the
bounce caps it at 0.8 (1.65x). `screen_multi_bounce_enabled` turns it off; the rig's `--prism-gi-single-bounce`
does, for the single-bounce comparison. To confirm on the GPU:

- **Gate 2:** both scenes' scores and light levels, the Cornell box most of all (it was 0.76x the reference's
  light, too dark for want of bounces). The albedo proxy's errors now count once per bounce.
- **Settling:** `tests/check_gi_convergence.py` after a light switches on. Each bounce arrives a frame later,
  through SVGF's history as well, so it may take longer to settle than the 1 frame measured before.
- **Cost:** four more texture reads per pixel in the radiance copy, likely too small to see.

2. Commit the reviewed reference images so visual regressions can be caught. Optionally add a self-hosted runner on
   a GPU machine, but only for manual runs and pushes to `main`: the repository is public, and a self-hosted runner
   must never run pull requests from forks.
3. Next, in order:
   - bring Sponza and volumetrics under the 2.40 ms Tier 2 budget without losing the image: the a-trous denoiser
     (0.35 to 0.75 ms) and the top cascade level (about 0.5 ms) are the biggest passes left; time several runs per
     change. The top level now traces one ray per direction instead of 2 x 2, a quarter of its rays: in the
     harness at 512 x 384 and 1024 x 768 it lights the rooms within 3% of before and flickers no more (two tests
     hold it). It also takes as many steps as the other levels now, not twice as many: in a room whose back wall
     shines through a row of thin pillars, twice the steps let more light through the pillars (1.23x the exact
     share against 1.06x, tested). Together that is an eighth of the top level's old trace. To time on Tier 2,
     and to check Gate 2 for any change.

     One limit found on the way, not fixed: pillars that hide behind one another from the camera (looking down
     a colonnade) are missing from the depth buffer, so light crosses them. In the harness, the wall beside such
     a colonnade gets 1.5x its exact light (a thickness of 0.3 instead of 0.1 brings it to 1.0x, at the cost of
     blocking light everywhere else). Screen space can't see them; the ray-traced world cache can, so it may
     explain part of what's left of the smears on Sponza's columns;
   - harness scenes large enough to reach the top cascade level, so its changes can be tested off the GPU: one
     so far (a black room at 512 x 384 whose left wall gets its light mostly through level 3; it fails if the top
     level stops halfway to the screen edge, which every 128 x 96 test misses), and more as the top level changes;
   - the custom Godot build (next section), with CI that builds the Linux editor and the Linux and Windows export
     templates;
   - on the custom build, what is left of Sponza's 1.1 dB gap to SDFGI after the ray-traced cache (above), once the
     screen levels' new bounces are measured: skinned, blend-shape and MultiMesh geometry in the scene mirror, Lumina's
     real albedo instead of the albedo proxy, and a specular term so metals reflect bounced light, to finish Gate 2;
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

State on 2026-11-20:

- Engine patches (`engine/patches/`): 0001 indirect-count draws, 0002 mesh and task shader stages, 0003 indirect mesh
  draws, 0004 the scene-renderer hook and the Track B capability check, 0005 nonuniform sampled-image indexing,
  0006 ray queries (VK_KHR_ray_query, TLAS instances without a hit shader binding table range, and a fix for a
  second build of a TLAS in one frame overwriting the first build's instances).
- Lumina's scene mirror (`LuminaSceneMirror`, in the GDExtension) keeps the main viewport's MeshInstance3D meshes in a
  TLAS, one BLAS per mesh, with each material's albedo colour and texture, UV transform, vertex colours, alpha
  scissor and cull mode. The Ray-Traced Albedo debug view sends one camera ray per pixel through it, and
  `tests/engine/run_scene_mirror.sh` checks that view against Godot's unshaded one, before and after the scene
  changes and after the mirror releases and starts again. It also holds the lights that light GI (directional, omni
  and spot) and each material's emission and cull mode, and Lumina's world cache traces it (above). Not mirrored
  yet: skinned and blend-shape meshes, MultiMesh, CSG, particles, ImmediateMesh and other Mesh types that are neither
  ArrayMesh nor PrimitiveMesh, AreaLight3D, and lights in physical units. The BLASes aren't indexed, because stock
  4.7.2's `blas_create` checks an index buffer's largest index, which it records only in debug builds and only for a
  buffer created with its data, so every indexed BLAS fails in a release export.
- The module (`engine/modules/prism_rp`) builds the meshlet builder from `addons/prism_rp/src/meshlet` on Godot's own
  meshoptimizer 1.1.1, and embeds the Track B shaders from `addons/prism_rp/shaders` (includes inlined by
  `embed_shaders.py`). Nothing reads either yet; the scene-renderer factory still returns the stock renderer.
- CI (`.github/workflows/engine.yml`) builds the Linux editor and four templates, and runs the lavapipe tests for each
  patch, for the material shading dispatch and for the benchmark scenes.
- Releases: pushing a tag `engine-v...` runs `.github/workflows/engine-release.yml`, which builds the editor and the
  templates from scratch (no SCons cache) and attaches `prism-godot-<release>-linux-editor.tar.gz` and
  `prism-godot-<release>-templates.tpz`. `docs/USING.md` section 4 tells users how to install them. No release has
  been cut yet, and the workflow has not run on GitHub.

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
