# GPU Test Plan

Everything in Prism RP so far was built and tested without a GPU. The shaders validate, the compute passes are
checked numerically on lavapipe (Mesa's CPU Vulkan driver), and CI boots every scene on a software rasterizer. None
of that says whether the images look right or run fast enough. This session on real hardware does.

It's one command. Budget about an hour: 20–40 minutes of runtime (more with Gate 2), the rest looking at images.

## Before you start (once)

You need Linux (Ubuntu-based distributions are what the commands below assume) or Windows, a Vulkan 1.3 GPU,
Python 3.11+, and:

```bash
git clone --recurse-submodules https://github.com/seloc0des/prism-rp.git
```

```bash
cd prism-rp && python3 -m pip install scons numpy pillow flip-evaluator OpenEXR
```

If you install these into a virtual environment at `prism-rp/.venv`, the session script uses it even when the
shell hasn't activated it.

On Linux, also install `build-essential vulkan-tools vulkan-validationlayers`. With the validation layers present,
the run reports Vulkan errors. Ubuntu 22.04's layers (also Pop!_OS 22.04's) predate Godot 4.7 and flag
`VUID-VkDeviceCreateInfo-pNext-pNext` in every run, plain Godot included; the report's comparison with the
`baseline_godot` runs separates that from errors Prism RP causes.

Optional:

- **Gate 2:** Blender 4.x or 5.x on `PATH`, or passed with `--blender-bin`. Ubuntu 22.04's `blender` package is 3.0,
  which is too old (the session skips Gate 2 and says so); the Linux `.tar.xz` from blender.org runs unpacked.
- **HDR check:** an HDR monitor with HDR turned on in the OS. On Linux it also needs a Wayland session:
  `echo $XDG_SESSION_TYPE` should print `wayland`.
- **Exported Windows game:** Wine (`sudo apt install wine`) and MinGW-w64 (`sudo apt install
  g++-mingw-w64-x86-64-posix`) let the export stage build and run the Windows export on Linux.
  Without them it checks the Linux export only. The first run downloads Godot's export templates
  (about 1 GB) unless the editor already has them.

You don't need Godot installed: the script downloads Godot 4.7.2 into `tests/artifacts/godot/`.

## Run it

```bash
git pull && git submodule update --init --recursive
python3 tests/gpu_session.py --tier tier2
```

Pick the tier that matches the GPU (`ROADMAP.md`; RTX 4060 Ti / RX 6700 class is `tier2`). When it finishes, the script
prints a summary and the path of the zip holding the results: `tests/artifacts/gpu_session/gpu_session_<timestamp>.zip`,
with screenshots as JPEG. The HDR stage asks two questions, so stay at the screen for the end of the run.

The stages run in this order: system, python, godot, build, project, motion, validation, settling, export, gate2,
hdr, package (`--skip-export` leaves the exported-game check out). A
stage that can't run says why and skips the ones that depend on it. The package is written either way.

Useful flags:

- `--skip-build`: reuse the library already in `addons/prism_rp/bin`.
- `--skip-gate2` and `--skip-hdr`: leave those stages out.
- `--non-interactive`: check that HDR works without asking you about it.

CI runs the same script in software mode (`--software`) on every pull request, so the script itself is tested.

If a run crashes or hangs, narrow it down with the scripts underneath:

```bash
python3 tests/run_gpu_validation.py --scenes 01_cornell_box --configs chronos_default --skip-gate1
```

```bash
python3 tests/check_motion_vectors.py
```

## What to look for

Every capture is frame 48, with the camera orbiting slowly and objects animating. Compare against `baseline_godot`, which is stock Forward+ with SDFGI.

### Errors first

The report's **Errors** section lists any run with a Vulkan validation message (`VUID-...`, `SYNC-HAZARD`), a script error, or an exception. Anything there matters more than how the images look. An error that also shows up in that scene's `baseline_godot` run is labeled as coming from the engine or driver; Godot can trip newer validation checks on its own (4.4 tripped `VUID-vkQueueSubmit-pSignalSemaphores-00067`, swapchain semaphore reuse). The **Prism RP messages** column should be empty except for two configurations. `chronos_godot_taa_guard` must show exactly one warning about Godot TAA, and `chronos_hdr` shows its HDR status lines.

### Chronos

| Config | Looks right | Looks wrong, and what it points to |
| :--- | :--- | :--- |
| `chronos_default` | Clean edges on the wires in scene 03, no swimming, colors close to baseline with a filmic shoulder | Image shifted or wobbling by a pixel: jitter sign. Smearing behind moving objects: velocity sign. Washed out or too contrasty: tonemap contract (Environment must be Linear, exposure 1) |
| `chronos_no_jitter` | Slightly more aliased than default, otherwise identical framing | If this one looks *better* than default, the jitter is being applied or removed with the wrong sign |
| `chronos_taa_only` | Godot's own Linear tonemap, so harsher highlights than default | Black or NaN-speckled frame: TAA history or the composite pass |
| `chronos_no_taa` | Aliased but stable, bloom present | Compare with `chronos_default` to see what TAA buys |
| `chronos_no_capture` | Scene 05 sparks and glass smear more than in default | If it looks the same as default, the transparency reactive term isn't doing anything |
| `chronos_godot_taa_guard` | Same as `chronos_no_taa` plus Godot's TAA, one warning in the log | No warning, or double-blurred image: the guard didn't trigger |
| `chronos_foliage_alpha_hash` (scene 02 only) | Leaf edges soft and stable, about as clean as `chronos_default`'s alpha-scissored leaves | Dither or crawling noise on the leaves: TAA isn't resolving the hash (history rejected on foliage, so check the reactive mask and velocity) |
| `chronos_hdr` (scenes 01, 05) | The log shows `HDR output at capture: enabled=true` with the display's peak, and Chronos' "HDR output detected" line. The PNG is clamped to SDR, so the image itself looks like default | `enabled=false` on an HDR monitor: the OS or compositor didn't grant HDR (on Linux it needs Wayland). The hdr stage below is where HDR is judged |
| `chronos_debug_velocity` | Mostly dark, smooth color on moving objects (cube in 01, louvers in 03) | Speckle or a uniform tint everywhere on a static background: jitter not removed from velocity |
| `chronos_debug_reactive` | Hot on scene 05 sparks, glass and the orb; cool on static walls | Hot everywhere: thresholds too low. Cold on the sparks: transparency term or opaque capture broken |
| `chronos_debug_bloom` | Soft glow around the ceiling emitter (01) and orb (05), round on screen, faint red halation | Stretched horizontally: aspect correction. Glow wrapping from the opposite edge: FFT padding |
| `chronos_debug_confidence` | Mostly green, red at disocclusions and screen edges | All red: history reset every frame. All green on moving objects: confidence never drops |

### Lumina

Lumina adds indirect light, computed from what's on screen and bounced more than once (the rig's `--prism-gi-single-bounce` keeps it to one). The Cornell box (01) is the clearest test: red and green walls should tint the white floor, block and ceiling next to them.

| Config | Looks right | Looks wrong, and what it points to |
| :--- | :--- | :--- |
| `lumina_default` | Red/green bleed in 01, softer shadowed areas than baseline, no visible noise | Uniform brightening with no color bleed: tracing misses everything. Blotchy: denoiser settings. Light leaking through walls: thickness too large |
| `lumina_white_albedo` | Bleed visible as tinted light on grey surfaces | Use this to judge the lighting alone. If it's right here but wrong in default, the albedo proxy is the problem |
| `lumina_no_denoise` | Noisy but the same overall brightness and color as default | Much darker or brighter than default: the denoiser isn't energy-conserving |
| `lumina_one_cascade` | Only short-range bleed near contacts | Identical to default: upper cascades aren't contributing (merge) |
| `lumina_no_world_cache` | Like `lumina_default`, with less bounce light in corners and near the screen edges | Identical to default: the cache adds nothing. Default has blotches that follow the camera and this doesn't: cache cells not being reset when the grid scrolls |
| `lumina_debug_world_only` | Soft, low-frequency light; still a red tint near the red wall in 01 | Black: the cache never fills (update rays rejected). Blocky 1 m pattern: probe interpolation |
| `lumina_debug_raw` | Grainy colored light, brighter near walls | Black: trace or integrate broken. Grid pattern at 8 px: probe interpolation |
| `lumina_debug_denoised` | Smooth version of raw, edges preserved | Halos across object edges: depth/normal edge-stopping |
| `lumina_debug_history` | Green on static surfaces, red at newly revealed areas | All red: reprojection rejecting everything (velocity sign or geometry test) |
| `lumina_debug_variance` | Dark with bright noise at edges and in motion | Uniformly bright: variance never drops |
| `lumina_debug_normals` | Floor greenish (+Y up), walls facing the camera bluish | Colors swapped or inverted: normal decode or view-space convention |
| `lumina_debug_albedo` | Flat-looking version of the scene's colors | Mostly grey on colored surfaces: proxy too weak |

### Everything together

`full_pipeline` should look like `lumina_default` with Chronos' anti-aliasing, bloom and filmic look applied on top. Watch scene 05 for ghosting on the sparks.

### Motion vectors

The `motion` stage checks the motion vectors Chronos uses, numerically:

- It freezes scene 01 and yaws the camera at a constant rate.
- It captures Chronos' signed velocity view (debug view 5) as a float EXR.
- It compares that against motion vectors computed from the logged camera orientations. Under pure rotation those
  don't depend on depth, so they're exact.

There are three runs: no motion (it must read zero); yawing without jitter; and yawing with `ChronosJitterCamera`'s
jitter, where compensation must leave the same field. It passes when the median error is below 0.25 internal
pixels. Results are in `motion_check/motion_check.json`.

What a failure points to:

- **Measured motion has the opposite sign** (error about twice the motion): Godot's motion-vector convention changed.
  Godot 4.6 reportedly touched motion vectors, which is why this check exists.
- **Only the jitter run is off, by about a pixel:** jitter compensation.
- **Off by a constant factor:** units (uv versus pixels).
- **The no-motion run reads non-zero:** jitter or sky handling.

### HDR

When Godot can drive the display in HDR, the `hdr` stage shows scene 05 for about 15 seconds twice. In the first run,
Chronos' look follows the display peak; in the second, it's limited to SDR. Then it asks two questions. Highlights
(sparks, the orb, glass reflections) should be visibly brighter in the first run without clipping to flat white,
while shadows and midtones look the same in both. If Godot can't enable HDR, the stage records why and skips.

### Timings

The Chronos, Optics and Lumina columns are GPU milliseconds from the in-engine timestamps. Compare `full_pipeline` against the tier budgets printed at the top of the report. For the Chronos breakdown, each run's `telemetry.json` has per-dispatch entries (`chronos_taau`, `chronos_fft_bloom_forward_rows`, and so on).

### Gate 1

The report ends with a Gate 1 table. It compares Chronos at 50% render scale against a 2x2 supersampled, non-temporal reference using NVIDIA FLIP; the roadmap's bar is a mean below 0.08. Godot caps supersampling at 2x, so for tier 2 this is 2880p rendered down to 1440p rather than a true 4K reference.

### Gate 2

Gate 2 needs Blender (4.x or 5.x) on `PATH`; the session runs it automatically when Blender is there. It renders Lumina's irradiance in Godot, exports the same frame, and path-traces it in Cycles with one diffuse bounce. To run it on its own:

```bash
python3 tests/run_gate2.py --scenes 01_cornell_box 04_dynamic_sponza
```

It compares bounce light as a viewer sees it (irradiance times each surface's albedo, in a matte version of the scene) and prints raw and brightness-fitted PSNR per scene. The fitted number is the one the gate uses: it must beat SDFGI's and clear the interim floor of 25 dB (`ROADMAP.md`, `docs/NEXT_STEPS.md`); the fitted scale says how far Godot's and Blender's light units are apart, so a scale far from 1 with a good fitted PSNR means calibration, not a GI bug. It also prints the fitted PSNR of the central 70% of the frame: a much better centre means the loss is at the screen edges, where screen-space rays run out. Cycles on CPU at 512 samples takes a few minutes per scene; `--samples 128` is fine for a first look.

What is compared: the bounce light of the scene's lights. Emission is off in every Gate 2 capture, because emissive surfaces are often stand-ins drawn next to a real light (the Cornell box's ceiling panel) and would light the scene twice. Stock Godot's SDFGI is measured the same way (the final image with SDFGI on minus off, one bounce, in the same matte scene, weighted the same way), and Gate 2 needs Lumina to beat it. Each scene's `compare.png` and `compare_sdfgi.png` show the reference, the GI (brightness-fitted) and their difference at one exposure; `gate2.json` has the numbers. CI runs the same comparison on lavapipe at 128 samples (report only), with `--noise-check`: the reference rendered twice with different seeds, which bounds the best score anything can reach against it.

The session's **settling** stage (`tests/check_gi_convergence.py`) switches the scene's lights on at frame 30 with the camera still and counts the frames until Lumina's irradiance is within 10% of where it ends up. Gate 2 needs 6 or fewer.

## After the run

Once the images look right, record the golden images and baseline timings so CI and later runs can catch regressions:

```bash
python3 tests/run_visual_regression.py --tier tier2 --update-golden
```

Then commit `tests/golden/`.
