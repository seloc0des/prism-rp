# GPU Test Plan

Everything in Prism RP so far was built and compiled without a GPU. The shaders validate, the extension loads, and the CI smoke test runs on a software rasterizer. None of that says whether the images are right. This plan is the single session on real hardware that does.

The run is one command. It produces a folder with a screenshot, GPU timings and a log for every scene and configuration, plus a `report.md` that lays them out side by side. Most of the judging is done by eye against the checklist below, so budget about an hour: roughly 15 minutes of runtime and the rest looking.

## Setup

You need a Vulkan 1.3 GPU (anything from the tier table in `ROADMAP.md`), Godot 4.4.1 on `PATH` as `godot`, Python 3.11+ and a build of the extension for your platform.

```bash
git clone --recurse-submodules https://github.com/seloc0des/prism-rp.git
```

```bash
cd prism-rp && git checkout claude/stoic-gauss-az2a98
```

```bash
scons platform=linux target=template_debug -j8
```

On Windows use `platform=windows`. macOS isn't a supported target right now.

```bash
ln -s "$(pwd)/addons/prism_rp" project/addons/prism_rp
```

On Windows, `mklink /D project\addons\prism_rp ..\..\addons\prism_rp` from an elevated prompt does the same.

```bash
godot --headless --path project --import
```

```bash
pip install flip-evaluator numpy pillow
```

To see Vulkan validation errors, install the Vulkan SDK (or your distro's `vulkan-validationlayers`). The scripts turn the layer on through `VK_INSTANCE_LAYERS`.

## Run it

```bash
python3 tests/run_gpu_validation.py --tier tier2
```

Pick the tier that matches the GPU. The output lands in `tests/artifacts/gpu_validation/<timestamp>/`. Open `report.md` in anything that renders Markdown with images (VS Code's preview works).

Send me that whole folder, or at least `report.md`, `results.json` and the `godot.log` of anything flagged. The logs carry the detail I need to fix things.

If a run crashes or hangs, narrow it down:

```bash
python3 tests/run_gpu_validation.py --scenes 01_cornell_box --configs chronos_default --skip-gate1
```

## What to look for

Every capture is frame 48, with the camera orbiting slowly and objects animating. Compare against `baseline_godot`, which is stock Forward+ with SDFGI.

### Errors first

The report's **Errors** section lists any run with a Vulkan validation message (`VUID-...`, `SYNC-HAZARD`), a script error, or an exception. Anything there matters more than how the images look. An error that also shows up in that scene's `baseline_godot` run is labeled as coming from the engine or driver; Godot 4.4 trips at least one newer validation check (`VUID-vkQueueSubmit-pSignalSemaphores-00067`, swapchain semaphore reuse) on its own. The **Prism RP messages** column should be empty except for `chronos_godot_taa_guard`, which must show exactly one warning about Godot TAA.

### Chronos

| Config | Looks right | Looks wrong, and what it points to |
| :--- | :--- | :--- |
| `chronos_default` | Clean edges on the wires in scene 03, no swimming, colors close to baseline with a filmic shoulder | Image shifted or wobbling by a pixel: jitter sign. Smearing behind moving objects: velocity sign. Washed out or too contrasty: tonemap contract (Environment must be Linear, exposure 1) |
| `chronos_no_jitter` | Slightly more aliased than default, otherwise identical framing | If this one looks *better* than default, the jitter is being applied or removed with the wrong sign |
| `chronos_taa_only` | Godot's own Linear tonemap, so harsher highlights than default | Black or NaN-speckled frame: TAA history or the composite pass |
| `chronos_no_taa` | Aliased but stable, bloom present | Compare with `chronos_default` to see what TAA buys |
| `chronos_no_capture` | Scene 05 sparks and glass smear more than in default | If it looks the same as default, the transparency reactive term isn't doing anything |
| `chronos_godot_taa_guard` | Same as `chronos_no_taa` plus Godot's TAA, one warning in the log | No warning, or double-blurred image: the guard didn't trigger |
| `chronos_debug_velocity` | Mostly dark, smooth color on moving objects (cube in 01, louvers in 03) | Speckle or a uniform tint everywhere on a static background: jitter not removed from velocity |
| `chronos_debug_reactive` | Hot on scene 05 sparks, glass and the orb; cool on static walls | Hot everywhere: thresholds too low. Cold on the sparks: transparency term or opaque capture broken |
| `chronos_debug_bloom` | Soft glow around the ceiling emitter (01) and orb (05), round on screen, faint red halation | Stretched horizontally: aspect correction. Glow wrapping from the opposite edge: FFT padding |
| `chronos_debug_confidence` | Mostly green, red at disocclusions and screen edges | All red: history reset every frame. All green on moving objects: confidence never drops |

### Lumina

Lumina adds one bounce of indirect light, computed from what's on screen. The Cornell box (01) is the clearest test: red and green walls should tint the white floor, block and ceiling next to them.

| Config | Looks right | Looks wrong, and what it points to |
| :--- | :--- | :--- |
| `lumina_default` | Red/green bleed in 01, softer shadowed areas than baseline, no visible noise | Uniform brightening with no color bleed: tracing misses everything. Blotchy: denoiser settings. Light leaking through walls: thickness too large |
| `lumina_white_albedo` | Bleed visible as tinted light on grey surfaces | Use this to judge the lighting alone. If it's right here but wrong in default, the albedo proxy is the problem |
| `lumina_no_denoise` | Noisy but the same overall brightness and color as default | Much darker or brighter than default: the denoiser isn't energy-conserving |
| `lumina_one_cascade` | Only short-range bleed near contacts | Identical to default: upper cascades aren't contributing (merge) |
| `lumina_world_cache` | Like `lumina_default`, slightly more bounce light in corners and near the screen edges | Brighter everywhere, or blotches that follow the camera: cache cells not being reset when the grid scrolls |
| `lumina_debug_world_only` | Soft, low-frequency light; still a red tint near the red wall in 01 | Black: the cache never fills (update rays rejected). Blocky 1 m pattern: probe interpolation |
| `lumina_debug_raw` | Grainy colored light, brighter near walls | Black: trace or integrate broken. Grid pattern at 8 px: probe interpolation |
| `lumina_debug_denoised` | Smooth version of raw, edges preserved | Halos across object edges: depth/normal edge-stopping |
| `lumina_debug_history` | Green on static surfaces, red at newly revealed areas | All red: reprojection rejecting everything (velocity sign or geometry test) |
| `lumina_debug_variance` | Dark with bright noise at edges and in motion | Uniformly bright: variance never drops |
| `lumina_debug_normals` | Floor greenish (+Y up), walls facing the camera bluish | Colors swapped or inverted: normal decode or view-space convention |
| `lumina_debug_albedo` | Flat-looking version of the scene's colors | Mostly grey on colored surfaces: proxy too weak |

### Everything together

`full_pipeline` should look like `lumina_default` with Chronos' anti-aliasing, bloom and filmic look applied on top. Watch scene 05 for ghosting on the sparks.

### Timings

The Chronos, Optics and Lumina columns are GPU milliseconds from the in-engine timestamps. Compare `full_pipeline` against the tier budgets printed at the top of the report. For the Chronos breakdown, each run's `telemetry.json` has per-dispatch entries (`chronos_taau`, `chronos_fft_bloom_forward_rows`, and so on).

### Gate 1

The report ends with a Gate 1 table. It compares Chronos at 50% render scale against a 2x2 supersampled, non-temporal reference using NVIDIA FLIP; the roadmap's bar is a mean below 0.08. Godot caps supersampling at 2x, so for tier 2 this is 2880p rendered down to 1440p rather than a true 4K reference.

### Gate 2

Gate 2 needs Blender (4.x or 5.x) on `PATH` and `pip install OpenEXR` as well. It renders Lumina's irradiance in Godot, exports the same frame, and path-traces it in Cycles with one diffuse bounce:

```bash
python3 tests/run_gate2.py --scenes 01_cornell_box 04_dynamic_sponza
```

It prints raw and brightness-fitted PSNR per scene. The fitted number is the one the gate uses (above 36 dB passes); the fitted scale says how far Godot's and Blender's light units are apart, so a scale far from 1 with a good fitted PSNR means calibration, not a GI bug. Cycles on CPU at 512 samples takes a few minutes per scene; `--samples 128` is fine for a first look.

## After the run

Once the images look right, record the golden images and baseline timings so CI and later runs can catch regressions:

```bash
python3 tests/run_visual_regression.py --tier tier2 --update-golden
```

Then commit `tests/golden/`.
