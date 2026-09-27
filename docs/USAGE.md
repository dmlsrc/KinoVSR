# Usage guide

KinoVSR processes video through an ordered chain of **stages**. A stage has a
processor (implementation), a capability (job), and sometimes a profile (named
model or native mode). Start with a short section of the source, inspect motion
as well as still frames, then add stages one at a time. The CLI reads a file;
[the host API](API.md) accepts frames from another application.

## Start with one operation

```bash
kinovsr run --video in.mp4 --output-dir out --upscale balanced
kinovsr run --video in.mp4 --output-dir out --deblock stdf
```

The first command uses VideoToolbox's native 4x `balanced` upscaler and needs no
external weights. The second cleans at source size because `--upscale` defaults
to `none`. Source audio is carried by default when present; use `--no-audio`
for silent output. Use `--start` and `--end` to try a short representative
section. An input named `My Clip.mov` gets output names beginning
`vsr_My_Clip_` by default.

| Problem in the source | First processor to try | Why |
| --- | --- | --- |
| Bars around the picture | [crop](processors/crop.md) | Remove bars before they influence restoration. |
| A few damaged edge rows | [sanitize_edges](processors/sanitize_edges.md) | Hide border damage while retaining the frame size. |
| Whole-frame exposure pumping | [level](processors/level.md) | Stabilize brightness before temporal analysis. |
| Flicker in otherwise static regions | [deflicker](processors/deflicker.md) | Correct only verified-static pixels. |
| Compression blocks or ringing | [stdf](processors/stdf.md), [toflow](processors/toflow.md), [fbcnn](processors/fbcnn.md) | Clean before enlarging the artifacts. |
| Temporal compression damage | [BasicVSR++ restore](processors/basicvsrpp.md) | Use neighboring frames to stabilize artifacts. |
| Sensor or grain noise | [fastdvdnet](processors/fastdvdnet.md), [bsvd](processors/bsvd.md), [pvdd](processors/pvdd.md), [mc](processors/mc.md) | Match the model's noise and temporal behavior to the source. |
| Small or soft picture | [VideoToolbox](processors/videotoolbox.md) or a [learned upscaler](#processor-pages) | Choose scale and rendering style after cleanup. |
| Too few frames per second | [VideoToolbox interpolation](processors/videotoolbox.md) | Synthesize frames on a new time grid. |

## How flag runs order stages

| Part of the chain | Default order | What can move with flags |
| --- | --- | --- |
| Timeline and geometry | `conform` -> `crop` -> `sanitize_edges` -> `square_pixels` -> `cut_detect` | These stay before restoration. Only selected stages appear. |
| Preprocess | `level` -> `restore` -> `deflicker` -> `deblock` -> `denoise` -> `nafnet` | Use `--denoise-first` or `--preprocess-order`. |
| Output processing | `upscale` -> frame interpolation | These stay after preprocess. Only selected stages appear. |

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--denoise-first</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Keep deblock before denoise.</td></tr>
<tr><td><code>on</code></td><td>Move denoise before deblock when noise was added after compression.</td></tr>
<tr><td><code>--preprocess-order</code></td><td>Order above</td><td><code>A,B,...</code></td><td>List enabled preprocess slots in execution order. Overrides <code>--denoise-first</code>; omitted slots follow in default order.</td></tr>
<tr><td rowspan="5" valign="top"><code>--deblock</code></td><td rowspan="5" valign="top"><code>off</code></td><td><code>off</code></td><td>Skip deblocking.</td></tr>
<tr><td><code>stdf</code></td><td><a href="processors/stdf.md">STDF</a> uses neighboring frames to remove compression artifacts.</td></tr>
<tr><td><code>toflow</code></td><td><a href="processors/toflow.md">TOFlow</a> uses a temporal deblock checkpoint.</td></tr>
<tr><td><code>fbcnn</code></td><td><a href="processors/fbcnn.md">FBCNN</a> cleans JPEG-family artifacts per frame.</td></tr>
<tr><td><code>NAME,NAME,...</code></td><td>Chain the listed deblockers left to right.</td></tr>
<tr><td rowspan="8" valign="top"><code>--denoise</code></td><td rowspan="8" valign="top"><code>off</code></td><td><code>off</code></td><td>Skip denoising.</td></tr>
<tr><td><code>spatial</code></td><td><a href="processors/spatial.md">Core Image</a> per-frame noise reduction.</td></tr>
<tr><td><code>mc</code></td><td><a href="processors/mc.md">MC</a> averages motion-aligned history.</td></tr>
<tr><td><code>fastdvdnet</code></td><td><a href="processors/fastdvdnet.md">FastDVDnet</a> uses a causal learned window.</td></tr>
<tr><td><code>bsvd</code></td><td><a href="processors/bsvd.md">BSVD</a> uses buffered future and past context.</td></tr>
<tr><td><code>toflow</code></td><td><a href="processors/toflow.md">TOFlow</a> uses its denoise checkpoint.</td></tr>
<tr><td><code>pvdd</code></td><td><a href="processors/pvdd.md">PVDD</a> offers real-noise and level-conditioned profiles.</td></tr>
<tr><td><code>NAME,NAME,...</code></td><td>Chain the listed denoisers left to right.</td></tr>
<tr><td rowspan="2" valign="top"><code>--restore</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Skip BasicVSR++ restoration.</td></tr>
<tr><td><code>PROFILE[,PROFILE...]</code></td><td>Run <a href="processors/basicvsrpp.md">BasicVSR++ restore</a> profiles left to right.</td></tr>
<tr><td rowspan="13" valign="top"><code>--upscale</code></td><td rowspan="13" valign="top"><code>none</code></td><td><code>none</code></td><td>Keep the processed picture at source size.</td></tr>
<tr><td><code>fast</code></td><td><a href="processors/videotoolbox.md">VideoToolbox</a> low-latency 2x.</td></tr>
<tr><td><code>balanced</code></td><td><a href="processors/videotoolbox.md">VideoToolbox</a> temporal 4x.</td></tr>
<tr><td><code>image</code></td><td><a href="processors/videotoolbox.md">VideoToolbox</a> per-frame 4x.</td></tr>
<tr><td><code>basicvsrpp</code></td><td><a href="processors/basicvsrpp.md">BasicVSR++</a> temporal 4x.</td></tr>
<tr><td><code>realbasicvsr</code></td><td><a href="processors/realbasicvsr.md">RealBasicVSR</a> cleaning and temporal 4x.</td></tr>
<tr><td><code>realesrgan</code></td><td><a href="processors/realesrgan.md">Real-ESRGAN</a> per-frame 4x.</td></tr>
<tr><td><code>safmn</code></td><td><a href="processors/safmn.md">SAFMN</a> per-frame 4x.</td></tr>
<tr><td><code>esc</code></td><td><a href="processors/esc.md">ESC</a> per-frame 4x.</td></tr>
<tr><td><code>realviformer</code></td><td><a href="processors/realviformer.md">RealViFormer</a> recurrent 4x.</td></tr>
<tr><td><code>realplksr</code></td><td><a href="processors/realplksr.md">RealPLKSR</a> per-frame 2x or 4x, depending on profile.</td></tr>
<tr><td><code>toflow</code></td><td><a href="processors/toflow.md">TOFlow</a> temporal 4x.</td></tr>
<tr><td><code>metalfx</code></td><td><a href="processors/metalfx.md">MetalFX</a> native 2x, 3x, or 4x.</td></tr>
<tr><td rowspan="2" valign="top"><code>--target-fps</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Keep source cadence after upscaling.</td></tr>
<tr><td><code>RATE</code></td><td>Interpolate to the requested frame rate after upscaling.</td></tr>
</tbody>
</table>

`--preprocess-order` accepts `level`, `restore`, `deflicker`, `deblock`,
`denoise`, and `nafnet`. It runs only enabled slots. Any enabled slot omitted
from the list is appended in the default preprocess order. In particular,
listing an order without `level` moves an enabled `level` stage after the
listed slots.

```bash
kinovsr run --video in.mp4 --output-dir out \
  --level hist --denoise spatial --deblock stdf \
  --preprocess-order level,denoise,deblock --upscale balanced
```

For several processors in one slot, the names and strength values line up
left to right:

```bash
kinovsr run --video in.mp4 --output-dir out \
  --deblock toflow,stdf --deblock-strength 0.5,0.7 \
  --upscale image
```

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--deblock-strength</code></td><td rowspan="2" valign="top"><code>1.0</code></td><td><code>S</code></td><td>Use one strength for every deblock stage.</td></tr>
<tr><td><code>S,S,...</code></td><td>Assign one strength to each deblock stage, in chain order.</td></tr>
<tr><td rowspan="2" valign="top"><code>--denoise-strength</code></td><td rowspan="2" valign="top"><code>0.5</code></td><td><code>S</code></td><td>Use one strength for every supported denoise stage.</td></tr>
<tr><td><code>S,S,...</code></td><td>Assign one strength to each denoise stage, in chain order.</td></tr>
<tr><td><code>--bsvd-strength</code> (example)</td><td>Slot value</td><td><code>S</code></td><td>A family control overrides that family's slot value. See its processor page.</td></tr>
</tbody>
</table>

A comma list of strengths must match the number of stages in its slot. The word
*strength* is family-specific: MC uses a history blend, BSVD and FastDVDnet map
it to noise sigma, and TOFlow uses an output blend. PVDD has no strength dial;
choose its noise conditioning instead. The [processor pages](#processor-pages)
explain each dial.

## Use TOML for an exact whole-chain order

A TOML `pipeline` lists stage table names in execution order. It can place
preprocess stages in any valid order and name repeated instances. This example
runs denoise before deblock, then upscales:

```toml
pipeline = ["denoise", "deblock", "upscale"]

[input]
video = "in.mp4"

[output]
output_dir = "out"

[denoise]
processor = "spatial"
capability = "denoise"
strength = 0.3

[deblock]
processor = "stdf"
capability = "deblock"
profile = "mfqev2"
strength = 0.7

[upscale]
processor = "videotoolbox"
capability = "upscale"
profile = "balanced"
```

```bash
kinovsr run --config run.toml
```

| TOML rule | Effect |
| --- | --- |
| `pipeline = ["name", ...]` | Each name refers to a stage table; the list owns the complete order. |
| `processor` and `capability` | Choose the implementation and its job; `profile` chooses a named model or mode. |
| `--denoise`, `--upscale`, `--preprocess-order`, and other stage selectors | Rejected when TOML already declares `pipeline`; the stage list has one owner. |
| Family dials such as `--bsvd-strength` | Still apply to matching TOML stages. |
| `--set name.key=value` | Change one named stage after all other settings; use TOML syntax for values. |

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--base-config</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Load a base TOML file before <code>--config</code>.</td></tr>
<tr><td><code>--config</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Load a run TOML file.</td></tr>
<tr><td><code>--set</code></td><td><em>none</em></td><td><code>name.key=value</code></td><td>Override one resolved setting last; repeated flags apply in order.</td></tr>
<tr><td rowspan="2" valign="top"><code>--print-config</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Run normally.</td></tr>
<tr><td><code>on</code></td><td>Print resolved TOML and exit; use it to find stage names and save a flag-authored run.</td></tr>
</tbody>
</table>

`--print-config` probes the input to determine geometry, so it needs a readable
source file. It does not process the clip:

```bash
kinovsr run --video in.mp4 --output-dir out --denoise bsvd --print-config > run.toml
```

See [Configuring runs](CONFIG.md) for merge precedence and all run-level TOML
tables.

## Conditioning maps

`--noise-map auto` estimates a local noise sigma. The map changes model
conditioning for supported denoisers; it is not a general output-opacity
control. `--deblock-map auto` instead estimates where blocking is visible and
blends in the selected deblock correction only there.

| Denoise model or profile | Auto noise map | Interaction with strength |
| --- | --- | --- |
| [FastDVDnet](processors/fastdvdnet.md) | Supported | Replaces the sigma derived from `--denoise-strength` or `--fastdvdnet-strength` once estimated; the chosen strength supplies the fallback sigma. |
| [BSVD](processors/bsvd.md) | Supported | Replaces the sigma derived from `--denoise-strength` or `--bsvd-strength` once estimated; the chosen strength supplies the fallback sigma. |
| [MC](processors/mc.md) | Supported | Replaces `--mc-sigma` when estimated. `--denoise-strength` or `--mc-strength` still sets the maximum history blend. |
| [PVDD `pvdd_level`](processors/pvdd.md) | Supported | Squares estimated sigma into the model's variance input. The selected noise preset or variance is the fallback; `--denoise-strength` does not apply. |
| PVDD `pvdd` | Unsupported: blind profile | No noise-map input; `--denoise-strength` also does not apply. |
| PVDD `crvd` | Unsupported: blind profile | No noise-map input; `--denoise-strength` also does not apply. |
| PVDD `davis` | Unsupported: blind profile | No noise-map input; `--denoise-strength` also does not apply. |
| PVDD `pvdd_raw` | Unavailable in the video pipeline | Needs packed Bayer input, which this pipeline does not provide. |
| PVDD `pvdd_raw_level` | Unavailable in the video pipeline | Needs packed Bayer input, which this pipeline does not provide. |
| [spatial](processors/spatial.md) | Unsupported | `--denoise-strength` or `--spatial-strength` still controls its native noise reduction. |
| [TOFlow denoise](processors/toflow.md) | Unsupported | `--denoise-strength` or `--toflow-strength` remains an output blend. |
| [NAFNet denoise](processors/nafnet.md) | Unsupported: separate preprocess slot | Its own profile and strength control apply; `--noise-map` targets the denoise slot. |

With `--noise-map constant` (the default), FastDVDnet and BSVD use their
strength-derived fixed sigma; MC uses `--mc-sigma`; a PVDD level profile uses
its preset or explicit variance. Shared `--denoise-luma-strength` and
`--denoise-chroma-strength` remain output blends with either map mode, including
for PVDD. The map's floor, gain, and trained-range clamp shape the estimated
sigma; they do not change MC's history-blend ceiling.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--noise-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Use the processor's fixed noise setting.</td></tr>
<tr><td><code>auto</code></td><td>Estimate a spatial noise sigma for supported models.</td></tr>
<tr><td><code>--noise-map-gain</code></td><td><code>1.0</code></td><td><code>G</code> (positive)</td><td>Scale the estimated sigma while retaining its spatial shape.</td></tr>
<tr><td><code>--noise-map-floor</code></td><td><code>0</code></td><td><code>S</code> (<code>0..1</code>)</td><td>Set a minimum sigma when estimation misses static grain or dirt. For FastDVDNet, BSVD, and MC; not accepted by PVDD.</td></tr>
<tr><td><code>--noise-map-masking</code></td><td><code>0</code></td><td><code>S</code> (<code>0..1</code>)</td><td>Favor flat regions and protect detail; <code>1</code> is full masking.</td></tr>
<tr><td rowspan="3" valign="top"><code>--noise-map-motion-cap</code></td><td rowspan="3" valign="top"><code>strict</code></td><td><code>strict</code></td><td>Suppress motion-like blocks; best starting point for moving footage.</td></tr>
<tr><td><code>loose</code></td><td>Preserve more persistent flicker in static-camera footage.</td></tr>
<tr><td><code>off</code></td><td>Do not suppress motion-like blocks; use only when the camera and subject are still.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map-floor-mode</code></td><td rowspan="2" valign="top"><code>mc</code></td><td><code>mc</code></td><td>Estimate the noise floor after motion alignment.</td></tr>
<tr><td><code>flat</code></td><td>Estimate it from flat pixels only.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map-upsample</code></td><td rowspan="2" valign="top"><code>edge</code></td><td><code>edge</code></td><td>Keep noise boundaries aligned with picture edges.</td></tr>
<tr><td><code>box</code></td><td>Use simpler box-blur upsampling for comparison.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map-refresh</code></td><td rowspan="2" valign="top"><code>64</code></td><td><code>0</code></td><td>Estimate once and hold in plain streaming mode.</td></tr>
<tr><td><code>N</code> (positive frame count)</td><td>Re-estimate every N frames for FastDVDNet, BSVD, or MC. PVDD refreshes per window and does not accept this control.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map-pulse</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Use no GOP-phase adjustment.</td></tr>
<tr><td><code>on</code></td><td>Follow per-frame GOP-phase noise spikes with either map mode.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map-debug</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Save no map diagnostics.</td></tr>
<tr><td><code>on</code></td><td>With <code>--noise-map auto</code>, save a viewable map and print its statistics.</td></tr>
</tbody>
</table>

Start with `--noise-map auto` and inspect the debug map before changing masking
or motion policy. Temporal differences cannot detect all static defects; the
floor supplies base cleaning while the map adapts above it.

| Deblock model | Auto block map | Interaction with strength |
| --- | --- | --- |
| [STDF](processors/stdf.md) | Supported | Its luma correction is multiplied by the map and `--deblock-strength` or `--stdf-strength`. |
| [FBCNN](processors/fbcnn.md) | Supported | Its RGB output blend is multiplied by the map and `--deblock-strength` or `--fbcnn-strength`. |
| [TOFlow deblock](processors/toflow.md) | Unsupported | `--deblock-strength` or `--toflow-strength` remains a uniform output blend. |

With `--deblock-map constant` (the default), STDF and FBCNN apply the chosen
strength everywhere. With `auto`, local correction is the selected strength
times the blockiness mask, which is capped at 1 after gain. Gain changes *where*
the correction is applied; strength changes *how much* correction those regions
receive. A gain above 1 cannot push the mask beyond full coverage, but a
strength above 1 can still overdrive the correction.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--deblock-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Apply the selected strength everywhere.</td></tr>
<tr><td><code>auto</code></td><td>Estimate a blockiness mask for STDF and FBCNN.</td></tr>
<tr><td><code>--deblock-map-gain</code></td><td><code>1.0</code></td><td><code>G</code> (positive)</td><td>Above 1 treats more area; below 1 is more conservative. Only used with <code>auto</code>.</td></tr>
</tbody>
</table>

Map flags broadcast only to supported stages in a chain. If no stage accepts a
map flag, the run reports an error. Blind PVDD profiles drop a broadcast noise
map with a warning; `--noise-map auto` still errors if no other stage can use it.

This example uses both maps with bundled cleanup checkpoints, retains 70% of
the STDF correction where blocking is found, and saves the noise map:

```bash
kinovsr run --video in.mp4 --output-dir out \
  --deblock stdf --deblock-strength 0.7 --deblock-map auto \
  --denoise fastdvdnet --noise-map auto --noise-map-floor 0.02 \
  --noise-map-debug
```

For MC, the automatic map adjusts the rejection scale while strength remains
the maximum history blend:

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise mc --denoise-strength 0.6 --noise-map auto
```

## Timeline, cuts, and windows

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="3" valign="top"><code>--conform-cfr</code></td><td rowspan="3" valign="top"><code>off</code></td><td><code>off</code></td><td>Keep source timestamps.</td></tr>
<tr><td><code>auto</code></td><td>Duplicate or drop pictures onto the source's nominal frame rate.</td></tr>
<tr><td><code>RATE</code></td><td>Duplicate or drop pictures onto this constant-rate grid before restoration.</td></tr>
<tr><td rowspan="2" valign="top"><code>--target-fps</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Keep the processed clip's cadence.</td></tr>
<tr><td><code>RATE</code></td><td>Synthesize pictures after upscaling on a new grid; a lower rate can also reduce cadence.</td></tr>
<tr><td rowspan="4" valign="top"><code>--cut-detect</code></td><td rowspan="4" valign="top"><code>off</code></td><td><code>off</code></td><td>Do not mark scene cuts.</td></tr>
<tr><td><code>simple</code></td><td>Use the simple frame-difference detector.</td></tr>
<tr><td><code>hist</code></td><td>Use histogram-based detection.</td></tr>
<tr><td><code>vtme</code></td><td>Use VideoToolbox motion estimates.</td></tr>
<tr><td rowspan="2" valign="top"><code>--gop-align</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Use each model's ordinary window and trim settings.</td></tr>
<tr><td><code>on</code></td><td>Anchor recurrent BSVD, BasicVSR++, RealBasicVSR, and PVDD windows on keyframes.</td></tr>
<tr><td><code>--gop-min-window</code></td><td><code>16</code></td><td><code>N</code> (positive frame count)</td><td>Merge short GOPs to obtain enough temporal context.</td></tr>
<tr><td><code>--gop-max-window</code></td><td><code>96</code></td><td><code>N</code> (positive frame count)</td><td>Split long GOPs to bound memory.</td></tr>
<tr><td rowspan="2" valign="top"><code>--snap-start</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Keep the exact requested start.</td></tr>
<tr><td><code>on</code></td><td>Move <code>--start</code> to a nearby keyframe, changing the requested range.</td></tr>
</tbody>
</table>

Source timestamps, including irregular gaps, are carried by default. A flag
run cannot combine `--conform-cfr` with `--target-fps`; choose frame
repetition/drop or interpolation. `--gop-align` supersedes individual recurrent
window and trim settings. For an exact mid-GOP `--start`, KinoVSR may decode
preceding frames as context; `--snap-start` instead changes the effective start.
See the [conform](processors/conform.md), [cut detection](processors/cut_detect.md),
and [VideoToolbox](processors/videotoolbox.md) pages for mode tradeoffs.

Choose one of these timeline operations for the same source:

```bash
# Keep original pictures and make the timestamps regular.
kinovsr run --video in.mp4 --output-dir out --conform-cfr 30000/1001 \
  --cut-detect hist --upscale image

# Synthesize new pictures at 60 fps.
kinovsr run --video in.mp4 --output-dir out --target-fps 60 \
  --cut-detect hist --upscale image
```

## Profiles, weights, and compute

A profile selects a checkpoint or native mode; it does not guarantee that the
weight file is present. Check before running a learned model:

```bash
kinovsr weights list
kinovsr weights verify
```

The bundled `general` profile is a low-friction way to compare a learned
upscaler with a native VideoToolbox result:

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale realesrgan --realesrgan-profile general
```

| Weight question | Where to look |
| --- | --- |
| Which profiles exist, and which artifacts do they use? | [Processor matrix](PROCESSORS.md) lists profiles, weight sources, and licenses. |
| How do I obtain external checkpoints? | Each family's `weights/README.md` explains download and conversion; runtime loads `.safetensors`, not `.pth` or `.t7`. |
| Which stages need no downloaded weights? | VideoToolbox and MetalFX are native system stages. |

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--&lt;family&gt;-weights</code></td><td>Selected profile</td><td><code>PATH</code></td><td>Override a model's named checkpoint. The exact flag is on its processor page.</td></tr>
<tr><td rowspan="3" valign="top"><code>--spynet-backend</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Use ANE where available and fall back to MLX as needed.</td></tr>
<tr><td><code>ane</code></td><td>Require the Neural Engine path for SpyNet flow.</td></tr>
<tr><td><code>mlx</code></td><td>Run SpyNet flow on MLX/GPU.</td></tr>
<tr><td><code>--mlx-cache-limit-gb</code></td><td><code>1.0</code></td><td><code>G</code> (nonnegative GB)</td><td>Cap MLX's reusable buffer cache; change only for measured memory needs.</td></tr>
<tr><td><code>--video-chunk-size</code></td><td><code>32</code></td><td><code>N</code> (positive frame count)</td><td>Upper bound on reader frames per chunk; a surface-budget limit may lower it.</td></tr>
</tbody>
</table>

MLX models use the GPU; BSVD can instead use an ANE backend. Backend benefit
depends on workload and geometry. See [Performance](PERFORMANCE.md) for device
envelopes and memory guidance.

## Input and output controls

<table>
<thead><tr><th>Input argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--video</code></td><td><em>required unless in config</em></td><td><code>PATH</code></td><td>Select the source file.</td></tr>
<tr><td rowspan="3" valign="top"><code>--reader</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Try native decoding, then the optional ffmpeg reader for unsupported containers.</td></tr>
<tr><td><code>native</code></td><td>Require native video decoding; audio may use PyAV if AVAudioFile rejects the container.</td></tr>
<tr><td><code>ffmpeg</code></td><td>Require the optional PyAV reader.</td></tr>
<tr><td rowspan="6" valign="top"><code>--start</code></td><td rowspan="6" valign="top"><em>beginning</em></td><td><code>N</code></td><td>Bare integer frame number; for example, <code>120</code>.</td></tr>
<tr><td><code>Nf</code></td><td>Explicit frame number; for example, <code>120f</code>.</td></tr>
<tr><td><code>Ns</code></td><td>Seconds with suffix; for example, <code>5s</code> or <code>2.5s</code>.</td></tr>
<tr><td><code>D.D</code></td><td>Decimal seconds without suffix; for example, <code>1.5</code>.</td></tr>
<tr><td><code>MM:SS</code></td><td>Clock time in minutes and seconds; fractional seconds work too, for example <code>0:04.5</code>.</td></tr>
<tr><td><code>HH:MM:SS</code></td><td>Clock time with hours; for example, <code>1:02:03</code>.</td></tr>
<tr><td rowspan="6" valign="top"><code>--end</code></td><td rowspan="6" valign="top"><em>end</em></td><td><code>N</code></td><td>Stop before this source frame; a bare integer means frames.</td></tr>
<tr><td><code>Nf</code></td><td>Stop before this explicitly numbered source frame.</td></tr>
<tr><td><code>Ns</code></td><td>Stop before this time in seconds.</td></tr>
<tr><td><code>D.D</code></td><td>Stop before this decimal-seconds time.</td></tr>
<tr><td><code>MM:SS</code></td><td>Stop before this minutes-and-seconds time.</td></tr>
<tr><td><code>HH:MM:SS</code></td><td>Stop before this hours-minutes-seconds time.</td></tr>
<tr><td rowspan="6" valign="top"><code>--max-frames</code></td><td rowspan="6" valign="top"><em>no cap</em></td><td><code>N</code></td><td>Limit output to N frames, including interpolated frames.</td></tr>
<tr><td><code>Nf</code></td><td>Same frame-count cap with an explicit suffix.</td></tr>
<tr><td><code>Ns</code></td><td>Limit output duration in seconds, measured at the output frame rate.</td></tr>
<tr><td><code>D.D</code></td><td>Limit output duration in decimal seconds.</td></tr>
<tr><td><code>MM:SS</code></td><td>Limit output duration in minutes and seconds.</td></tr>
<tr><td><code>HH:MM:SS</code></td><td>Limit output duration with hours, minutes, and seconds.</td></tr>
<tr><td rowspan="4" valign="top"><code>--source-color</code></td><td rowspan="4" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Trust source color tags or VideoToolbox's guess for untagged clips.</td></tr>
<tr><td><code>bt709</code></td><td>Force BT.709 interpretation.</td></tr>
<tr><td><code>bt601</code></td><td>Force BT.601 interpretation.</td></tr>
<tr><td><code>bt2020</code></td><td>Force BT.2020 interpretation.</td></tr>
<tr><td rowspan="3" valign="top"><code>--source-range</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Trust source range tags; assume video range if untagged.</td></tr>
<tr><td><code>video</code></td><td>Force limited-range interpretation.</td></tr>
<tr><td><code>full</code></td><td>Force full-range interpretation.</td></tr>
</tbody>
</table>

<table>
<thead><tr><th>Output argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--output-dir</code></td><td><em>required unless in config</em></td><td><code>PATH</code></td><td>Select the deliverable directory.</td></tr>
<tr><td><code>--output-prefix</code></td><td><code>vsr_&lt;input stem&gt;</code></td><td><code>TEXT</code></td><td>Prefix timestamped deliverable names. For <code>My Clip.mov</code>, the default begins <code>vsr_My_Clip_</code>; an explicit value overrides it.</td></tr>
<tr><td rowspan="2" valign="top"><code>--audio</code></td><td rowspan="2" valign="top"><code>on</code></td><td><code>on</code></td><td>Include source audio when present; explicit <code>--audio</code> overrides a config that disabled it.</td></tr>
<tr><td><code>off</code></td><td>Use <code>--no-audio</code> for silent output, even when a config enables audio.</td></tr>
<tr><td rowspan="2" valign="top"><code>--audio-codec</code></td><td rowspan="2" valign="top"><code>alac</code></td><td><code>alac</code></td><td>Mux lossless audio.</td></tr>
<tr><td><code>aac</code></td><td>Mux AAC audio.</td></tr>
<tr><td rowspan="2" valign="top"><code>--overwrite</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Refuse a pre-existing deliverable set.</td></tr>
<tr><td><code>on</code></td><td>Replace an existing complete deliverable set.</td></tr>
<tr><td><code>--encode-quality</code></td><td><code>0.65</code></td><td><code>0..1</code></td><td>Trade output size against hardware HEVC quality.</td></tr>
<tr><td rowspan="3" valign="top"><code>--encode-chroma</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Choose chroma subsampling for the upscale mode.</td></tr>
<tr><td><code>420</code></td><td>Request 4:2:0 output.</td></tr>
<tr><td><code>422</code></td><td>Request 4:2:2 output.</td></tr>
<tr><td rowspan="2" valign="top"><code>--save-pre-frames</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Do not save source frame images.</td></tr>
<tr><td><code>on</code></td><td>Save frames before processing.</td></tr>
<tr><td rowspan="2" valign="top"><code>--save-post-frames</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Do not save processed frame images.</td></tr>
<tr><td><code>on</code></td><td>Save processed frames.</td></tr>
<tr><td rowspan="2" valign="top"><code>--comparison</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Do not write a comparison MP4.</td></tr>
<tr><td><code>on</code></td><td>Write a side-by-side before/after MP4.</td></tr>
<tr><td rowspan="2" valign="top"><code>--skip-post-mp4</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Write the processed MP4.</td></tr>
<tr><td><code>on</code></td><td>Omit the processed MP4 when frame output is enough.</td></tr>
<tr><td rowspan="2" valign="top"><code>--save-audio-sidecar</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Do not save a separate audio file.</td></tr>
<tr><td><code>on</code></td><td>Save muxed audio as a WAV sidecar.</td></tr>
</tbody>
</table>

`--max-frames` counts *output* frames, so interpolation can reach the cap
sooner. Color and range overrides are for
sources whose tags cause visibly wrong color or contrast. The
`--source-range` override is unavailable with `--upscale fast`.

The processed MP4 is named `<prefix>_YYYYMMDD_HHMMSS_<8-hex>.mp4`. The time is
shown to the second; the short random suffix avoids collisions.

Some older containers allow native video decoding but cannot be opened by
AVAudioFile for audio. When the optional `ffmpeg` extra is installed, KinoVSR
uses a bounded PyAV audio reader for those tracks while keeping native video
decoding. Unusual audio sample rates are converted to 48 kHz for MP4 output.

For a seven-second comparison clip, source audio is included automatically:

```bash
kinovsr run --video in.mov --output-dir out --start 5s --end 12s \
  --upscale image --encode-quality 0.8 --comparison
```

To make a silent derivative, add `--no-audio`:

```bash
kinovsr run --video in.mov --output-dir silent-out --upscale image --no-audio
```

## Processor pages

Each page lists profiles, defaults, values, and practical limits. The
[processor matrix](PROCESSORS.md) remains the exact artifact and license
inventory.

For example, `--denoise spatial --upscale image` combines the native
[spatial](processors/spatial.md) denoiser and the native
[VideoToolbox](processors/videotoolbox.md) image upscaler:

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise spatial --upscale image
```

| Processor | Primary use |
| --- | --- |
| [basicvsrpp](processors/basicvsrpp.md) | Temporal restoration and 4x upscale. |
| [bsvd](processors/bsvd.md) | Buffered learned denoising. |
| [conform](processors/conform.md) | Constant frame rate without synthesis. |
| [crop](processors/crop.md) | Remove bars and reframe. |
| [cut_detect](processors/cut_detect.md) | Detect scene cuts. |
| [deflicker](processors/deflicker.md) | Stabilize static-region flicker. |
| [esc](processors/esc.md) | Learned per-frame upscale. |
| [fastdvdnet](processors/fastdvdnet.md) | Causal learned denoising. |
| [fbcnn](processors/fbcnn.md) | Per-frame JPEG artifact cleanup. |
| [level](processors/level.md) | Stabilize global exposure. |
| [mc](processors/mc.md) | Motion-compensated denoising. |
| [metalfx](processors/metalfx.md) | Native MetalFX spatial upscale. |
| [nafnet](processors/nafnet.md) | Per-frame deblur, denoise, or restoration. |
| [pvdd](processors/pvdd.md) | Real-noise temporal denoising. |
| [realbasicvsr](processors/realbasicvsr.md) | Clean and upscale real-world video 4x. |
| [realesrgan](processors/realesrgan.md) | Per-frame GAN and fidelity upscaling. |
| [realplksr](processors/realplksr.md) | Per-frame 2x or 4x upscale. |
| [realviformer](processors/realviformer.md) | Causal recurrent 4x upscale. |
| [safmn](processors/safmn.md) | Lightweight or perceptual per-frame upscale. |
| [sanitize_edges](processors/sanitize_edges.md) | Repair edge damage without resizing. |
| [spatial](processors/spatial.md) | Native per-frame noise reduction. |
| [square_pixels](processors/square_pixels.md) | Correct non-square pixel aspect. |
| [stdf](processors/stdf.md) | Temporal compression cleanup. |
| [toflow](processors/toflow.md) | Temporal denoise, deblock, and upscale. |
| [videotoolbox](processors/videotoolbox.md) | Native upscale and interpolation. |
