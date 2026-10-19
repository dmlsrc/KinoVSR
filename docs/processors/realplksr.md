# RealPLKSR: per-frame restoration or upscaling

RealPLKSR is a stateless image model. The `deh264` checkpoint restores H.264
artifacts at the original size; the other profiles upscale by 2x or 4x.
Every frame is processed independently, so inspect adjacent frames for changing
texture or artifact removal. Weights are external.

## Restore H.264 video

```bash
kinovsr run --video in.mp4 --output-dir out \
  --restore deh264 --restore-strength 1.0 --upscale none
```

`--restore-strength` blends the restored frame with the input: 0 keeps the
input, 1 applies the full model output. It also accepts a comma list when
chaining restore profiles. `deh264` is a 1x per-frame profile; it does not use
the BasicVSR++ temporal restore flags (`--restore-window`, `--restore-trim`,
`--restore-flow`, or `--restore-ensemble`). It can precede an upscaler.

The equivalent hand-written pipeline stage is:

```toml
pipeline = ["cleanup"]

[cleanup]
processor = "realplksr"
capability = "restore"
profile = "deh264"
strength = 1.0
```

## Upscale arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>realplksr</code></td><td>Run a RealPLKSR checkpoint on every frame.</td></tr>
<tr><td rowspan="3" valign="top"><code>--realplksr-profile</code></td><td rowspan="3" valign="top"><code>public2x</code></td><td><code>public2x</code></td><td>2x real-world photo/JPEG model.</td></tr>
<tr><td><code>public2x-nn</code></td><td>2x sibling trained without noise, for cleaner sources.</td></tr>
<tr><td><code>nomos4x</code></td><td>4x web-photo model.</td></tr>
<tr><td><code>--realplksr-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file for restore or upscale.</td></tr>
<tr><td rowspan="3" valign="top"><code>--realplksr-scale</code></td><td rowspan="3" valign="top"><em>profile scale</em></td><td><code>1</code></td><td>Same-size restoration; required for an explicit 1x path without a profile.</td></tr>
<tr><td><code>2</code></td><td>Required for explicit 2x weights and must match the checkpoint.</td></tr>
<tr><td><code>4</code></td><td>Required for explicit weights and must match the checkpoint.</td></tr>
<tr><td rowspan="2" valign="top"><code>--realplksr-dtype</code></td><td rowspan="2" valign="top"><code>float16</code></td><td><code>float16</code></td><td>Normal fast path.</td></tr>
<tr><td><code>float32</code></td><td>Useful for numerical comparison.</td></tr>
</tbody>
</table>

For an explicit 1x restoration checkpoint, declare `scale = 1` in a pipeline
stage or use `--realplksr-scale 1` with `--restore deh264`. Named profiles
already declare their scale. Use `--realplksr-dtype float32` to compare full
precision against the default float16 path; the existing 4x precision result
does not validate this checkpoint.

## Upscale example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale realplksr --realplksr-profile public2x
```

This processor has no temporal state. Compare adjacent frames for changing
fine texture before choosing a sharper profile.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
