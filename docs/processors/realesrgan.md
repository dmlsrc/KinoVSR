# Real-ESRGAN: per-frame learned upscaling

Real-ESRGAN combines several stateless image-model styles. `general` is a
bundled, relatively fast starting point; larger GAN models can look sharper
but may shimmer on video. Most other checkpoints are external.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>realesrgan</code></td><td>Run a Real-ESRGAN family checkpoint independently on each frame.</td></tr>
<tr><td rowspan="9" valign="top"><code>--realesrgan-profile</code></td><td rowspan="9" valign="top"><code>general</code></td><td><code>general</code></td><td>Bundled 4x SRVGG general-purpose model; relatively fast and gentle.</td></tr>
<tr><td><code>x4plus</code></td><td>4x crisp RRDBNet GAN model at much higher cost.</td></tr>
<tr><td><code>realesrnet</code></td><td>4x fidelity-oriented, softer output.</td></tr>
<tr><td><code>bsrnet</code></td><td>4x fidelity-oriented, softer output.</td></tr>
<tr><td><code>bsrgan</code></td><td>4x perceptual alternative.</td></tr>
<tr><td><code>x2plus</code></td><td>2x model.</td></tr>
<tr><td><code>anime</code></td><td>4x animation-oriented models.</td></tr>
<tr><td><code>animevideo</code></td><td>4x animation-oriented models.</td></tr>
<tr><td><code>esrgan</code></td><td>4x original ESRGAN model.</td></tr>
<tr><td><code>--realesrgan-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td><code>--realesrgan-scale</code></td><td><em>profile scale</em></td><td><code>N</code> (positive integer)</td><td>Declare the factor for explicit weights; it must match the checkpoint.</td></tr>
<tr><td><code>--realesrgan-denoise-strength</code></td><td><code>1.0</code></td><td><code>0..1</code></td><td>For <code>general</code> only: 1 uses general; lower blends toward companion WDN weights and retains more grain/texture.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale realesrgan --realesrgan-profile general
```

The denoise-strength setting is checkpoint interpolation, not a general
post-upscale filter. There are no flow or window controls because each frame
is independent.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
