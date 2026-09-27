# ESC: per-frame 4x real-world upscaling

ESC applies an image upscaler independently to each frame. Choose `gan` for
perceptual sharpening or `mse` for a fidelity-oriented result; compare
moving detail before deciding. Neither checkpoint is bundled.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>esc</code></td><td>Run the ESC 4x processor.</td></tr>
<tr><td rowspan="2" valign="top"><code>--esc-profile</code></td><td rowspan="2" valign="top"><code>gan</code></td><td><code>gan</code></td><td>Perceptual 4x output with synthesized sharpness.</td></tr>
<tr><td><code>mse</code></td><td>Fidelity-oriented 4x output.</td></tr>
<tr><td><code>--esc-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td><code>--esc-scale</code></td><td>profile scale <code>4</code></td><td><code>N</code> (positive integer)</td><td>Required with a custom weight path; must match that checkpoint's output factor.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale esc --esc-profile mse
```

ESC has no window or optical-flow controls. Clean compression and noise
before magnifying them.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
