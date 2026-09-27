# SAFMN: per-frame lightweight or perceptual upscaling

SAFMN is a family of stateless image upscalers. Profiles vary in scale,
training domain, and license. The stock `real` models can produce a transient
block lattice; `--safmn-pool-clamp` moderates it. No profile is bundled.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>safmn</code></td><td>Run a SAFMN checkpoint on every frame.</td></tr>
<tr><td rowspan="6" valign="top"><code>--safmn-profile</code></td><td rowspan="6" valign="top"><code>light</code></td><td><code>light</code></td><td>4x compact fidelity model trained on compressed content.</td></tr>
<tr><td><code>real</code></td><td>4x real-world perceptual model.</td></tr>
<tr><td><code>real2x</code></td><td>2x real-world perceptual model.</td></tr>
<tr><td><code>purescale</code></td><td>4x clean-source model; noncommercial license.</td></tr>
<tr><td><code>purescale2x</code></td><td>2x clean-source model; noncommercial license.</td></tr>
<tr><td><code>purescale2x-sharp</code></td><td>2x clean-source deblur variant; noncommercial license.</td></tr>
<tr><td><code>--safmn-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--safmn-scale</code></td><td rowspan="2" valign="top"><em>profile scale</em></td><td><code>2</code></td><td>Required for explicit weights and must match the checkpoint.</td></tr>
<tr><td><code>4</code></td><td>Required for explicit weights and must match the checkpoint.</td></tr>
<tr><td rowspan="3" valign="top"><code>--safmn-safm-up</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Use the upsampler the checkpoint was trained with.</td></tr>
<tr><td><code>nearest</code></td><td>Force nearest-neighbor modulation.</td></tr>
<tr><td><code>bicubic</code></td><td>Force smooth modulation; a creative texture dial on stock <code>real</code> models that can shimmer.</td></tr>
<tr><td rowspan="2" valign="top"><code>--safmn-pool-clamp</code></td><td rowspan="2" valign="top"><code>0</code></td><td><code>0</code></td><td>Off.</td></tr>
<tr><td><code>K</code> (positive)</td><td>Clamp pooled features to K sigmas; lower K suppresses the lattice more strongly. Start at 3 for stock <code>real</code> weights.</td></tr>
</tbody>
</table>

PureScale checkpoints use a noncommercial CC BY-NC-SA 4.0 license; see the
[processor matrix](../PROCESSORS.md) and family weight attribution. They
have little noise-removal prior, so pre-clean noisy or compressed input and
check for etched, flickering texture. PureScale models do not need the pool
clamp; very low K on stock weights can dull highlights and texture.

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale safmn --safmn-profile real2x --safmn-pool-clamp 3
```

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
