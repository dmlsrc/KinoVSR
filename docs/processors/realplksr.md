# RealPLKSR: per-frame 2x or 4x upscaling

RealPLKSR is a stateless image upscaler. Its named checkpoints target
different scales and source conditions. Weights are external.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>realplksr</code></td><td>Run a RealPLKSR checkpoint on every frame.</td></tr>
<tr><td rowspan="3" valign="top"><code>--realplksr-profile</code></td><td rowspan="3" valign="top"><code>public2x</code></td><td><code>public2x</code></td><td>2x real-world photo/JPEG model.</td></tr>
<tr><td><code>public2x-nn</code></td><td>2x sibling trained without noise, for cleaner sources.</td></tr>
<tr><td><code>nomos4x</code></td><td>4x web-photo model.</td></tr>
<tr><td><code>--realplksr-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--realplksr-scale</code></td><td rowspan="2" valign="top"><em>profile scale</em></td><td><code>2</code></td><td>Required for explicit weights and must match the checkpoint.</td></tr>
<tr><td><code>4</code></td><td>Required for explicit weights and must match the checkpoint.</td></tr>
<tr><td rowspan="2" valign="top"><code>--realplksr-dtype</code></td><td rowspan="2" valign="top"><code>float16</code></td><td><code>float16</code></td><td>Normal fast path.</td></tr>
<tr><td><code>float32</code></td><td>Useful for numerical comparison.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale realplksr --realplksr-profile public2x
```

This processor has no temporal state. Compare adjacent frames for changing
fine texture before choosing a sharper profile.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
