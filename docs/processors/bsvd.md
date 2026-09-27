# BSVD: buffered temporal denoising

BSVD uses future and past context to denoise video, adding about 16 frames of
output delay. Its checkpoints are external. The implementation can run on
MLX/GPU or an Apple Neural Engine backend.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--denoise</code></td><td><code>off</code></td><td><code>bsvd</code></td><td>Add BSVD to the denoise chain.</td></tr>
<tr><td rowspan="2" valign="top"><code>--bsvd-profile</code></td><td rowspan="2" valign="top"><code>c64</code></td><td><code>c64</code></td><td>Matches the public unblind test configuration.</td></tr>
<tr><td><code>c32</code></td><td>Smaller alternative with weaker provenance.</td></tr>
<tr><td><code>--bsvd-strength</code></td><td><code>0.5</code></td><td><code>0..1</code></td><td>Override the denoise slot strength for BSVD; maps to trained sigma 5-55/255.</td></tr>
<tr><td rowspan="2" valign="top"><code>--bsvd-dtype</code></td><td rowspan="2" valign="top"><code>float16</code></td><td><code>float16</code></td><td>Normal MLX path.</td></tr>
<tr><td><code>float32</code></td><td>Useful for numerical parity checks.</td></tr>
<tr><td rowspan="3" valign="top"><code>--bsvd-backend</code></td><td rowspan="3" valign="top"><code>mlx</code></td><td><code>mlx</code></td><td>GPU implementation; simplest standalone choice. Settings can override this default.</td></tr>
<tr><td><code>ane</code></td><td>Core ML Neural Engine implementation that can overlap downstream GPU work.</td></tr>
<tr><td><code>mpsgraph</code></td><td>MPSGraph Neural Engine route with a different supported geometry envelope.</td></tr>
<tr><td><code>--bsvd-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Use the strength-derived sigma everywhere.</td></tr>
<tr><td><code>auto</code></td><td>Use estimated per-pixel sigma in place of strength-derived sigma; strength remains the fallback. See <a href="../USAGE.md#conditioning-maps">map controls</a>.</td></tr>
<tr><td><code>--denoise-luma-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code> (unclamped)</td><td>Lower to retain more of the original luma detail.</td></tr>
<tr><td><code>--denoise-chroma-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code> (unclamped)</td><td>Lower to retain more of the original chroma.</td></tr>
<tr><td><code>--gop-align</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Anchor BSVD windows on source keyframes.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise bsvd --bsvd-profile c64 --denoise-strength 0.4
```

The accelerator backends require FP16 and refuse unsupported sizes rather
than silently switching devices. `ane` needs at least 96 pixels per side;
`mpsgraph` has a separate padded-size envelope. A first run at a new size
may compile. See [Performance](../PERFORMANCE.md) before selecting a backend
for a multi-stage chain.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
