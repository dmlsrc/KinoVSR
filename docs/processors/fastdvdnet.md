# FastDVDnet: learned five-frame denoising

FastDVDnet uses a causal five-frame CNN to remove changing noise. Both
checkpoints are bundled. Its noise strength is a model-conditioning sigma,
not a simple output blend.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--denoise</code></td><td><code>off</code></td><td><code>fastdvdnet</code></td><td>Add FastDVDnet to the denoise chain.</td></tr>
<tr><td rowspan="2" valign="top"><code>--fastdvdnet-profile</code></td><td rowspan="2" valign="top"><code>clipped</code></td><td><code>clipped</code></td><td>Trained with clipped noise; good first choice for moderate real footage noise.</td></tr>
<tr><td><code>standard</code></td><td>Plain additive-white-noise model for closer matching sources.</td></tr>
<tr><td><code>--fastdvdnet-strength</code></td><td><code>0.5</code></td><td><code>0..1</code></td><td>Override the denoise slot strength; maps to trained sigma 5-55/255.</td></tr>
<tr><td><code>--fastdvdnet-weights</code></td><td><em>bundled profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Use the strength-derived sigma everywhere.</td></tr>
<tr><td><code>auto</code></td><td>Use estimated per-pixel sigma in place of strength-derived sigma; strength remains the fallback. See <a href="../USAGE.md#conditioning-maps">map controls</a>.</td></tr>
<tr><td><code>--denoise-luma-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Lower it to retain more original luma texture.</td></tr>
<tr><td><code>--denoise-chroma-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Lower it to retain more original chroma.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise fastdvdnet --fastdvdnet-profile clipped --denoise-strength 0.35
```

Deblock strongly compressed footage first, so coding edges are not treated
as noise. Compare moving texture at normal playback speed.

[Usage guide](../USAGE.md)
