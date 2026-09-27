# MetalFX: fast spatial upscaling

MetalFX uses the macOS spatial scaler for a single-frame upscale. Its model
is part of the OS, so no downloaded weights or profile names are needed.
It can make edges crisp cheaply, but synthesized texture may crawl in video.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>metalfx</code></td><td>Run the OS-provided spatial scaler.</td></tr>
<tr><td rowspan="3" valign="top"><code>--metalfx-scale</code></td><td rowspan="3" valign="top"><code>2</code></td><td><code>2</code></td><td>Upscale by 2x with no external checkpoint.</td></tr>
<tr><td><code>3</code></td><td>Upscale by 3x with no external checkpoint.</td></tr>
<tr><td><code>4</code></td><td>Upscale by 4x with no external checkpoint.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale metalfx --metalfx-scale 2
```

The stage is spatial only: no temporal state, denoising, or frame-rate
conversion. Preflight rejects an output exceeding the Metal texture side
limit. Judge stochastic textures such as sand, brick, and foliage in motion.

[Usage guide](../USAGE.md) | [VideoToolbox](videotoolbox.md)
