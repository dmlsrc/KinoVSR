# Spatial: quick per-frame denoising

Spatial uses Core Image noise reduction on each frame independently. It has
no model weights, future-frame delay, or motion state. Use it for light
cleanup when temporal consistency is less important than simplicity.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--denoise</code></td><td><code>off</code></td><td><code>spatial</code></td><td>Add per-frame Core Image noise reduction.</td></tr>
<tr><td><code>--spatial-strength</code></td><td><code>0.5</code></td><td><code>0..1</code></td><td>Override the denoise slot strength; increase Core Image noise reduction.</td></tr>
<tr><td><code>--denoise-luma-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Lower to retain more original luma detail.</td></tr>
<tr><td><code>--denoise-chroma-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Lower to retain more original chroma.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise spatial --spatial-strength 0.3
```

The automatic noise map has no input in this processor. A flag run using
only spatial denoising rejects `--noise-map auto`; in a mixed chain,
the map applies only to compatible denoisers. Compare moving texture:
independent per-frame filtering can vary from frame to frame.

[Usage guide](../USAGE.md) | [MC temporal denoise](mc.md)
