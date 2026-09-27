# TOFlow: temporal denoise, deblock, and 4x upscale

TOFlow aligns a seven-frame neighborhood with task-oriented flow. Bundled
converted checkpoints cover denoise, deblock, and 4x upscale; these are
separate capabilities. The manifest also lists an `interp` artifact, but the
current processor factory has no TOFlow interpolation capability.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--denoise</code></td><td><code>off</code></td><td><code>toflow</code></td><td>Run the seven-frame denoise graph.</td></tr>
<tr><td><code>--deblock</code></td><td><code>off</code></td><td><code>toflow</code></td><td>Run the seven-frame deblock graph.</td></tr>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>toflow</code></td><td>Run the distinct bundled 4x super-resolution graph.</td></tr>
<tr><td rowspan="2" valign="top"><code>--toflow-profile</code></td><td><code>denoise</code></td><td><code>denoise</code></td><td>Default in the denoise slot; seven-frame denoise model.</td></tr>
<tr><td><code>deblock</code></td><td><code>deblock</code></td><td>Default in the deblock slot; seven-frame compression-cleanup model.</td></tr>
<tr><td rowspan="2" valign="top"><code>--toflow-strength</code></td><td><code>0.5</code></td><td><code>FLOAT</code> (nonnegative)</td><td>Default in the denoise slot. 0 passes input through; 1 uses full output; above 1 extrapolates.</td></tr>
<tr><td><code>1.0</code></td><td><code>FLOAT</code> (nonnegative)</td><td>Default in the deblock slot; the same output blend applies.</td></tr>
<tr><td><code>--toflow-passes</code></td><td><code>1</code></td><td><code>N</code> (positive integer)</td><td>Repeat the model within one stage while reusing computed flow.</td></tr>
<tr><td rowspan="3" valign="top"><code>--toflow-flow-scale</code></td><td rowspan="3" valign="top"><code>full</code></td><td><code>full</code></td><td>Faithful network computation.</td></tr>
<tr><td><code>half</code></td><td>Less flow work, with potential alignment loss.</td></tr>
<tr><td><code>quarter</code></td><td>Less flow work, with potential alignment loss.</td></tr>
<tr><td rowspan="2" valign="top"><code>--toflow-dtype</code></td><td rowspan="2" valign="top"><code>float32</code></td><td><code>float32</code></td><td>Use parity-oriented arithmetic for cleanup.</td></tr>
<tr><td><code>float16</code></td><td>Use faster, lower-precision cleanup arithmetic.</td></tr>
<tr><td><code>--toflow-weights</code></td><td><em>selected cleanup profile</em></td><td><code>PATH</code></td><td>Override the denoise/deblock safetensors checkpoint.</td></tr>
<tr><td><code>--toflow-graph</code></td><td><em>selected cleanup profile</em></td><td><code>PATH</code></td><td>Override the denoise/deblock graph JSON.</td></tr>
<tr><td><code>--toflow-sr-weights</code></td><td>bundled <code>sr</code></td><td><code>PATH</code></td><td>Override the separate 4x upscale checkpoint.</td></tr>
<tr><td><code>--toflow-sr-graph</code></td><td>bundled <code>sr</code></td><td><code>PATH</code></td><td>Override the separate 4x upscale graph JSON.</td></tr>
<tr><td rowspan="2" valign="top"><code>--toflow-sr-dtype</code></td><td rowspan="2" valign="top"><code>float32</code></td><td><code>float32</code></td><td>Use parity-oriented arithmetic for upscaling.</td></tr>
<tr><td><code>float16</code></td><td>Use faster, lower-precision upscale arithmetic.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --deblock toflow --toflow-strength 0.7 --upscale toflow
```

Automatic noise/blockiness maps do not feed TOFlow's blind graphs. Use
VideoToolbox `--target-fps` for frame-rate conversion.

[Usage guide](../USAGE.md) | [VideoToolbox interpolation](videotoolbox.md)
