# PVDD: real-noise temporal denoising

PVDD processes a bidirectional frame window and targets real-world video
noise. Its checkpoints are external. Unlike other denoisers, it has no
dry/wet strength control: intensity comes from the checkpoint and, for level
profiles, noise-variance conditioning.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--denoise</code></td><td><code>off</code></td><td><code>pvdd</code></td><td>Add PVDD to the denoise chain.</td></tr>
<tr><td rowspan="6" valign="top"><code>--pvdd-profile</code></td><td rowspan="6" valign="top"><code>pvdd</code></td><td><code>pvdd</code></td><td>Blind real-world sRGB video noise.</td></tr>
<tr><td><code>crvd</code></td><td>Real high-ISO noise.</td></tr>
<tr><td><code>davis</code></td><td>Synthetic additive-noise comparison.</td></tr>
<tr><td><code>pvdd_level</code></td><td>sRGB model with explicit noise-level conditioning.</td></tr>
<tr><td><code>pvdd_raw</code></td><td>Manifest only; unavailable in the current video pipeline, which has no packed Bayer input.</td></tr>
<tr><td><code>pvdd_raw_level</code></td><td>Manifest only; unavailable in the current video pipeline, which has no packed Bayer input.</td></tr>
<tr><td><code>--pvdd-window</code></td><td><code>10</code></td><td><code>N</code> (integer at least 2)</td><td>Increase temporal context at greater cost.</td></tr>
<tr><td><code>--pvdd-trim</code></td><td><code>0</code></td><td><code>N</code> (0 &lt;= N &lt; window/2)</td><td>Discard warm-up frames at joins; 0 uses non-overlapping chunks.</td></tr>
<tr><td rowspan="4" valign="top"><code>--pvdd-noise-preset</code></td><td rowspan="4" valign="top"><code>M</code></td><td><code>S</code></td><td>Variance 0.00069.</td></tr>
<tr><td><code>M</code></td><td>Variance 0.0022.</td></tr>
<tr><td><code>L</code></td><td>Variance 0.0055.</td></tr>
<tr><td><code>off</code></td><td>Disable preset when giving an explicit variance.</td></tr>
<tr><td><code>--pvdd-noise-variance</code></td><td><em>selected preset</em></td><td><code>FLOAT</code> (nonnegative variance)</td><td>Override the preset with variance (sigma squared), not sigma.</td></tr>
<tr><td rowspan="2" valign="top"><code>--pvdd-dtype</code></td><td rowspan="2" valign="top"><code>float16</code></td><td><code>float16</code></td><td>Normal runtime path.</td></tr>
<tr><td><code>float32</code></td><td>Useful for parity checks.</td></tr>
<tr><td><code>--pvdd-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Use the chosen level variance or blind model behavior.</td></tr>
<tr><td><code>auto</code></td><td>With <code>pvdd_level</code>, square estimated sigma into a variance map; the chosen preset or variance is the fallback. See <a href="../USAGE.md#conditioning-maps">map controls</a>.</td></tr>
<tr><td><code>--denoise-strength</code></td><td><code>0.5</code></td><td><code>FLOAT</code> (ignored for PVDD)</td><td>The CLI warns; use noise preset/variance or an automatic map instead.</td></tr>
<tr><td><code>--gop-align</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Use source keyframes instead of ordinary window/trim joins.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise pvdd --pvdd-profile pvdd --gop-align
```

`--noise-map-pulse` also requires a level checkpoint. Shared luma/chroma
output blends apply after PVDD, even though `--denoise-strength` does not.

[Usage guide](../USAGE.md)
