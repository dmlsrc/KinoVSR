# STDF: temporal compression cleanup

STDF fuses a seven-frame neighborhood with deformable alignment to clean
video compression damage. It is a luma-oriented deblock stage with bundled
weights. Try it before denoising or enlarging a compressed picture.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--deblock</code></td><td><code>off</code></td><td><code>stdf</code></td><td>Add STDF to the deblock chain.</td></tr>
<tr><td rowspan="2" valign="top"><code>--stdf-profile</code></td><td rowspan="2" valign="top"><code>mfqev2</code></td><td><code>mfqev2</code></td><td>Multi-QP HEVC training domain.</td></tr>
<tr><td><code>vimeo90k</code></td><td>All-Intra QP37 training domain.</td></tr>
<tr><td><code>--stdf-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Override the deblock slot strength; lower can retain more fine texture.</td></tr>
<tr><td><code>--stdf-weights</code></td><td><em>bundled profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--deblock-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Apply the chosen strength everywhere.</td></tr>
<tr><td><code>auto</code></td><td>Multiply the selected strength by an estimated blockiness mask to protect clean detail; see <a href="../USAGE.md#conditioning-maps">map controls</a>.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --deblock stdf --stdf-profile mfqev2 --deblock-strength 0.7
```

A lower strength often preserves facial and fine texture at the cost of
remaining artifacts. Compare short sections before adding another deblocker.

[Usage guide](../USAGE.md)
