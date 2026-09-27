# Sanitize edges: keep border junk out of models

Use `sanitize_edges` for thin capture garbage or synthetic edge rows that
make restoration models invent texture. It fills those pixels from the
interior before processing; the default flag route restores the original
border afterward without changing frame dimensions.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--sanitize-edges</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>auto</code></td><td>Detect persistent anomalous edges, up to 8 pixels per side.</td></tr>
<tr><td><code>T,B,L,R</code></td><td>Give explicit top, bottom, left, and right pixel counts.</td></tr>
<tr><td rowspan="3" valign="top"><code>--sanitize-edges-fill</code></td><td rowspan="3" valign="top"><code>restore</code></td><td><code>restore</code></td><td>Put the quiet source border back with a feathered join.</td></tr>
<tr><td><code>extend</code></td><td>Keep the replicated interior pixels; the edge may shimmer with picture motion.</td></tr>
<tr><td><code>trim</code></td><td>Crop the affected pixels and change output dimensions before aspect cropping.</td></tr>
<tr><td><code>--sanitize-edges-feather</code></td><td><code>2</code></td><td><code>N</code> (nonnegative source pixels)</td><td>Crossfade into processed content; 0 makes a hard seam.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --sanitize-edges auto --upscale balanced
```

Automatic sanitation leaves thick letterbox bars alone; use [crop](crop.md)
for those. An explicit TOML `sanitize_edges` stage defaults `fill` to `extend`,
so set `fill = "restore"` there to match the flag example.

[Usage guide](../USAGE.md)
