# Crop: bars and reframing

Crop bars or select a display-aspect window before restoration. The output
contains only the remaining picture; nothing is stretched.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--crop-bars</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>auto</code></td><td>Detect persistent extreme bars in sampled frames, up to 45% per edge.</td></tr>
<tr><td><code>T,B,L,R</code></td><td>Remove explicit top, bottom, left, and right pixel counts.</td></tr>
<tr><td><code>--crop-aspect</code></td><td><code>off</code></td><td><code>W:H</code></td><td>Keep the largest even-size window with this on-screen aspect after bars are removed.</td></tr>
<tr><td rowspan="9" valign="top"><code>--crop-anchor</code></td><td rowspan="9" valign="top"><code>center</code></td><td><code>center</code></td><td>Center the selected window.</td></tr>
<tr><td><code>top-left</code></td><td>Anchor the window at top left.</td></tr>
<tr><td><code>top</code></td><td>Anchor the window at top.</td></tr>
<tr><td><code>top-right</code></td><td>Anchor the window at top right.</td></tr>
<tr><td><code>left</code></td><td>Anchor the window at left.</td></tr>
<tr><td><code>right</code></td><td>Anchor the window at right.</td></tr>
<tr><td><code>bottom-left</code></td><td>Anchor the window at bottom left.</td></tr>
<tr><td><code>bottom</code></td><td>Anchor the window at bottom.</td></tr>
<tr><td><code>bottom-right</code></td><td>Anchor the window at bottom right.</td></tr>
<tr><td><code>--crop-offset</code></td><td><code>0,0</code></td><td><code>DX,DY</code></td><td>Move right/down from the anchor; the window remains inside the frame.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out --crop-bars auto --upscale balanced
```

`--crop-aspect` accounts for anamorphic pixel aspect, so `16:9` means 16:9
on screen. [Sanitize edges](sanitize_edges.md) is better for a few damaged
border rows when you need to retain the frame dimensions.

[Usage guide](../USAGE.md) | [Square pixels](square_pixels.md)
