# FBCNN: per-frame JPEG-style artifact removal

FBCNN uses a single-image color model to remove blocks and ringing. It is a
deblock stage before denoising and upscaling. The `color` checkpoint is
external. Quality factor (QF) conditions the network separately from the
final output blend.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--deblock</code></td><td><code>off</code></td><td><code>fbcnn</code></td><td>Add FBCNN to the deblock chain.</td></tr>
<tr><td rowspan="3" valign="top"><code>--fbcnn-quality</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Estimate quantization per tile over a rolling frame window.</td></tr>
<tr><td><code>1..100</code></td><td>Pin a global QF. Lower means heavier compression and stronger cleanup.</td></tr>
<tr><td><code>blind</code></td><td>Use the model's own estimate; it can under-treat loop-filtered H.264/HEVC.</td></tr>
<tr><td><code>--fbcnn-quality-fallback</code></td><td><code>50</code></td><td><code>1..100</code></td><td>Use where <code>auto</code> finds no quantization evidence; lower for known heavy compression.</td></tr>
<tr><td><code>--fbcnn-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Override the deblock slot strength; lower retains texture, while above 1 can ring.</td></tr>
<tr><td rowspan="2" valign="top"><code>--fbcnn-gop</code></td><td rowspan="2" valign="top"><code>on</code></td><td><code>on</code></td><td>Align automatic QF refreshes with source sync samples.</td></tr>
<tr><td><code>off</code></td><td>Refresh on the counter cadence only.</td></tr>
<tr><td><code>--fbcnn-weights</code></td><td><code>color</code> profile</td><td><code>PATH</code></td><td>Use an explicit color <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--deblock-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Apply the chosen strength everywhere.</td></tr>
<tr><td><code>auto</code></td><td>Multiply the selected output-blend strength by an estimated blockiness mask; see <a href="../USAGE.md#conditioning-maps">map controls</a>.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --deblock fbcnn --fbcnn-quality auto --fbcnn-strength 0.7
```

For compressed video, compare `auto` with a fixed QF on a short section.
FBCNN runs frame by frame, so inspect for changing treatment across frames.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
