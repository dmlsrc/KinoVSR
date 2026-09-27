# BasicVSR++: temporal restoration and 4x upscaling

BasicVSR++ propagates information forward and backward through a frame
window. Its `restore` capability repairs temporal damage at source size;
its `upscale` capability enlarges by 4x. Both need external weights.

## Restore arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="7" valign="top"><code>--restore</code></td><td rowspan="7" valign="top"><code>off</code></td><td><code>decompress_track1</code></td><td>Starting point for changing compression artifacts.</td></tr>
<tr><td><code>decompress_track2</code></td><td>Alternative compression restorers; compare detail and temporal crawl.</td></tr>
<tr><td><code>decompress_track3</code></td><td>Alternative compression restorers; compare detail and temporal crawl.</td></tr>
<tr><td><code>denoise</code></td><td>Temporal noise restoration.</td></tr>
<tr><td><code>deblur_dvd</code></td><td>Motion-deblur domains matching their training clips.</td></tr>
<tr><td><code>deblur_gopro</code></td><td>Motion-deblur domains matching their training clips.</td></tr>
<tr><td><code>PROFILE[,PROFILE...]</code></td><td>Run several profiles left to right, for example <code>decompress_track1,denoise</code>.</td></tr>
<tr><td><code>--restore-strength</code></td><td><code>1.0</code></td><td><code>0..1</code></td><td>0 keeps input; 1 uses the full restored output. A comma list assigns one strength per restore stage.</td></tr>
<tr><td><code>--restore-window</code></td><td><code>14</code></td><td><code>N</code> (positive frame count)</td><td>More frames give more context and use more memory.</td></tr>
<tr><td><code>--restore-trim</code></td><td><code>2</code></td><td><code>N</code> (nonnegative frame count)</td><td>Discard transient frames at window joins; window must exceed twice the trim.</td></tr>
<tr><td rowspan="3" valign="top"><code>--restore-flow</code></td><td rowspan="3" valign="top"><code>spynet</code></td><td><code>spynet</code></td><td>Trained flow network.</td></tr>
<tr><td><code>vision</code></td><td>Vision optical flow revision 1.</td></tr>
<tr><td><code>zero</code></td><td>No motion alignment; useful for diagnosing flow artifacts.</td></tr>
<tr><td><code>--restore-ensemble</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Average flipped/rotated predictions to reduce orientation artifacts; about 8x compute.</td></tr>
<tr><td><code>--restore-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
</tbody>
</table>

## Upscale arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>basicvsrpp</code></td><td>Run the 4x BasicVSR++ capability.</td></tr>
<tr><td rowspan="4" valign="top"><code>--basicvsrpp-profile</code></td><td rowspan="4" valign="top"><code>vimeo90k_bd</code></td><td><code>vimeo90k_bd</code></td><td>Smaller model trained with blur downsampling; useful first choice for native video.</td></tr>
<tr><td><code>vimeo90k_bi</code></td><td>Bicubic-trained models; often softer on other degradations.</td></tr>
<tr><td><code>reds4</code></td><td>Bicubic-trained models; often softer on other degradations.</td></tr>
<tr><td><code>ntire_vsr</code></td><td>Larger, sharper model with higher memory demand.</td></tr>
<tr><td><code>--basicvsrpp-window</code></td><td><code>14</code></td><td><code>N</code> (positive frame count)</td><td>Increase context at a memory/compute cost.</td></tr>
<tr><td><code>--basicvsrpp-trim</code></td><td><code>2</code></td><td><code>N</code> (nonnegative frame count)</td><td>Discard transient frames at joins; window must exceed twice the trim.</td></tr>
<tr><td rowspan="3" valign="top"><code>--basicvsrpp-flow</code></td><td rowspan="3" valign="top"><code>spynet</code></td><td><code>spynet</code></td><td>Trained flow network.</td></tr>
<tr><td><code>vision</code></td><td>Vision optical flow revision 1.</td></tr>
<tr><td><code>zero</code></td><td>Disable motion alignment as a diagnostic control.</td></tr>
<tr><td><code>--basicvsrpp-history-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code> (nonnegative)</td><td>1 is reference strength; 0 disables temporal propagation.</td></tr>
<tr><td rowspan="2" valign="top"><code>--basicvsrpp-history-gate</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Reference behavior.</td></tr>
<tr><td><code>improve</code></td><td>Admit aligned history only when its photometric match improves.</td></tr>
<tr><td><code>--basicvsrpp-ensemble</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Average flipped/rotated predictions; about 8x compute.</td></tr>
<tr><td><code>--basicvsrpp-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align \
  --restore decompress_track1 --upscale basicvsrpp
```

`--gop-align` uses source keyframes instead of the two capabilities' individual
window/trim joins. Check moving detail and flat regions before using ensemble
on a whole clip.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
