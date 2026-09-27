# RealBasicVSR: recurrent real-world 4x upscaling

RealBasicVSR cleans frames, propagates aligned features through a
bidirectional window, then reconstructs at 4x. It can add detail to
degraded footage, but inspect moving texture for lattice, crawl, or ghosts.
The `x4` checkpoint is external.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>realbasicvsr</code></td><td>Run recurrent real-world 4x upscaling.</td></tr>
<tr><td><code>--realbasicvsr-profile</code></td><td><code>x4</code></td><td><code>x4</code></td><td>The released real-world 4x model.</td></tr>
<tr><td><code>--realbasicvsr-weights</code></td><td><code>x4</code> profile</td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td><code>--realbasicvsr-window</code></td><td><code>14</code></td><td><code>N</code> (positive frame count)</td><td>More frames give more context and use more memory.</td></tr>
<tr><td><code>--realbasicvsr-trim</code></td><td><code>0</code></td><td><code>N</code> (nonnegative frame count)</td><td>Discard warm-up frames at joins; 0 matches non-overlapping reference chunks.</td></tr>
<tr><td><code>--realbasicvsr-clean-threshold</code></td><td><code>5.0</code></td><td><code>FLOAT</code> (0..255 scale)</td><td>Stop early when cleanup is small; 255 forces one pass.</td></tr>
<tr><td><code>--realbasicvsr-clean-iters</code></td><td><code>3</code></td><td><code>N</code> (nonnegative integer)</td><td>Maximum cleaning passes before recurrent propagation.</td></tr>
<tr><td><code>--realbasicvsr-residual-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code></td><td>Scale the residual added to the bilinear base; lower reduces aggressive texture.</td></tr>
<tr><td rowspan="3" valign="top"><code>--realbasicvsr-flow</code></td><td rowspan="3" valign="top"><code>spynet</code></td><td><code>spynet</code></td><td>Trained flow network.</td></tr>
<tr><td><code>vision</code></td><td>Vision revision 1 optical flow.</td></tr>
<tr><td><code>zero</code></td><td>Disable motion alignment for diagnosis.</td></tr>
<tr><td><code>--realbasicvsr-flow-consistency</code></td><td><code>0</code></td><td><code>0..1</code></td><td>Down-weight history where forward/backward flow disagrees.</td></tr>
<tr><td><code>--realbasicvsr-history-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code> (nonnegative)</td><td>1 is reference strength; 0 disables temporal propagation.</td></tr>
<tr><td rowspan="2" valign="top"><code>--realbasicvsr-history-gate</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Reference behavior.</td></tr>
<tr><td><code>improve</code></td><td>Admit aligned history only where its photometric match improves.</td></tr>
<tr><td><code>--gop-align</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Use source keyframes instead of ordinary window/trim joins.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align \
  --upscale realbasicvsr --realbasicvsr-residual-strength 0.75
```

Flow and history controls alter the reference behavior. Start with the
trained defaults and change one control at a time.

[Usage guide](../USAGE.md) | [Performance](../PERFORMANCE.md)
