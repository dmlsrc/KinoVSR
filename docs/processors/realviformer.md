# RealViformer: causal recurrent 4x upscaling

RealViformer streams frame by frame while carrying aligned history. Its
external `x4` checkpoint can lock texture over time, but wrong motion or a
long static shot can leave etched patterns. Inspect a representative video
section, not just a still frame.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--upscale</code></td><td><code>none</code></td><td><code>realviformer</code></td><td>Run causal recurrent 4x upscaling.</td></tr>
<tr><td><code>--realviformer-profile</code></td><td><code>x4</code></td><td><code>x4</code></td><td>The released real-world 4x model.</td></tr>
<tr><td><code>--realviformer-weights</code></td><td><code>x4</code> profile</td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="2" valign="top"><code>--realviformer-dtype</code></td><td rowspan="2" valign="top"><code>float16</code></td><td><code>float16</code></td><td>Normal fast MLX path.</td></tr>
<tr><td><code>float32</code></td><td>Closer numerical comparison to the reference.</td></tr>
<tr><td rowspan="2" valign="top"><code>--realviformer-window</code></td><td rowspan="2" valign="top"><code>100</code></td><td><code>N</code> (positive frame count)</td><td>Reset history every N frames; shorter limits long-run texture lock but adds joins.</td></tr>
<tr><td><code>0</code></td><td>Never reset, allowing unbounded recurrent history.</td></tr>
<tr><td rowspan="3" valign="top"><code>--realviformer-flow</code></td><td rowspan="3" valign="top"><code>spynet</code></td><td><code>spynet</code></td><td>Trained flow network.</td></tr>
<tr><td><code>vision</code></td><td>Vision revision 1 optical flow.</td></tr>
<tr><td><code>zero</code></td><td>Disable motion alignment as a diagnostic control.</td></tr>
<tr><td><code>--realviformer-history-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code> (nonnegative)</td><td>1 is reference strength; 0 removes temporal history.</td></tr>
<tr><td rowspan="3" valign="top"><code>--realviformer-history-gate</code></td><td rowspan="3" valign="top"><code>off</code></td><td><code>off</code></td><td>Reference merge behavior.</td></tr>
<tr><td><code>improve</code></td><td>Admit history only where flow-warped RGB improves the match.</td></tr>
<tr><td><code>holistic</code></td><td>Opt-in combined risk policy with confidence, memory, and cleanup controls.</td></tr>
<tr><td><code>--realviformer-history-cleanup</code></td><td><code>0.25</code></td><td><code>0..1</code></td><td>Maximum blend toward local smoothing in risky regions.</td></tr>
<tr><td><code>--realviformer-history-gate-drop</code></td><td><code>0.85</code></td><td><code>0..1</code></td><td>Maximum fraction of the history gate removed in risky regions.</td></tr>
<tr><td><code>--realviformer-history-risk-decay</code></td><td><code>0.8</code></td><td><code>FLOAT</code> (0 &lt;= value &lt; 1)</td><td>Decay the flow-warped risk memory between frames.</td></tr>
<tr><td><code>--realviformer-history-static-cap</code></td><td><code>0</code></td><td><code>0..1</code></td><td>Cap admitted confidence in perfectly static regions.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale realviformer --realviformer-window 100
```

The last four controls apply only to `holistic`. Cut detection resets the
causal state at scene boundaries.

[Usage guide](../USAGE.md)
