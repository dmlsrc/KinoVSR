# MC: motion-compensated temporal denoising

MC warps earlier frames toward the current frame and blends history only
where its content agrees. It needs no denoiser checkpoint; the optional
SpyNet flow route uses bundled SpyNet weights.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--denoise</code></td><td><code>off</code></td><td><code>mc</code></td><td>Add motion-compensated denoising.</td></tr>
<tr><td><code>--mc-strength</code></td><td><code>0.5</code></td><td><code>0..1</code></td><td>Override the denoise slot strength; increase the blend where motion and residual gates admit history.</td></tr>
<tr><td rowspan="2" valign="top"><code>--mc-window</code></td><td rowspan="2" valign="top"><code>0</code></td><td><code>0</code></td><td>Recursive output history; strongest accumulation, potentially longest ghosts.</td></tr>
<tr><td><code>N</code> (positive frame count)</td><td>Causal window of recent input frames; bounds ghost lifetime but costs more flow work.</td></tr>
<tr><td rowspan="3" valign="top"><code>--mc-flow</code></td><td rowspan="3" valign="top"><code>vision</code></td><td><code>vision</code></td><td>Vision revision 1 flow; quality-oriented default.</td></tr>
<tr><td><code>vtme</code></td><td>Media-engine block motion; isolates work from a saturated GPU.</td></tr>
<tr><td><code>spynet</code></td><td>Learned flow on MLX/GPU.</td></tr>
<tr><td><code>--mc-flow-weights</code></td><td><em>bundled SpyNet</em></td><td><code>PATH</code></td><td>Override SpyNet weights when using <code>--mc-flow spynet</code>.</td></tr>
<tr><td rowspan="2" valign="top"><code>--mc-gate</code></td><td rowspan="2" valign="top"><code>smooth</code></td><td><code>smooth</code></td><td>Compare history to a smoothed current frame.</td></tr>
<tr><td><code>curr</code></td><td>Compare history to the raw current frame.</td></tr>
<tr><td><code>--mc-sigma</code></td><td><code>0.06</code></td><td><code>FLOAT</code> (positive luma scale)</td><td>Sets how quickly current/history disagreement closes the blend gate.</td></tr>
<tr><td><code>--mc-clamp</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Clamp warped history to the current frame's local color range.</td></tr>
<tr><td><code>--mc-occlusion</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Reject history where forward/backward flow disagrees.</td></tr>
<tr><td><code>--mc-confidence</code></td><td><code>off</code></td><td><code>on</code> / <code>off</code></td><td>Reduce history weight when flow magnitude is large.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-map</code></td><td rowspan="2" valign="top"><code>constant</code></td><td><code>constant</code></td><td>Use <code>--mc-sigma</code>.</td></tr>
<tr><td><code>auto</code></td><td>Use estimated local sigma in place of <code>--mc-sigma</code>; <code>--mc-strength</code> remains the maximum history blend. See <a href="../USAGE.md#conditioning-maps">map controls</a>.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --denoise mc --mc-flow vision --mc-strength 0.3
```

The three protection flags can be combined. Watch the run's gate-openness
diagnostic: if history is mostly rejected, increasing strength cannot
replace a poor motion estimate. Shared luma/chroma output blends also apply.

[Usage guide](../USAGE.md)
