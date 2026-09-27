# Deflicker: stabilize static-region coding flicker

Deflicker integrates neighboring samples only where local motion checks
verify static content. Use it for GOP pulses or unstable quantization on
still regions. Moving content passes through. It needs no learned weights.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--deflicker</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>on</code></td><td>Run before deblock and denoise in the default flag chain.</td></tr>
<tr><td><code>off</code></td><td>Leave temporal flicker unchanged.</td></tr>
<tr><td><code>--deflicker-window</code></td><td><code>8</code></td><td><code>N</code> (positive frame count)</td><td>Cover the flicker period; larger is slower and adds equal frame delay.</td></tr>
<tr><td><code>--deflicker-strength</code></td><td><code>1.0</code></td><td><code>0..1</code></td><td>Reduce below 1 for a gentler effect; above 1 overshoots and can invert flicker.</td></tr>
<tr><td><code>--deflicker-band</code></td><td><code>0.1</code></td><td><code>FLOAT</code> (nonnegative luma width)</td><td>Count samples near the temporal median as the same coding state.</td></tr>
<tr><td><code>--deflicker-frac</code></td><td><code>0.5</code></td><td><code>FLOAT</code> (fraction)</td><td>Raise to be conservative; lowering admits weaker consensus and can ghost real changes.</td></tr>
<tr><td><code>--deflicker-max-fix</code></td><td><code>0.25</code></td><td><code>FLOAT</code> (luma delta)</td><td>Refuse larger corrections rather than clamping them; not a strength dial.</td></tr>
<tr><td rowspan="2" valign="top"><code>--deflicker-gop</code></td><td rowspan="2" valign="top"><code>on</code></td><td><code>on</code></td><td>Use source sync markers to recognize single I-frame pumping steps.</td></tr>
<tr><td><code>off</code></td><td>Use temporal evidence without sync-aware rescue.</td></tr>
<tr><td rowspan="2" valign="top"><code>--deflicker-jitter</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>on</code></td><td>Align small global shifts during static verification, at extra cost.</td></tr>
<tr><td><code>off</code></td><td>Require static alignment in the original frame coordinates.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --deflicker on --deflicker-window 8 --deblock stdf
```

For whole-frame exposure changes, use [level](level.md) before deflicker.
Jitter compensation rejects shifts beyond about 3 pixels.

[Usage guide](../USAGE.md)
