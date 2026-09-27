# Cut detection: reset temporal state at scene changes

A hard cut can leave history from the previous scene in a stateful processor.
`cut_detect` marks cuts for every downstream temporal stage. Use it on edited
footage; leave it off on a known continuous shot.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="4" valign="top"><code>--cut-detect</code></td><td rowspan="4" valign="top"><code>off</code></td><td><code>off</code></td><td>Do not mark cuts.</td></tr>
<tr><td><code>simple</code></td><td>Fast downsampled pixel difference; sensitive to large image changes.</td></tr>
<tr><td><code>hist</code></td><td>Color histogram distance; more tolerant of fast motion.</td></tr>
<tr><td><code>vtme</code></td><td>Media-engine motion trackability; useful with exposure flicker or analog damage.</td></tr>
<tr><td><code>--cut-threshold</code></td><td><code>mode-specific</code></td><td><code>FLOAT</code> (positive)</td><td>Defaults: 0.25 for <code>simple</code>/<code>hist</code>, 0.07 for <code>vtme</code>. Lower catches more cuts and more false positives.</td></tr>
<tr><td><code>--cut-log</code></td><td><em>not set</em></td><td><code>PATH</code></td><td>Write detected source frame indices, one per line.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out --cut-detect vtme --upscale balanced
```

A false positive discards useful temporal context at that frame. The
threshold statistics have different scales, so compare the log and output
before tuning. `vtme` needs no learned weights.

[Usage guide](../USAGE.md)
