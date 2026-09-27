# Level: whole-frame exposure stabilization

Level matches each frame's luma distribution to a centered temporal
reference. It handles auto-exposure hunting and whole-picture brightness
pumping, and runs first among preprocess stages.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--level</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>hist</code></td><td>Match each frame's luma histogram to its temporal reference.</td></tr>
<tr><td><code>off</code></td><td>Leave exposure unchanged.</td></tr>
<tr><td><code>--level-window</code></td><td><code>5</code></td><td><code>N</code> (positive frame count)</td><td>Cover the pumping cadence; adds equal frame delay. Slower trends are followed.</td></tr>
<tr><td><code>--level-deadband</code></td><td><code>0.003</code></td><td><code>FLOAT</code> (0..1 luma shift)</td><td>Below this, pass a frame unchanged. Raise if normal scene motion is being touched.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --level hist --level-window 5 --denoise mc
```

The end-of-run pumping meter reports how often the leveler corrected a
frame. Slow lighting changes and fades are followed; clipped highlights
cannot be recovered. For flicker only on static patches, add
[deflicker](deflicker.md) after level.

[Usage guide](../USAGE.md)
