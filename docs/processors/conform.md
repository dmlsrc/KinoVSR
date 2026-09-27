# Conform: constant frame rate without synthesis

Use `conform` when source timestamps are irregular but an editor or player
needs a constant frame rate (CFR). It duplicates or drops original frames on
a new grid and reports both counts and the largest time shift. It never
synthesizes an in-between image.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--conform-cfr</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>auto</code></td><td>Use the source's nominal rate; an explicit TOML conform stage also defaults to this.</td></tr>
<tr><td><code>RATE</code></td><td>Use a positive rate such as 25 or 30000/1001; each slot takes its nearest original frame.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out --conform-cfr auto
```

Conform runs first in a flag-authored chain so dropped frames do not consume
downstream processing. It needs one frame of lookahead. For synthesized
motion, use [VideoToolbox interpolation](videotoolbox.md) with
`--target-fps` instead; those two flag selectors cannot be combined.

[Usage guide](../USAGE.md)
