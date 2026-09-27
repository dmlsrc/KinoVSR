# VideoToolbox: native upscale and frame-rate conversion

VideoToolbox offers spatial upscaling and motion-based frame interpolation
without downloaded weights. The two are separate capabilities; the flag CLI
runs spatial upscale before interpolation when both are selected.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="3" valign="top"><code>--upscale</code></td><td rowspan="3" valign="top"><code>none</code></td><td><code>fast</code></td><td>2x per-frame low-latency mode; input up to 960x960.</td></tr>
<tr><td><code>balanced</code></td><td>4x temporal mode with previous-frame context; input up to 1920x1080.</td></tr>
<tr><td><code>image</code></td><td>4x per-frame image mode; input up to 1920x1080.</td></tr>
<tr><td rowspan="2" valign="top"><code>--vt-sr-flow</code></td><td rowspan="2" valign="top"><code>internal</code></td><td><code>internal</code></td><td>Let VideoToolbox compute its own motion field.</td></tr>
<tr><td><code>vision</code></td><td>Feed explicit Vision revision 1 flow. Invalid for <code>fast</code>/<code>image</code>.</td></tr>
<tr><td><code>--target-fps</code></td><td><code>off</code></td><td><code>FPS</code> (positive frame rate)</td><td>Carry source timing when off; synthesize frames when increasing cadence or reduce cadence.</td></tr>
<tr><td rowspan="2" valign="top"><code>--temporal-mode</code></td><td rowspan="2" valign="top"><code>normal</code></td><td><code>normal</code></td><td>Standard frame-rate conversion.</td></tr>
<tr><td><code>high</code></td><td>Higher-compute quality-prioritized mode; compare on the actual clip.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --upscale balanced --target-fps 60 --temporal-mode normal
```

A hard cut resets `balanced`'s previous-frame chain. On macOS 27, KinoVSR
checks whether interpolation at the chosen geometry needs edge padding for
correct motion rendering. For CFR output without synthesis, use
[conform](conform.md) instead. Inspect motion, fine texture, and borders in
video: native profiles differ in rendering style as well as scale.

[Usage guide](../USAGE.md) | [Performance](../PERFORMANCE.md)
