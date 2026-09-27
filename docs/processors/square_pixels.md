# Square pixels: correct anamorphic storage

This stage horizontally resamples non-square-pixel video to 1:1 pixel
aspect before restoration. It helps players and editors that ignore pixel
aspect metadata. Already square-pixel sources are unchanged.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--square-pixels</code></td><td rowspan="2" valign="top"><code>off</code></td><td><code>off</code></td><td>Preserve the source pixel aspect without resampling.</td></tr>
<tr><td><code>on</code></td><td>Resample horizontally with Lanczos-3 at source resolution and tag output as 1:1.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --square-pixels --upscale balanced
```

There are no weights, profiles, or strength dials. To select a different
picture area or display aspect, use [crop](crop.md).

[Usage guide](../USAGE.md)
