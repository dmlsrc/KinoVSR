# NAFNet: per-frame deblur, denoise, or restoration

NAFNet applies a single-image residual late in the default preprocess chain.
A strong correction can flicker on video, so inspect motion as well as still
frames. All checkpoints are external.

## Arguments

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="6" valign="top"><code>--nafnet</code></td><td rowspan="6" valign="top"><code>off</code></td><td><code>gopro</code></td><td>Motion deblur, width 64.</td></tr>
<tr><td><code>gopro32</code></td><td>Smaller motion-deblur model, width 32.</td></tr>
<tr><td><code>sidd</code></td><td>Real-noise denoise, width 64.</td></tr>
<tr><td><code>sidd32</code></td><td>Smaller real-noise denoiser, width 32.</td></tr>
<tr><td><code>reds</code></td><td>General restoration.</td></tr>
<tr><td><code>off</code></td><td>Omit NAFNet.</td></tr>
<tr><td><code>--nafnet-strength</code></td><td><code>1.0</code></td><td><code>FLOAT</code> (nonnegative)</td><td>Lower for a light video pass; above 1 overdrives the residual.</td></tr>
<tr><td><code>--nafnet-weights</code></td><td><em>selected profile</em></td><td><code>PATH</code></td><td>Use an explicit <code>.safetensors</code> file.</td></tr>
<tr><td rowspan="3" valign="top"><code>--nafnet-pool</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Follow the selected checkpoint's trained mode.</td></tr>
<tr><td><code>local</code></td><td>Force local TLSC pooling.</td></tr>
<tr><td><code>global</code></td><td>Force global pooling.</td></tr>
<tr><td rowspan="7" valign="top"><code>--nafnet-guard</code></td><td rowspan="7" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Use <code>reject</code> for GoPro profiles, <code>off</code> otherwise.</td></tr>
<tr><td><code>off</code></td><td>Use the raw predicted residual.</td></tr>
<tr><td><code>reject</code></td><td>Pass through risky frames, lock out after repeated failures, and re-probe later.</td></tr>
<tr><td><code>residual</code></td><td>Apply a local soft knee to unusually large residuals.</td></tr>
<tr><td><code>fast</code></td><td>Use single-pass residual attenuation.</td></tr>
<tr><td><code>control</code></td><td>Rerun risky regions on a smoothed control input.</td></tr>
<tr><td><code>control-source</code></td><td>Predict from a stable control input, then add the residual to the original.</td></tr>
<tr><td><code>--nafnet-guard-threshold</code></td><td><code>0.12</code></td><td><code>FLOAT</code></td><td>Lower catches more risky regions but may suppress legitimate deblur.</td></tr>
<tr><td><code>--nafnet-guard-fast-fraction</code></td><td><code>0.85</code></td><td><code>FLOAT</code> (frame-risk fraction)</td><td>Switch from regional control to control-source when risk covers this fraction.</td></tr>
<tr><td><code>--nafnet-guard-lockout</code></td><td><code>48</code></td><td><code>N</code> (nonnegative frame count)</td><td>Wait this long between re-probes; 0 remains locked for the clip.</td></tr>
<tr><td><code>--nafnet-guard-ramp</code></td><td><code>12</code></td><td><code>N</code> (nonnegative frame count)</td><td>Smooth the return of restoration; 0 switches abruptly.</td></tr>
<tr><td><code>--nafnet-guard-fall</code></td><td><em>derived from ramp</em></td><td><code>N</code> (nonnegative frame count)</td><td>0 cuts abruptly; omitted value is ramp/4, at least 2.</td></tr>
</tbody>
</table>

## Example

```bash
kinovsr run --video in.mp4 --output-dir out \
  --nafnet gopro --nafnet-strength 0.6
```

Start with the profile and automatic guard. Pooling overrides and guard
thresholds change the model's behavior, so compare representative motion
before applying them to a full clip.

[Usage guide](../USAGE.md) | [Profiles and weights](../PROCESSORS.md)
