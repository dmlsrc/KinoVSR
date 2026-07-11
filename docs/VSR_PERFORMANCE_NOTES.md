# VSR Harness Performance Notes

Findings, gotchas, and methodology from the 2026-07 optimization campaign over
the `kinovsr/` processors (deblockers, denoisers, restorers, and learned
upscalers driven by the `kinovsr` CLI). Everything here was measured on
an M1 Max (64 GB) with MLX, fp16-first, MLX buffer cache capped at 1 GB.

The single most important lesson: **"compute-bound" must be established by
kernel-path analysis, not FLOP counting.** Four separate "this net is at the
hardware ceiling" verdicts were overturned during this campaign by looking at
which Metal kernel each op actually dispatches to, and why.

---

## 1. The MLX conv dispatch gates (read this first)

`mlx/backend/metal/conv.cpp` routes every `mx.conv2d` call through a decision
tree. Ops that miss a gate fall to a *general* kernel that is silently 2-4x
slower. The gates, as of the MLX version in use:

| path | conditions | character |
| --- | --- | --- |
| depthwise | groups: C_per_group==1, O_per_group==1, C==O, C%16==0, k<=7, stride<=2 | fixed 8x8x4 tile, scalar MACs, no MMA |
| grouped implicit GEMM | (C_per_group<=4 or %16==0) and (O_per_group<=16 or %16==0) | fine for small group counts |
| winograd | 3x3, stride 1, no dilation/flip, C%32==0, O%32==0, **C+O>=256**, input >= 4096 px | fastest for mid-size 3x3 convs |
| implicit GEMM (specialized) | (C<=4 or C%16==0) and (O<=16 or O%16==0) | the good default |
| implicit GEMM (general) | everything else | the 2-4x-slower fallback |

Practical rules:

- **Audit every conv's (C, O) against these gates when porting a net.** A `%16`
  miss is invisible in the code and costs 2-4x. Found and fixed in this codebase:
  STDF `offset_mask` (32->189, its FLOP-heaviest conv, 1.9x), STDF `in_conv`
  (7->32, 2.0x), FastDVDnet `inc0` (30 outputs/group on the explicit-grouped
  fallback, 4.1x), SPyNet's first conv (8->32 7x7, 2.5x), BasicVSR++
  `conv_offset.0` (196->64, 1.45x), RealBasicVSR backbone `main.0` (67->64, 1.55x).
- **Audit runtime concat widths, not just weight files.** The last two misses
  above only exist at runtime (`concat(cond, flow1, flow2)` = 196 channels); a
  weights-only shape sweep cannot see them.
- **The fix is zero-padding, and it is exact.** Pad weight columns/filters with
  zeros to the next %16 boundary (and append matching zero channels to the input,
  or slice junk output channels off). Zero weights contribute nothing; only the
  kernel changes. Most of these fixes measured bit-exact end to end.
- **Padding is not automatically a win.** FastDVDnet `inc3` (90->32) on the
  general path *beat* the specialized path at 96->32 (thin-N inefficiency), and
  padding toward the winograd gate costs real FLOPs (zero-padded weights still
  multiply) -- 64->64 at C+O=128 cannot be pushed to 256 profitably. Measure the
  exact shapes before and after; never assume.

## 2. Kernel-shape pathologies (and their exact-math fixes)

### Depthwise conv: `mx.conv2d(groups=C)` collapses at many-channel/small-spatial

Measured 83x over the memory-bandwidth floor at 1024ch / 60x108 (the NAFNet deep
stage). The dedicated depthwise kernel uses a fixed 8x8x4 tile with a
threadgroup-memory halo and scalar 9-tap MACs -- no MMA (depthwise has no
K-contraction to feed the matrix units). At large-spatial/few-channel scales it
is only 1-7x over floor (acceptable).

**Fix:** a manual 9-tap shift-and-add (`sum over (i,j) of
xp[:, i:i+H, j:j+W, :] * w[:, i, j, 0]`) is 8.2x faster at the pathological
scale, 1.0x elsewhere, so it is safe to apply unconditionally. See
`kinovsr/processors/nafnet/net.py:_depthwise3x3`. Whole-net NAFNet: 1.37-1.44x.
Output shift ~55 dB PSNR (fp32 summation-order compounding through 36 residual
blocks; the op itself matches conv2d to 1e-6).

### `mx.fast.layer_norm` / `rms_norm` are transformer-shaped

One threadgroup per normalized row, sized to the normalized axis. For a conv
channel-norm (small C = 32-512, many N*H*W rows) the threadgroup underfills
(~8 of 32 lanes at C=64) across ~100k tiny threadgroups: **2.2-2.5x slower**
than a hand-rolled `mx.mean`/`mx.rsqrt` reduction. The penalty is shape-bound,
not dtype-bound (the kernel accumulates in fp32 regardless). Reserve
`mx.fast.*norm` for transformer-width axes; keep manual reductions for NHWC
channel norms (see `kinovsr/processors/nafnet/net.py:_layernorm`). Re-verified on
RealPLKSR (C=64): manual is 1.89x faster than `mx.fast.layer_norm`.

### GroupNorm: reduce the contiguous spatial axes first

A channel-first NHWC GroupNorm (reshape to `(N,H,W,g,cg)`, normalize over
`(H,W,cg)` per group) is tempting to write as one `mx.mean` over the strided
axes `(1,2,4)`. That strided reduction is **4.5x slower** than reducing the
contiguous spatial axes `(1,2)` first (keeping `(g,cg)`) and then the small
channel-group axis -- and on RealPLKSR's 4x GroupNorm variant the strided form
was the whole net's #1 op (173 ms/28-blocks at 288x352, more than the 17x17
partial-large-kernel conv). Fixing it took the whole 4x forward 597 -> 434
ms/frame (1.38x). The reduction stays fp32 (load-bearing: refine-out activations
reach ~370, so both `mean(x)` and `mean(x^2)` overflow an fp16 accumulator);
use the two-pass form (mu, then `(x-mu)^2`) rather than `E[x^2]-E[x]^2` so the
variance cannot go negative from cancellation. See
`kinovsr/processors/realplksr/net.py:_groupnorm`.

Corollary from the same audit: a numerically stable softplus Mish
(`max(x,0)+log1p(exp(-|x|))`, `exp(-|x|)` <= 1 so no fp16 overflow) needs no
fp32 island -- it is speed-neutral in the compiled/fused graph and within
~4e-3 of fp32. Only the norm reductions genuinely need fp32.

### Winograd collapses at large spatial extents

The same 64->256 3x3 conv runs at 10.2 TF/s-effective at 480x854 but 3.3 TF/s at
960x1708. This is why BasicVSR++'s `upsample2` (conv at 2x resolution) dominates
its reconstruction tail. Two exact reformulations were tested and **rejected**:
2x2 spatial tiling with halos (no recovery; the collapse tracks total working
set, not dispatch size) and a subpixel-conv rewrite of shuffle-then-conv as four
phase convs at LR (0.98x: the phase decomposition inherently carries 1.78x the
FLOPs, exactly canceling the better GEMM rate). The tail is at its practical
floor.

### Dense blocks (RRDB / DenseNet-style): restack weights by input segment

Conv-over-concat equals the sum of per-segment convs, and in a dense block every
conv's x / x1 / x2 ... segments convolve the *same* tensors. Restacking the
weight slices by produced tensor (done once at load, bit-identical values) turns
five thin concat-fed convs into one fat conv per produced tensor: the x-stack
(64 -> 224) crosses the winograd gate, and the incremental concats (4 full
copies per block x 69 blocks) disappear. RRDBNet (bsrgan / x4plus / esrgan /
realesrnet / bsrnet / anime / x2plus): **1.54x**, output parity 65.5 dB. The
recombination sums must run in fp32: the split rounds each partial conv output
to fp16 where the original GEMM accumulated all of K in fp32. See
`kinovsr/processors/realesrgan/net.py:_restack_rdb_weights` and `_rdb`.

### Window attention is intrinsically expensive on M1

ESC-Real's 32x32-window attention (405 windows x 4 heads x 1024 tokens x head_dim
16 at 480p) costs ~230 ms per layer and ~60% of each block, and BOTH formulations
land there: `mx.fast.scaled_dot_product_attention` with a dense additive mask at
head_dim 16 takes a slow path (234 ms), and the manual two-batched-GEMMs + precise
softmax is 228 ms -- the batched (1620 x) 1024x16x1024 GEMMs are thin-K/launch
bound, ~5x over the traffic floor. There is no exact-math fix; nets built around
many-token window attention are simply a quality tier on this hardware (papers'
efficiency claims usually assume CUDA flex-attention). Channel attention (MDTA:
C x C over tokens, as in RealViformer) is cheap by contrast -- the matrices are
tiny and the cost lives in the qkv convs.

### Deformable conv: follow the input dtype

The DCNv2 path (`kinovsr/modeling/deform_conv.py`) originally forced fp32: three
cast-copies plus a `Cin*K*K x N*oH*oW` fp32 columns buffer (~1.9 GB at
128ch/480p) written by the im2col kernel and re-read by an fp32 GEMM -- 2x the
necessary traffic for data that only ever had fp16 precision. Running the whole
path in fp16 (sampling reads, columns, GEMM with MLX's internal fp32
accumulation; tap positions stay float) is **3.8x on the op**; BasicVSR++
1.19x whole-net (58.1 dB vs the fp32 path), STDF 1.12x (78.3 dB).

Rejected follow-up: reorganizing the im2col kernel from thread-per-(channel,
pixel) to thread-per-(group,pixel) to share sampling positions across the
group's channels measured *slower* (46 vs 42 ms) -- the kernel is
write-scatter-bound, not position-math-bound. It still sits ~9x over its write
floor; a genuinely better kernel would need a different output layout, which
the downstream GEMM constrains.

## 3. Things that do NOT work at video resolutions

All measured, all worth not re-litigating:

- **Frame batching** (K frames per forward for stateless per-frame nets): 1.00x
  at 480p and 0.98x for NAFNet. These conv nets are compute-bound at >=360p; a
  single frame already saturates the GPU. Batching only pays below ~240p
  (1.37-1.39x at 90x160) -- real video never lives there. Param count does not
  determine dispatch-boundedness; frame size does.
- **Cross-stage fusion** (compiling consecutive stateless stages as one graph):
  1.01-1.04x when stages are balanced; 0.98x for the realistic
  fbcnn+fastdvd+nafnet chain (a temporal stage isolates the stateless ones, and
  the fusable pair was dominated by one net). Barrier removal is noise next to
  conv compute.
- **SPyNet pair batching** (all flow pairs on the batch axis): 1.12x at 256x448
  but 0.92x at 480x854; the per-pair eval also serves as the documented
  memory-spike guard. Left sequential.
- **conv_transpose rewrites**: a 2x2-stride-2 convT is exactly a 1x1 conv +
  pixel-shuffle, but the only shape that wins (fbcnn's 512->256, 1.58x) is a
  0.8 ms op. Not worth the code.

## 4. dtype policy

- **fp16 by default** where provably safe -- fastest conv/GEMM dtype on M1.
- **fp32 islands only where the math demands it:** NAFNet's whole body
  (SimpleGate multiplies channel halves; magnitudes square past fp16's 65504 and
  the harness once silently wrote NaN frames), channel LayerNorm reductions, the
  RDB recombination sums above. bf16 is NOT a good compromise here: bf16 conv is
  slower than fp32 conv on M1.
- **Do not upcast fp16 data "for safety"** on memory-heavy paths -- it doubles
  traffic and adds nothing (the deform_conv lesson). Upcasting is for
  *accumulation and reductions*, not for storage.
- **Any fp16 reduction spanning the full spatial extent overflows.** A global
  average pool (`mx.mean` over H*W: ESC's dynamic-kernel predictor) or an L2
  normalization over H*W tokens (RealViformer's q/k normalize) sums ~400k fp16
  values at 480p -- far past 65504. These pass every small-input parity gate and
  only blow up on real-resolution frames; give every such reduction an fp32
  island. The channel-axis reductions (LayerNorm over 32-512 channels) are safe.
- Watch for fp32-constant contamination: `mx.zeros`/`mx.full`/`mx.arange`
  default to fp32; on an fp16 path pass `dtype=x.dtype` explicitly.
- fp32 summation-order noise compounds through deep residual stacks: a 1e-6
  per-op reorder became 2.6e-2 max after 36 NAFNet blocks. Judge end-to-end
  deviations in PSNR against the 8-bit encode floor (~48 dB), not per-op.

## 5. Graph mechanics (compile / eval / caches)

- `mx.compile` every shape-stable forward once, in a module-level cache. Gains:
  1.3-1.4x for dispatch-bound graphs (many small ops), ~1.05x for compute-bound
  ones. Every net here follows the `make_forward` + bounded-cache pattern.
- Compile caches must be **bounded** (`kinovsr/modeling/compile_cache.py`, FIFO cap
  16): entries close over the checkpoint, so an unbounded id(p)-keyed dict
  retains every checkpoint ever constructed in the process. Eviction is safe --
  a params dict can only be collected (and its id recycled) after its entries
  are gone.
- `mx.eval` exactly once per output frame, at the point the frame is produced.
  The per-step evals inside recurrent propagation loops are **load-bearing** --
  they bound the lazy graph and the DCN column transients (removing them once
  produced a 57 GB OOM). The redundant ones (re-evaluating already-materialized
  outputs at a second layer) are free to delete but worth ~1%.
- One eval is also a sync barrier: do not add them inside a forward.

## 6. Pipeline glue (measured attribution)

Per frame at 854x480, 60-frame runs: decode + passthrough pack + HEVC encode =
**7 ms**; with 4x output the emit path costs ~20 ms (dominated by the ~50 MB
fp16-RGBA upload -- unavoidable at 4x). MLX<->CVPixelBuffer conversion for
preprocessing adds ~7 ms. Conclusion: the glue layer is healthy; the nets
dominate wherever they should.

**Gotcha: never derive per-frame cost from short runs.** A 5-frame smoke test
implied 76 ms/frame of "glue" that was actually one-time warmup (compile traces,
session setup) amortized over too few frames. Harness startup is ~4-5 s
(imports + pyobjc lazy-attribute initialization); it only matters for short
clips. Use 60+ frames for wall-clock numbers.

**VTOpticalFlow denoise gotcha:** use `Quality`, not `Normal`, for mc denoise.
The Normal tier can under-read known global motion badly enough to make history
blending unsafe. The mc denoiser now runs a synthetic global-shift self-test at
startup and refuses clips/devices where VT returns all-zero or wrong-scale flow
while reporting success. Frame-sized destination buffers and source-pixel flow
units are correct; do not reintroduce a fake "native grid" rescale.

## 7. Reference numbers (854x480 input, M1 Max, post-campaign)

Per-frame processing cost, fp16, compiled, capped cache. Preprocessors:

| processor | ms/frame | notes |
| --- | --- | --- |
| stdf deblock | 53 | 1.25x cumulative (compile + gate pads + fp16 deform) |
| fastdvdnet denoise (steady state) | ~53 | 1.14x (grouped-conv gate pad) |
| fbcnn deblock | ~230 | flat U-Net; audited, nothing actionable |
| nafnet gopro (fp32) | ~340 | 1.4x (manual depthwise) |

Decoded RGBAHalf video can carry legal YUV->RGB overshoot outside [0,1]
(observed around -0.4..1.25 on saturated compressed SD material). Learned
preprocessors are trained on clipped RGB, so their entry points clamp before
inference rather than suppressing large residuals afterward. On the 640x480
news stress clip, this changes the failure mode directly: GoPro32 residuals
on frame 60 drop from mean 9.92 to 2.44 when NAFNet sees clipped input, and
FBCNN->GoPro32 on frame 115 drops from mean 0.067 to 0.030 when FBCNN also
gets clipped input. Blocky sources can still need an upstream deblocker, but
the model input domain must be fixed first.

The full upscaler ladder (all parity-gated against independent torch
reimplementations and visually verified on real motion-blurred video):

NAFNet's official GoPro/REDS eval configs instantiate `NAFNetLocal`, not plain
`NAFNet`: their TLC/TLSC converter replaces each SCA global average pool with a
fixed local-window average pool, with window sizes frozen by a dummy 256x256
training-crop forward. SIDD denoise configs stay plain `NAFNet`. The harness
default is `--nafnet-pool auto`, which follows the selected variant:
GoPro/GoPro32/REDS -> local, SIDD/SIDD32 -> global. `--nafnet-pool global`
disables TLSC for deblur checkpoints and keeps the plain global-pooling behavior
suggested in upstream artifact discussions as a possible out-of-distribution
fallback. Keep the resolved mode explicit in reports.

GoPro/GoPro32 NAFNet can still resonate on periodic compression/scanline luma
structure: the deep SimpleGate + SCA stack turns a block lattice into a residual
lattice. After compressed/scanline clips still corrupted under forced
`control-source`, `--nafnet-guard auto` is deliberately conservative: it
defaults to `reject` for the GoPro variants and `off` elsewhere. Reject mode
keeps healthy frames bit-exact and judges each frame on explosion *area*, not
peak: a real resonance covers a visible patch of the frame (measured 4-100% of
pixels with smoothed local residual above half of
`--nafnet-guard-threshold`, default 0.12), while healthy deblur residuals
spike on isolated pixels (<0.1% area), so a raw peak threshold false-trips on
good clips. A tripped frame is emitted as passthrough on the spot; the net is
locked out only after two consecutive trips (or one catastrophic frame, mean
local residual above the threshold), and a locked stage re-probes on a
`--nafnet-guard-lockout` cadence (default 48 frames, about 1.6-2s; 0 = stay
locked for the rest of the clip) so a scene change wins the stage back instead
of losing everything after one bad segment. Intermittent single-frame
explosions therefore get per-frame passthrough while the rest of the clip
still restores, and locked frames skip the net entirely. Guard transitions are
flicker-smoothed twice over. First, a locked stage resumes only on a
clearly-clean probe (hysteresis at half the trip area -- marginal probes stay
locked, so the guard cannot oscillate around the trip line). Second, emission
strength is a gain eased through a smoothstep: it rises over
`--nafnet-guard-ramp` clean frames (default 12, about 0.4s; 0 = hard
switching) and falls over `--nafnet-guard-fall` frames on moderate trips
(default ramp/4, min 2; 0 = hard cut on trips only), emitting the
knee-damped residual on the way down instead of cutting -- a linear
fade-in-only ramp reads as flicker no matter how long it is, because it
starts and stops with a temporal edge and every trip still pops to raw in
one frame (the longer the ramp, the harder a mid-ramp trip pops).
Catastrophic frames (mean local residual above the threshold) or explosions
past 4x the trip area still cut to passthrough immediately; trip evidence is
always measured on the raw residual, and once the gain settles healthy frames
pass through bit-exact. Measured on the flapping worst case (short lockout on
boundary content), hysteresis plus the eased gain cut peak blockwise excess
temporal delta about 15x versus hard switching.

Upstream deblock/denoise composes with the guard but is NOT a safety dial for
this failure. The resonance responds non-monotonically to preprocessing
strength (measured on a 480p lattice stress clip: half-strength FBCNN trips
more frames than full strength while full strength enlarges the explosions
that survive; sweeping luma denoise, severity collapses through 0.15-0.2 and
then spikes again in a pocket around 0.3 before defusing by 0.5), so
intermediate settings can land ON the resonant band that stronger or weaker
settings miss. The trigger is luma-borne: any moderate luma denoising defuses
it, while chroma-only denoising (--denoise-luma-strength 0) makes the peak
explosion WORSE than no denoising at all -- chroma noise was acting as dither
decorrelating the lattice. The guard measures its evidence on NAFNet's actual
input (whatever the chain produced) and a rejected frame emits that input, so
upstream work is preserved and every preprocessing config lands in one of two
safe outcomes: the net behaves, or the guard locks it out. The older experimental
guards remain explicit A/B tools: `residual` applies an output-side soft knee,
`control` does regional two-pass luma-control blending, `control-source`
predicts a whole-frame residual from a luma-control input, and `fast` applies
compiled single-pass residual attenuation. Use `--nafnet-guard off` for raw
model A/Bs.

| ms/frame | --spatial-mode / weights | character |
| --- | --- | --- |
| 19 | safmn light | fidelity, trained on compressed content; the fast default |
| 148 | realesrgan general (SRVGG) | light perceptual |
| ~754 | safmn real2x (2x output) | real-world perceptual, the HD -> 4K class tool |
| 756 | safmn real (SAFMN_L_Real_LSDIR) | real-world perceptual, per-frame |
| 872 | safmn purescale / purescale2x / -sharp | SAFMN-L retrained, lattice-free (see below) |
| 817 | realviformer (streaming) | real-world perceptual + TEMPORAL consistency |
| 976 | realbasicvsr | temporal GAN (cleaning + BasicVSR) |
| 1391 | basicvsrpp | temporal fidelity; 70% propagation / 20% tail |
| ~2400 | realesrgan bsrgan (RRDBNet) | heavy perceptual (1.5x from dense restack) |
| ~3900 | esc gan / mse | window-attention quality tier; mse = fidelity twin |

Input-domain rule: decoded RGBAHalf carries legal YUV->RGB overshoot
(measured -0.14..+1.25; ~2% of pixels beyond gamut at saturated color edges,
worst in dark scenes with colored lights), and every learned net is trained
on clipped RGB. Feeding the overshoot drives the nets outside their input
domain -- measured 56x the confetti-speck area on one GAN checkpoint, with
per-frame noise re-rolling the overshoot pattern into flickering specks.
The preprocessor entries were clamped when this was first found; the learned
UPSCALER entries were missed until it resurfaced -- `to_rgb_batch` now clips
once for all of them. When reproducing artifacts in-process, match the
harness read path (`read_buffer_rgb_f32`, unclipped) -- probes that read
via uint8 or pre-clip their input can never reproduce overshoot-driven
artifacts.

The full range audit, so this never needs re-deriving. WHY overshoot exists:
(a) the legal YUV gamut is larger than the RGB unit cube, so saturated-chroma
video maps outside [0,1] on conversion -- inherent colorimetry, not a defect;
(b) codec DCT ringing pushes values past the original near hard edges;
(c) our own Lanczos/bicubic resamplers have negative lobes and re-create
overshoot from in-range input at hard edges. WHERE it enters: only the
fp16-preserving decode path (`read_rgbahalf_rgb`, used by balanced/image/none
and all learned-upscaler modes) -- the NV12/fast path reads 8-bit (clipped by
construction). GUARDS, both boundaries: every learned-net ENTRY clips (preprocessors:
nafnet/fbcnn/stdf/spatial/mc/fastdvd; upscalers via `to_rgb_batch`;
realviformer also locally), and every stage OUTPUT clips (audited: safmn,
realesrgan, esc, realviformer, basicvsrpp, realbasicvsr, stdf, fastdvd,
fbcnn, luma_chroma_blend; the one gap was the nafnet reject-guard's
raw/blended emissions, now clipped once in denoise()). So between-stage
overshoot is doubly impossible: outputs are in-range and entries clip anyway.
Deliberately UNCLAMPED: the VT-session/writer path (VideoToolbox and the
encoder handle extended-range fp16 correctly; the passthrough stays
authentic) and the restore-borders composite (original border content is
restored verbatim).

LEVELS are part of the input domain too. Every model in the harness trains
on full-range images (image datasets have no limited-range notion; the
Real-ESRGAN recipe additionally clamps the synthesized LQ to the uint8
lattice), so a correctly-flagged limited-range source -- expanded to
full-range RGB at decode -- lands exactly in the training domain. A
MIS-flagged source does not, and measurably: fed a range-shifted frame and
compared after un-shifting, outputs match the correct render at only
29-35 dB, because full-range-trained GANs contrast-pump toward full swing
(squeezed input still produces 0..1 output) -- the shift is not linearly
recoverable downstream. Direction asymmetry: read-as-full when really
limited (washed out) makes the stock GAN over-etch (+23-26% luma high-pass);
read-as-limited when really full (the default assumption for untagged
sources, e.g. screen recordings) crushes shadows to flat black before any
model runs and inflates GAN chroma specks (+86% area on one checkpoint).
`--source-range video|full` forces the interpretation for mis-flagged
sources: the forced path decodes YUV in the container's own range format
(code values pass through unrescaled) and retypes the buffer to the
overridden range variant, so VideoToolbox's converter applies the other
scaling to identical bytes -- a true reinterpretation; asking the decoder
for the other range format directly would rescale instead. Output tags
follow. Not available on the NV12 fast path (its YUV never crosses a range
conversion point).

RealViformer streams (causal recurrence, per-frame state, reset at cuts) --
temporal consistency at barely more than a per-frame net's cost. ESC's numbers
are attention-bound (section 2); its value is output quality plus the only
fidelity/perceptual twin pair above the SRVGG class.

The stock SAFMN-L models can flash a small transient block lattice on video:
SAFM's multi-scale branch max-pools by up to 8x8 and nearest-upsamples back, so
one hot activation is broadcast as a constant block into the modulation gate,
and the winner flips frame to frame. This is trained-in (swapping the pooling
at inference on the stock weights corrupts the output -- the gate expects
max-pool peak statistics). The `purescale*` variants are third-party SAFMN-L
retrains with the fixed branch (avg pool + bicubic upsample, both trained in;
the SAFM mode follows the weights filename) that eliminate the artifact. Their
domain is DECENT-QUALITY sources. Their genuine limit, isolated with
single-factor tests: NO denoising prior -- the curated-HD training reads
noise as detail, so temporally random noise becomes rendered etched texture
(gray-snow overlay multiplies output high-frequency energy 29x; fresh noise
per frame passes through as flicker at 0.74x where the stock real model
suppresses to 0.22x), and sharp in-gamut color edges add mild chroma specks
(7.5x clean area). Pure blockiness is benign, and the stock real model
behaves inversely (more degradation = smoother output; restorer vs upscaler
training). Historical note: the frame-wide confetti epidemic originally
blamed on purescale was a harness bug -- unclipped decode overshoot at the
upscaler entry (see the input-domain rule below); fixing it cut the speck
area 1000x on the stress clip. Use purescale on clean to lightly-noisy
material, real + --safmn-pool-clamp on noisy or compressed video. The
bicubic phases cost ~20% over stock SAFMN-L (872 vs 729 ms measured same-run). They also degrade far more
gracefully on hard out-of-distribution frame edges (a junk half-dark border
row becomes a near-scale thin line rather than a thick smeared band with
blotches above it). CAUTION: unlike every other checkpoint in this table, the
PureScale weights are CC BY-NC-SA 4.0 -- NON-COMMERCIAL use only (see
`kinovsr/processors/safmn/ATTRIBUTION.md`).

Two SAFM dials expose what is and is not swappable at inference.
`--safmn-safm-up auto|nearest|bicubic`: the upsampler is a mild shape-only
choice (forcing the untrained mode shifts the output by only ~0.005 mean);
auto runs each checkpoint's trained mode (verified against the reference:
the stock real models use max pool + nearest). On the stock real models,
forcing bicubic is a creative dial: the blocky nearest-up gate flattens
micro-texture within its blocks, and a smooth gate frees the GAN's texture
synthesis (~30% more output high-frequency energy, visibly grainier/sharper
surfaces; hallucinated micro-texture can shimmer temporally, so judge on
video). Lanczos-3 was tested as a third upsampler and measures within 1% of
bicubic here -- the effect is smooth-vs-blocky, not the kernel -- and on the
purescale retrains all kernels are visually identical (the trained-smooth
gate field has no high frequencies for a sharper kernel to act on). The
POOLING statistic is not a
dial: the gate calibrates to it during training and swapping it corrupts the
output (0.2-0.37 mean shift) -- it always follows the checkpoint.
`--safmn-pool-clamp K` winsorizes each SAFM level's pooled features to
mean +/- K sigma per channel, bounding the hot-activation block broadcast on
the stock models. K is the allowed width, so LOWER K = stronger suppression.
Validated against a real mid-frame lattice occurrence and cost-swept on four
healthy clips: K=4 is visually free (mean shift 0.0004-0.0007) and greatly
reduces the lattice but a residue remains findable frame by frame; K=3 makes
the lattice imperceptible at still-negligible cost (0.001-0.002 mean) -- the
recommended setting; K=2.5 is the boundary where specular-rich content starts
dulling; K=2 visibly mutes highlights and flattens texture sparkle. Failure
is graceful at any K: the clamp pulls outliers toward each frame's own
statistics, so it can only under-modulate, never corrupt. Frame-boundary
pooled cells (a one-cell margin: 2/4/8 input px per level) are exempt, and
the statistics are computed over the interior only: synthetic border
structures (letterbox bars, junk capture rows) saturate the features there
into a quiet self-limiting response, and clamping them back into the
plausible-texture range re-engages the GAN's texture machinery -- measured
on a junk border row, the clamped border visibly bloomed where the
unclamped one stayed a quiet band, regardless of clamp direction. The
artifact and legitimate speculars overlap in feature distribution, so no K
reaches exactly zero lattice at zero cost -- the purescale retrains remain
the root fix; this dial is for when the stock models' rendering (or the
PureScale license) rules them out.

### Synthetic borders: sanitize before the chain (`--sanitize-edges`)

Every learned stage is trained on photographic content run through a
synthetic degradation pipeline; a synthetic border embedded in the frame
(letterbox line, junk capture row, analog edge ramp) has zero probability
under that distribution. GAN stages hallucinate around such edges -- a
1-px junk row measurably corrupts ~35 input rows above it through the
receptive field -- and feature-space mitigations do not compose there (see
the pool-clamp border exemption above).

`--sanitize-edges auto` samples early frames and detects edge lines that are
anomalous in EVERY sample (two rules: near-constant extreme bar rows, and
edge rows sitting far darker than their interior neighbor), capped at 8 px
per edge. The nets always receive the frame with those lines replicate-
extended from the interior; what the VIEWER sees is a separate policy
(`--sanitize-edges-fill`), because a replicate fill that reaches the screen
turns a quiet static-dark border into visible MOVING content -- worse than
the junk it replaced, and perceptible even at 1 px (replicated content
shimmers with the interior where the eye expects stillness). `restore` (the
default) composites the original border back over the processed output,
nearest-scaled and feathered into the content over
`--sanitize-edges-feather` source px (default 2) so the seam between the
soft authentic border and the crisply processed interior does not read as a
line; the restored band is exactly as static as the source itself. `extend`
keeps the fill for those who want the junk gone and accept the shimmer;
`trim` crops the junk lines off entirely, folded into the crop BEFORE the
aspect window is computed (bottom/right bumped 1 px when needed to keep
even dimensions) -- the right mode when reframing anyway. `--square-pixels`
resamples anamorphic sources to a 1:1 pixel aspect (horizontal-only
Lanczos-3 at SOURCE resolution, where the upscaler re-synthesizes the mild
softness; output tagged square) for PAR-ignorant toolchains; the default
passes the source pixel aspect through losslessly. Measured in the DISPLAY
domain (PAR output stretched as a player must), square-pixels is also a ~7%
sharpness win: the stretch happens pre-SR where the net re-synthesizes
detail, instead of post-SR in the player where it dilutes it -- and real
players often stretch with cheaper kernels than the Lanczos used for this
measurement. Comparison hygiene: storage-pixel crops of PAR output display
squished in untagged viewers; display-correct before judging sharpness
across geometries.
Dimensions and pixel-aspect are untouched in every mode. `--sanitize-edges
T,B,L,R` forces explicit counts. Old test captures routinely carry junk on
multiple edges (a classic CIF clip: one junk bottom row plus a 3-px
black-to-dim ramp on the left; another: a near-black right column) and the
auto detector finds these while leaving clean modern content untouched.

True letterbox/pillarbox bars are a different job: `--crop-bars auto`
detects constant-extreme bars (scanned up to 45% per edge -- a 9:16 portrait
shoved into 16:9 leaves ~35% per side), crops them off BEFORE processing,
and outputs only the active picture, rounded so dimensions stay even. The
pixel aspect is unchanged; the display aspect becomes the content's true
aspect, which is the point (16:9-in-4:3 comes out true 16:9). It composes
with `--sanitize-edges` -- junk-line detection runs on the cropped picture,
so a letterboxed VHS capture with a garbage line at the content boundary
gets both treatments. Thick bars are never replicate-filled (that would
fabricate imagery); cropping is the only correct handling for them.
`--crop-bars T,B,L,R` is a general edge-anchored manual crop; if the
requested active area comes out odd, the bottom/right trim is bumped by one
pixel (printed) rather than erroring. `--crop-aspect W:H` center-crops to
the largest even-dimension window with that display aspect (16:9 on 4:3,
1:1, 9:16 portrait extracts) -- a DISPLAY aspect: on anamorphic sources
the pixel aspect is folded into the storage target automatically, and the
closest even-integer window is chosen -- placed with `--crop-anchor` (nine
positions), applied after the bar crop; `--crop-offset
DX,DY` shifts the window from center, clamped inside the frame. Even
dimensions matter beyond cropping: 4:2:0 stores one chroma sample per 2x2
luma block, so odd-dimension video misaligns or fails on the NV12 input path
(--spatial-mode fast) and 4:2:0 encodes; the harness now warns at setup when
the source itself has odd dimensions.

### Case study: the "river" artifact was a merge-normalization port bug

On heavily degraded sources (old low-bitrate footage) RealViformer produced
"rolling waves": dense crocodile micro-texture hallucinated out of compression
noise, position-locked by the recurrence, with smooth wakes sweeping through it
wherever motion broke the flow. Clean sources (DAVIS-class) looked fine. Every
in-repo attribution instrument said "network behavior": raw pre-encode fp32
reproduced it (not fp16, not the encoder), and the port passed its 113+ dB
torch parity gate.

The actual cause was a one-line port deviation in the channel-attention merge
-- the block the paper builds its degradation-robustness claim on. The
reference REBINDS the query to its LayerNorm before both the attention and the
output concat (`x = self.norm_q(x); ... cat([x, norm_out(v)])`); the port
concatenated the RAW shallow features. Raw feature magnitude scales with the
source's noise energy, so the merge FFN ran off its trained operating point
exactly on degraded content, over-locking history and amplifying hallucinated
texture. One line fixed both the river and a mild softness on clean content.

Lessons, now part of the validation protocol:

- Gate 1 (parity vs your own torch reimplementation) is blind to a SHARED
  misreading of the reference: both sides implement the same wrong spec and
  agree to 120 dB. When transcribing a reference forward, hunt specifically
  for variable REBINDING (`x = self.norm(x)` followed by later uses of `x`) --
  a reimplementation that inlines expressions reads past it silently.
- If an artifact contradicts the paper's central claim (here: channel
  attention chosen FOR degradation robustness), suspect the port before the
  network, no matter how many internal instruments agree.
- Behavior gates (fp32/fp16, pre/post-encode, per-frame control) attribute an
  artifact to the net-as-implemented; only a line-level re-audit of the
  reference attributes net-as-implemented to net-as-published.

`--cut-detect hist` (default off) remains relevant for recurrent upscalers on
content with hard cuts. The attribution instrument, when this class of question
comes up again: feed decoded LR frames straight to the upscaler in-process
(bypassing the output encoder), stack output crops of the artifact region
across consecutive frames, and compare the same crop across variants (fp16 vs
fp32, per-frame control net, weight dicts) as tiled contact sheets.

### Known model behaviors on calm/degraded content (net, not port)

Verified on ancient-web-rip degradations: re-encode a clean master with a
hardware H.264 encode at ~0.07 bpp (High profile, long GOP, ~400 kb/s at
544x408 or ~220 kb/s at CIF) and upscale that. The Xiph derf CIF talking-head
masters (akiyo, mother-daughter -- freely downloadable y4m) are the right calm
test content: near-static people expose behaviors that DAVIS-class action
footage hides under motion blur.

- Woven micro-texture ("crosshatch") prior: the params model fills ambiguous
  mid-tone surfaces (animal hide, teeth) with a fabric-like weave. Selective
  by surface class -- truly flat walls stay clean, strong structure gets
  correct material (wood grain, thatch). Identical in fp16 and fp32, present
  pre-encode, absent from the input: pure GAN texture prior.
- Recurrent etching on near-static content: hallucinated shading lines get
  locked by the recurrence and re-sharpened every frame; small jitter turns
  them into notched ridges (a ladder seam down a news anchor's nose) and, on
  smooth gradients, slow ripple contours. Measured mechanism: FORMATION is
  input-gated and fast -- the seam appears within ~10 frames whenever the
  degraded input's structure in that region spikes (tracked seam-region edge
  energy follows the input's own encode-refresh cycle across every config) --
  while PERSISTENCE is state-driven: once formed, a deep state carries the
  etch through quiet input stretches where a fresh state stays clean, and on
  long static shots low-amplitude ripples keep accumulating for hundreds of
  frames (background horizontal-structure energy crosses visibility around
  frame ~200 unbounded). The same frame can render clean or etched depending
  only on hidden state history. The reference inference chunks sequences at
  100 frames (a CUDA-memory workaround that incidentally bounds this);
  `--realviformer-window` (default 100) reproduces that envelope, 0 restores
  unbounded streaming. The window bounds how long a formed etch survives and
  caps slow accumulation; it cannot prevent formation (even window 12 re-forms
  the seam within a chunk when the input triggers), and each chunk join is a
  small texture refresh. For near-static talking heads the recurrent tool is
  simply fragile; the per-frame safmn real weights cannot etch at all.

  Current flow evidence: the checkpoint's END-TO-END FINE-TUNED SpyNet is a
  measurably degraded flow estimator. On controlled synthetic translations of
  Akiyo frame 184, median endpoint error was VT 0.057 px, stock SpyNet 0.064
  px, checkpoint `params` SpyNet 0.488 px, and checkpoint `params_ema` SpyNet
  0.479 px. `params_ema` improves the checkpoint flow slightly, mostly in the
  tail, but it is still far behind VT/stock and the reference loads `params`.
  On a procedural periodic texture, VT stayed sane (0.014 px median EPE), the
  checkpoint variants stayed mediocre (0.544/0.528 px), and stock SpyNet locked
  to wrong repeated offsets (7.788 px). The practical ranking is therefore
  CONTENT-DEPENDENT: VT is the most robust generic flow, stock is excellent on
  simple real-frame shifts but can alias badly, and the fine-tuned checkpoint
  flow is model-compatible but inaccurate.

  Raw flow accuracy still does not predict final VSR quality. In the 300-frame
  Akiyo pre-encoder comparison with `history_gate=improve`, VT was the cleanest
  artifact-wise, finetuned SpyNet had a tiny/noise-level VMAF edge, and stock
  SpyNet changed the nose/face region most despite good simple-shift flow. The
  gate proxy explains the split: on the blue-screen ROI all flows increase
  photometric residual and the gate mostly closes, matching the visible line
  reduction; on face/nose ROIs stock often looks photometrically best and opens
  history hardest, yet produces the most objectionable changes. The working
  failure model is the recurrent loop plus model compatibility: bilinear warp
  resampling low-passes the state, the restoration body re-sharpens it, and
  sub-pixel/content-shaped flow or over-admitted history turns locked edges
  into ridges. RealBasicVSR's black-line ghosting belongs to the same class.
  `--basicvsrpp-flow-mode zero`, `--realbasicvsr-flow-mode zero`, and
  `--realviformer-flow-mode zero` are diagnostic controls for separating
  recurrence-only state from motion-compensated recurrence; they are not
  quality presets.

  The improve gate ports to RealBasicVSR's bidirectional propagation
  (`--realbasicvsr-history-gate improve` + `--realbasicvsr-history-strength`):
  the same per-pixel admission map (shared `history_improve_gate` in
  vsr_blocks) multiplies the warped features after each of the backward and
  forward warps, computed from the CLEANED frames the flows were estimated on.
  A zero gate equals the trained window-start zero features, so gating is
  in-distribution. Verified on the selfie clip at the default window 14: the
  dark scratchy propagation streaks on the face (the artifact the old
  window=1 workaround dodged) are eliminated, at the cost of slight softness
  where those streaks had passed for detail; moving-background regions lose
  some propagation crunch. Composes multiplicatively with the older
  `--realbasicvsr-flow-consistency` occlusion mask; default off = reference
  behavior, untouched math.

  BasicVSR++ has the same knobs (`--basicvsrpp-history-gate` /
  `--basicvsrpp-history-strength`): the gate multiplies the deform-aligned
  history bundle, and the second-order source gates with the better of its two
  flows (the alignment can draw from either). Measured effect on the selfie
  clip is subtle -- fidelity training plus the learned deformable offset
  correction leave little smear to fix -- so it is a knob for content where
  its ghosting actually shows, not a general preset.

  `vt` is available as a flow source on all three recurrent upscalers
  (`--{basicvsrpp,realbasicvsr,realviformer}-flow-mode vt`): VTOpticalFlow at
  the Quality tier through frame-sized flow buffers, with the startup
  synthetic-shift self-test (VT silently returns zeros for portrait input and
  small flow-buffer sizes; the self-test turns that into a loud error). The
  windowed nets take one VT call per direction per frame pair, using the
  empirically validated pull-flow convention (negated forward flow of a
  source->next pair, anchored at next). Character per the flow study above:
  smooth fields, no content-shaped noise, conservative on local motion --
  an option to A/B per content, not a proven preset.
- Output-encode visibility: a downstream HEVC encode makes the weave read
  STRONGER while halving measured fine-detail energy -- the encoder strips
  the incoherent noise floor and the coherent periodic pattern survives
  quantization, standing out against the cleaner background. Judge texture
  artifacts on the encoded deliverable, not only on raw net output.

Peak MLX memory, one pass at 854x480 (1 GB cache cap): stdf 0.8 GB, fastdvd
1.0, general 1.2, fbcnn 1.9, nafnet-fp32 2.3, bsrgan 4.0, realbasicvsr-5-window
6.4, basicvsrpp-5-window 7.2 GB. Scales roughly linearly with window length and
pixel count; 1080p BasicVSR++ projects to ~25-30 GB.

## 8. Methodology for future performance work

Ordered; stop at the first step that explains the time.

1. **Attribute wall-clock first.** 60+ frame run, passthrough config as the
   baseline, cProfile for the breakdown. Do not optimize a stage before knowing
   its share.
2. **Phase-profile the net.** Time its stages with per-phase evals (see the
   BasicVSR++ split above). Estimates from FLOP counts routinely miss by 3x.
3. **Micro-bench suspect ops against both rooflines.** FLOPs / peak-TF/s for
   compute; bytes / ~400 GB/s for bandwidth. An op far from both floors is on
   the wrong kernel.
4. **Kernel-path analysis.** Classify every conv (C, O, kernel, stride, groups)
   AND every runtime concat width against the dispatch gates in
   `mlx/backend/metal/conv.cpp`. Check `mx.fast.*` kernels' shape assumptions
   against the actual tensor shapes (threadgroup-per-row vs many-small-rows).
5. **Apply exact-math transformations only**, in this order of preference:
   gate padding (zero filters/columns; usually bit-exact), weight restacking
   (dense blocks; load-time reordering), manual formulations for pathological
   kernels (depthwise shift-add, hand-rolled channel norms), dtype-following
   (fp16 through memory-heavy paths, fp32 for reductions).
6. **Gate every change on parity.** Bit-exact when the transformation allows
   it; otherwise report max|d| and PSNR vs the previous path and accept only
   deviations far below the 8-bit encode floor. Full-net A/B with compiled
   forwards, capped cache, realistic resolution, then a harness smoke test.
7. **Record rejections with their mechanism.** A rejected idea without the
   measured "why" gets re-attempted. This document and the git history are the
   ledger; the failed ideas in section 3 cost as much to establish as the wins.
8. **When the well is dry, change the math, not the schedule.** After
   exact-math options are exhausted, further speed means lighter architectures
   (smaller trained variants such as nafnet width32; different nets such as the
   SAFMN family) -- i.e., accepting different output. Quantization is NOT a
   lever for these conv nets: MLX's quantized kernels are matmul/LLM-shaped and
   the nets are compute-bound in fp16, not bandwidth-bound.

## 9. Porting new models: selection, validation, drift traps

Lessons from porting the SAFMN family, ESC-Real, and RealViformer (and from
rejecting SMFANet). These cost real debugging time; follow them in order.

### Pick the checkpoint by its TRAINING DEGRADATION, not its name or awards

- A model is only as video-appropriate as the degradations it trained on. The
  Real-ESRGAN high-order pipeline (blur + noise + compression) transfers to real
  video; bicubic-benchmark training (DF2K/DIV2K classic SR) does not, and
  challenge checkpoints are tuned to the challenge's exact degradation and
  metrics. The AIM 2025 "SAFMN-L" (stills, synthetic degradation, no-reference
  perceptual metrics) hallucinates crusty texture over motion blur on real video
  -- verified to be the network, not the port -- while SAFMN_L_Real_LSDIR
  (Real-ESRGAN pipeline, blur included) is clean on the same frames.
- Watch for name collisions: three different checkpoints all named "SAFMN-L"
  exist (DF2K benchmark, AIM challenge, Real_LSDIR). Identify by tensor shapes
  and training provenance, not filename.
- If a project publishes ONLY bicubic-benchmark checkpoints (SMFANet), it has no
  video-appropriate model to port. Skip it.
- Release pages mix architectures: the ESC release ships the author's retrained
  comparison baselines (ATD/HiTSRF/SRFormer) and torch-only variants (FlashBias)
  alongside the real models.

### Validation protocol (all four gates, in order)

1. **Parity vs an independently written torch reimplementation.** Reference code
   is a written spec only -- never imported or executed, not even for parity.
   Reimplement the reference in torch with matching state_dict keys (declare
   vestigial checkpoint params as unused so strict=True verifies everything
   else), load the checkpoint with weights_only=True, and gate the MLX port
   against that. Expect 110-130 dB at fp32 on small inputs; test both the
   size-aligned and the padding code paths. KNOWN BLIND SPOT: a misreading of
   the reference shared by both reimplementations passes at full dB (see the
   river case study in section 7) -- transcribe the reference forward
   line-by-line watching for variable rebinding, and re-audit it whenever an
   artifact contradicts the model's published behavior.
2. **fp16 on REAL frames at REAL resolution.** The fp32 small-input gate cannot
   see fp16 range problems; the full-spatial-reduction overflows (section 4)
   only appear here. Check `mx.all(mx.isfinite(...))` on actual video frames.
3. **Eyeball output frames on real video before shipping.** "Runs, finite, high
   parity on random input" catches neither a wrong checkpoint nor an
   out-of-distribution failure. Include a motion-blurred frame -- it is the
   degradation real video always has and synthetic training pipelines often
   lack, and it is what exposed the AIM checkpoint.
4. **Harness smoke end to end** (geometry, encoder, cut/reset behavior).

### Expect checkpoint-vs-source drift

Released weights routinely disagree with the repo's current code; the weights
win. Cases hit: ESC-Real's upsample tail is Upsample-first in the checkpoint
(params at Sequential indices 1/4/6/8) but conv-first in today's source;
RealViformer's checkpoint carries a vestigial attention parameter the reference
inference silently skips with strict=False; light_SAFMN++ ships ffn_scale 1.5
against the paper's 2.0. Derive EVERYTHING derivable from tensor shapes
(widths, block counts, expansion ratios, scale) and read the key layout before
writing the forward.

The same rule ranks the paper below the code: RealViformer's paper writes its
merge FFN input as the RAW current-frame features (`C[At*Vt; ft]`) while the
released code concatenates the LayerNormed ones -- the weights were trained
with the code, and porting the paper's equation reproduces the river artifact
above. Precedence when sources disagree: checkpoint keys > reference inference
code > paper equations.

### `params` vs `params_ema`: pick the dict the reference loads

BasicSR-style checkpoints often carry BOTH `params` and `params_ema`, and they
are DIFFERENT weights (here: summed max-abs 0.1-0.6 over the tensor set --
visibly different GAN texture, softer edges around movers on the EMA dict).
The dicts also differ in texture prior, not just sharpness: RealViformer's
`params` synthesizes a crisp woven micro-texture ("crosshatch") on ambiguous
mid-tone surfaces (animal hide, skin) where `params_ema` renders soft felt --
both hallucinate, with different styles; the selectivity is the GAN prior
keying on surface class. The embedded SpyNet differs too: in the synthetic
translation probe, RealViformer `params_ema` improved median EPE only from
0.488 to 0.479 px on Akiyo and from 0.544 to 0.528 px on the texture case, so
switching to EMA is not a flow fix. Which dict is "the model" varies per repo,
so read the reference inference:
SAFMN's loaders (`app.py`, `inference_real_safmn.py`, AIS2024) and
RealViformer's `inference_realviformer.py` load `['params']`; ESC's
`scripts/inference.py` loads `['params_ema']`. `pth_to_safetensors.py
--param-key <key>` pins the choice explicitly; if both dicts exist and no
`--param-key` is given, the converter refuses to guess. A converter that
silently picks the wrong dict passes every parity gate you build on the same
choice: the parity reimplementation must load the dict THE REFERENCE loads, not
the dict the converter picked.

Full audit of every shipped videotoolbox weight against its source checkpoint
(2026-07): only the BasicSR-trained trio was ever ambiguous. SAFMN (all
variants) and RealViformer carried both dicts (fixed to `params`); ESC carries
both and correctly ships `params_ema`. Everything else is single-dict by
construction and therefore unambiguous: mmedit checkpoints (BasicVSR++ x4,
RealBasicVSR) are `state_dict`-only; Real-ESRGAN releases each contain exactly
one dict (x2plus/x4plus/anime/RealESRNet are params_ema-only -- EMA IS the
released model there; SRVGG general/animevideo and ESRGAN are params-only);
KAIR-style FBCNN, BSRGAN/BSRNet, FastDVDnet, and STDF are flat single
state_dicts; NAFNet ships `params` only.

### Structure notes

- Causal/unidirectional recurrent nets (RealViformer) want a STREAMING driver:
  per-frame feed with carried state and reset() at cuts -- no window buffering,
  no trim. Bidirectional nets (BasicVSR++ class) need the windowed driver.
- U-Nets constrain input divisibility (2 downs -> /4; deepest pool -> /8;
  window attention -> /32 handled internally). Match the reference pad/crop
  exactly. RealViformer reflect-pads TOP/LEFT to a multiple of 4, then crops
  the scaled top/left offset from the output; bottom/right replicate padding
  was a port bug. Akiyo is already divisible by 4, so the earlier Akiyo flow
  and gate metrics were unaffected, but arbitrary untagged video sizes need the
  reference geometry.
- Reuse the shared blocks: a checkpoint's SpyNet is usually BasicSR's or
  mmagic's -- semantically identical to `vsr_blocks.spynet_flow`; remap key
  names at load and the compiled + gate-padded implementation comes free.
- Fractional FFN expansion factors produce gate-missing widths: RealViformer's
  2.66 makes hidden dims 127/255/510, putting all 75 GDFN convs on the general
  path. The per-half zero-pad (project_in/dwconv rows become [first_h, zeros,
  second_h, zeros] so the gate-multiply chunk stays aligned; project_out gets
  zero columns) is bit-exact -- but measure the WHOLE net: the per-conv ~1.9x
  amounted to 1.02x end-to-end here (the compiled net is dominated elsewhere).
  Post-pass verdicts on the other two: SAFMN-real is healthy (its big CCM conv
  rides winograd at 15 TF/s-eff; all widths aligned); ESC's 13x13 partial conv
  runs at 2.9 TF/s-eff but output-padding it wider is a wash (measured), and
  the precise softmax costs nothing -- keep it.

## 10. Cross-references: prior MLX performance work

Several earlier MLX performance findings border this work; checked for
applicability 2026-07:

- **The steel_gemm BlockLoader "pretranspose cliff"** (PERFORMANCE_NOTES.md,
  BlockLoader characterization): a plain `x @ w.T` matmul at large K uses the
  `_nt_` loader whose 64-row x ldb span breaks coalescing -- pretransposed
  weights (`_nn_` loader) are up to +20% in fp16. This is a MATMUL rule; convs
  go through the steel conv kernels and are unaffected. In the VSR ports the
  only large-K matmuls are the channel-attention q@kT contractions (tiny
  outputs, minor share) and the deform GEMM (no transpose) -- checked, no
  action. Remember it if a future processor leans on big Linear layers.
- **The local STEEL attention retile is NOT reusable here.** Its domain is
  hard-constrained (D=64/128, no mask; masked text attention falls back to
  stock), so ESC's D=16 window attention with a dense additive bias is out of
  scope, and the KinoMLX ISA-level analysis concluded that kernel is at the M1
  silicon floor for its own shapes -- corroborating section 2's verdict that
  there is no custom-kernel rescue for window attention on M1.
- **Custom fused elementwise/FFN kernels bottomed out at a 3-7% ceiling** in
  the transformer work (and FlashAttention-2 ports were abandoned twice).
  Consistent with this campaign's experience: mx.compile already fuses the
  elementwise chains; hand kernels only paid where an MLX kernel was
  pathological (never for fusion alone).
- **Hot-path hygiene scans** (PERFORMANCE.md): the rg patterns for accidental
  numpy round-trips / dict(mx.load) / .tolist() apply verbatim to processor
  code; run them on new ports.
- The transformer-side norm usage (mx.fast.rms_norm at model width) and this
  doc's channel-norm rule (section 2) are the same shape rule from both sides.

## 11. Benchmarking gotchas checklist

- Cap the MLX buffer cache (`mx.set_cache_limit(1 GB)`) -- an uncapped cache
  contaminates both speed and peak-memory numbers.
- Fresh process per configuration for headline numbers; at minimum
  `mx.reset_peak_memory()` and separate compile caches between configs.
- Warm up before timing (compile traces retrace per input shape).
- 60+ frames for anything reported as per-frame cost (see section 6).
- Serial GPU work only -- concurrent MLX benchmark processes contend and can
  hang the GPU (M1 compute hangs are non-preemptible; recovery is a reboot).
  Write logs/sidecars to `$SHARED_TEMP_DIR` before running risky kernels.
- `mx.eval` the output inside the timed region, once.
- Isolated-op wins must be re-measured end to end: compile fusion, memory
  pressure, and phase overlap change the arithmetic (several 2x op wins landed
  as 1.1-1.2x whole-net; one 1.6x op win was a 0.3% whole-net no-op).
