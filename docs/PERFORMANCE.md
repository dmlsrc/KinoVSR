# Performance guidance

Short, practical guidance for choosing backends and chains. Everything
here is measured behavior; the deep reference - kernel paths, dtype
policy, per-family optimization history, and measurement methodology -
is `docs/VSR_PERFORMANCE_NOTES.md`. Absolute times below are one
recorded example (Apple M1 Max, 640x480, 300 frames) to show the shape
of the tradeoffs; measure your own machine and content before deciding.

## Picking a BSVD backend

`--bsvd-backend` selects where the temporal denoiser runs:

- `mlx` (default): the reference GPU path. Fastest standalone
  (example: 28 s solo). It shares the GPU with VideoToolbox and MLX
  upscalers, so in chains it overlaps almost nothing.
- `ane`: the full per-step network as one Core ML dispatch pinned to
  the Neural Engine. Slower alone (example: 39 s solo) but it vacates
  the GPU, and the streaming runtime keeps its dispatches advancing
  while downstream stages run: the example chain with the VideoToolbox
  balanced upscaler finishes in 45 s where the solo stages sum to 73 s.
  First run at a new geometry compiles the model (seconds to a minute,
  cached); frames need at least 96 px per side.
- `mpsgraph`: the same network on the Neural Engine through MPSGraph,
  with no Core ML prediction. GOP windows use one cached schedule-generic
  four-step entry with persistent ANE state and stable skip-ring bindings;
  the ordinary stream keeps the one-step graph. One resident program avoids
  mapped-product phase re-entry, and the generic runtime verifies owned
  result buffers before exposing them: an all-results-untouched activation
  is retried within a fixed bound, while partial or repeated failures raise
  instead of serving stale frames. It has the same overlap behavior as `ane`
  and a different window envelope (padded geometries from 128x256 through
  1024x576). Both accelerator backends fail loudly outside their envelopes
  rather than degrade.

Rule of thumb: `mlx` for a denoise-only run or a chain that is
otherwise idle; an accelerator backend whenever the chain also has GPU
stages, because the denoiser's cost then largely hides under them.

## Chains overlap; plan for it

Since the streaming runtime landed, stages run as independently
advancing owners connected by bounded channels. A chain's cost
approaches its slowest resource, not the sum of its stages - but only
across DIFFERENT resources (ANE vs GPU vs media engine). Two GPU
stages still serialize on the GPU. Judge a processor by its cost
inside your real chain, not its solo benchmark.

## GOP alignment

`--gop-align` anchors recurrent windows on source keyframes (both
recurrence directions cold-start on clean I-frames) and gives the
denoiser per-window conditioning. Overhead is roughly one reprocessed
frame per GOP. Memory for the windowed families scales with window
length: `--gop-max-window` (default 96) is the bound, and long-GOP
sources split internally when they exceed it. `--snap-start` is the
zero-cost way to start a windowed run cleanly when the exact first
frame is negotiable; an exact mid-GOP `--start` decodes and warms up
from the enclosing keyframe, which on long-GOP sources can process
hundreds of context frames before your first output frame.

## Memory expectations

- Recurrent upscalers (BasicVSR++, RealBasicVSR) hold several feature
  maps per buffered frame; peak memory scales with the window dial
  (`--basicvsrpp-window`, `--gop-max-window`). If memory is tight,
  lower the window before lowering resolution.
- `--restore-ensemble` / `--basicvsrpp-ensemble` run the reference
  8-way self-ensemble: 8x the compute of the plain profile.
- The MLX buffer cache is capped automatically at run start; no manual
  cache management is needed.

## Encoding

VideoToolbox hardware HEVC is the output path; `--encode-quality`
(default 0.65) trades size for fidelity and `--encode-chroma 422`
selects 4:2:2 where the hardware supports it. Encoding runs
concurrently with processing and is rarely the bottleneck.

## Measuring

`scripts/dev/bench_endpoint_gates.py` records a machine- and
clip-specific baseline for the endpoint and gates later runs against
it, refusing to compare across changed conditions - the honest way to
answer "did this get slower on my machine".
