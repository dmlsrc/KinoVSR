# The public host API

`kinovsr.api` is the supported import surface. Everything else -
underscore-prefixed names, submodules, the flat-options CLI plumbing -
is internal and may change without notice.

## Exported names

| name | what it is |
| --- | --- |
| `process_video_file(config, *, video, output, ...)` | Run a pipeline config file-to-file; returns `VideoProcessResult`. |
| `open_pipeline(config, input_spec, *, settings=None, reporter=None)` | Validate the same config against a host stream's `StreamSpec`; returns `PipelineSession`. |
| `PipelineSession` | One validated chain, run at most once; `process(units)` yields output `FrameUnit`s. |
| `VideoProcessResult` | What a file run produced (paths, frame counts, wall time). |
| `resolve_mlx_cache_limit_gb` | The effective MLX cache cap for a `Settings`. |

The stream vocabulary (`StreamSpec`, `FrameUnit`, `Layout`, typed
errors) is imported from `kinovsr.processors`; the CLI-facing config
loader lives in `kinovsr.config`.

## The session contract

```python
from kinovsr.api import open_pipeline

config = {"pipeline": ["dn", "up"],
          "dn": {"processor": "fastdvdnet", "strength": 0.3},
          "up": {"processor": "metalfx", "scale": 2}}

session = open_pipeline(config, input_spec)      # validates everything now
with session, session.process(my_units()) as run:
    for unit in run:                             # natural backpressure
        consume(unit.payload, unit.pts)
```

- **Validation is at open.** Unknown families, bad stage config, and
  stream-contract violations raise typed errors
  (`PipelineError` subclasses) from `open_pipeline`, before any frame
  is touched. A session that opens will not fail preflight mid-stream.
- **Weights load at the first pull.** Opening is cheap; stage
  `prepare` runs when iteration starts.
- **One session, one run.** Stage instances are stateful; a consumed
  session refuses a second `process` instead of silently reusing
  state. Open another session for the next stream.
- **Closing cancels deterministically.** Closing the iterator, the
  session, or leaving the `with` block - at any point, including
  before the first pull - drains nothing, resets nothing, and closes
  every stage exactly once. Exception precedence is fixed and
  documented on `kinovsr.pipeline.scheduler.ChainRun`.
- **Layouts share one path.** MLX RGB frames and CVPixelBuffers are
  both just `FrameUnit` payloads; the input spec's `Layout` states
  which one, and the same validator accepts or refuses the chain.
- **Progress is host-neutral.** Pass any
  `kinovsr.reporting.Reporter`; the CLI passes its Rich-backed one,
  a host can pass its own, tests use `RecordingReporter`.

## Frame ownership and lifetime

A `FrameUnit` payload is either an MLX array or a `CVPixelBuffer` - the
input `StreamSpec`'s `Layout` says which. MLX arrays are immutable
values, so they are never an ownership question. CVPixelBuffers are, and
`process(units, *, retain_outputs=True)` is the switch.

- **Input CVPixelBuffers you pass to `process()` are borrowed, and a
  stateful stage may hold one across pulls.** Temporal interpolation
  keeps the previous source frame until the next one arrives, so a
  buffer handed in on one pull can still be read on the next. Do not
  mutate it, overwrite it, or return it to your own pool until the run
  finishes or you close it; hand in a fresh or retained buffer per
  unit.
- **Outputs are yours to keep by default (`retain_outputs=True`).** MLX
  outputs pass through as the immutable values they are; for a
  CVPixelBuffer layout, `process()` yields a fresh deep copy of each
  output, so you can retain it indefinitely - even after you feed or
  recycle the next input.
- **`retain_outputs=False` is the zero-copy opt-out.** Then outputs are
  yielded exactly as produced: a CV payload may alias a borrowed input
  (a pass-through or identity stage yields the input buffer unchanged)
  or a stage's reused buffer, so it is valid only until the next pull.
  Copy it, or hand it off (retain it, enqueue it to an encoder) before
  advancing the iterator. The file sink runs this way - it consumes each
  unit into the encoder synchronously, so there is nothing to retain.
- **Frame count is not preserved.** One unit in can yield zero, one, or
  several out (interpolation), and a stateful stage can absorb several
  before it emits. Do not assume a 1:1 mapping or a stable total; drive
  off the units the iterator actually yields.

## File runs

`process_video_file` is the same chain grounded by the file endpoints:
the input is probed into a concrete spec, the chain preflights against
it, decoded frames stream through with bounded memory, and the sink
verifies the declared output timeline unit by unit while carrying
audio only when duration was preserved (the synchronization-correct
policy). The CLI's `[pipeline]`-config route calls exactly this
function.

File runs plan and reserve the complete artifact set before opening a
writer: post and comparison videos, audio sidecar, cut log, debug images,
and frame-dump directories. Every artifact is written to a temporary
sibling, every writer finishes while private, and the set is published
together with rollback if any rename fails. Existing destinations are an
error by default; pass `overwrite=True` (or CLI `--overwrite`) to replace
the complete set transactionally. Source aliases, aliases between outputs,
hard links, symlinks, and overlapping file/directory targets are rejected
before output mutation.

The file endpoint currently accepts constant-frame-rate video only. Before
reserving any output, it scans compressed sample metadata, counts the actual
display samples, and verifies that their exact presentation timestamps form a
single cadence (allowing only one-source-tick quantization). Variable-frame-
rate input raises `MediaError` and leaves no output or partial artifact; convert
it to CFR explicitly before calling `process_video_file`. Host-managed
`open_pipeline` sessions can still describe `VariableCadence.VFR`, but file VFR
timestamp propagation is not yet implemented and is never silently
approximated.

## Audio

The session is a **video** contract. A `FrameUnit` carries one video
frame, no stage reads or writes audio, and `open_pipeline` /
`PipelineSession` take no audio argument. This is deliberate: a video
processor can rewrite frame timing (interpolation changes the cadence),
and only a muxer that owns both tracks can keep audio aligned across
that - so the honest stream promise is exact per-unit `pts` /
`duration` on the validated output timeline, which the host uses to
resynchronize its own audio.

`process_video_file` is the supported way to get audio *out* of
KinoVSR: it reads the source track, trims it to the input window and
any output cap, and muxes it **only when the chain preserved clip
duration** - a duration-rewriting chain drops the carry rather than
shipping a track that drifts against the video. For a duration-
preserving cadence change (interpolation), the final output frame is
trimmed to end exactly at the source-window duration, so the muxed
audio stays in sync instead of drifting by the target grid's tail
rounding. Audio carry also requires the source audio and video tracks to
share an origin. A staggered-track file is rejected before any output is
reserved; timestamp-aware audio padding and trimming are not implemented, so
rebasing both tracks independently would silently destroy their offset.
A file carry resolves the exact rational audio sample window before decode.
Native and compatibility readers then pull approximately 250 ms of float PCM
only when each writer reports readiness; post and comparison writers own
independent cursors, and a WAV sidecar uses a separate cursor with a 4 MiB PCM
pull limit. Memory therefore follows the selected window's active chunk and
codec state rather than the complete source duration. Timestamp gaps in
compatibility inputs are retained as bounded silence instead of concatenating
post-gap sound early.
Custom reader adapters that carry audio must implement the bounded
`read_audio_track_window(...)` capability; the former whole-track
`read_audio_track(path)` hook is rejected for file carry rather than allowed to
bypass the memory contract.
A public stream-side
file sink that a host could feed with its own `AudioTrack` stays
deferred until a real out-of-tree adapter proves its shape; today,
encode-with-audio means `process_video_file`.

## Compatibility policy

Pre-1.0: additions land freely; breaking changes to exported names or
their semantics are called out in commit history and the planning
record, and the surface is pinned by `tests/api/test_public_surface.py`.
