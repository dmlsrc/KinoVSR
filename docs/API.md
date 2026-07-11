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
| `VideoFileConfig`, `process_video_options` | Transitional flat-options entry for the flag CLI; retires at parity. Hosts should not target it. |
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

## File runs

`process_video_file` is the same chain grounded by the file endpoints:
the input is probed into a concrete spec, the chain preflights against
it, decoded frames stream through with bounded memory, and the sink
verifies the declared output timeline unit by unit while carrying
audio only when duration was preserved (the synchronization-correct
policy). The CLI's `[pipeline]`-config route calls exactly this
function.

## Compatibility policy

Pre-1.0: additions land freely; breaking changes to exported names or
their semantics are called out in commit history and the planning
record, and the surface is pinned by `tests/api/test_public_surface.py`.
