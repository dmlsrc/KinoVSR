# Core Image Cache Lifecycle Benchmark

This note records the PERF-08 correction and its acceptance protocol. The
problem was not an unbounded product pool: Core Image rendering work in the
typed pipeline had no common accounting or cleanup owner, so framework caches
could accumulate for the entire process lifetime. The legacy native encoder
performed local cleanup every 64 input frames, but host sessions and file runs
did not.

## Corrected Contract

Every KinoVSR-owned Core Image render enters `ci_render_scope()`. The scope:

- creates a PyObjC autorelease pool around the render;
- admits the render only while no cache clear is in progress;
- records one completed unit of actual render work;
- claims periodic maintenance when 64 render scopes have completed; and
- preserves the active processing exception if maintenance also fails.

The counted render boundaries are RGB upload to a pixel buffer, RGB readback
from a pixel buffer, the Core Image spatial processor, and Lanczos resize.
Direct RGBAHalf memory copies do not render through Core Image and therefore do
not advance the counter.

Public processing entry points also acquire an identity-based cache-owner
lease. The last owner to close performs final dirty-cache cleanup. Ownership
covers:

- `PipelineSession.process`, including processor construction failure,
  exhaustion, explicit cancellation, and abandonment;
- exported `run_plan`, with the same lifecycle guarantees;
- `run_file`, including automatic geometry probing and post-chain comparison
  or diagnostic work; and
- the legacy `encode_video_videotoolbox` utility on success and failure.

The janitor uses identity render tokens, identity clear claims, and one atomic
render ledger. A clear runs outside the synchronization condition, while new
renders remain gated. At the interval boundary it gates new entrants
immediately; the last already-admitted finisher clears all work from that
cohort. The threshold-crossing finisher never waits for its peers, because its
caller may still hold an application lock needed by another render body.
Staggered concurrent loops therefore cannot starve cleanup by keeping at least
one render active, and cleanup does not create a lock inversion. This prevents
a callback from deadlocking the accounting lock and prevents interrupts from
stranding a numeric counter or a global boolean gate. Ordinary framework
cache-clear failures are logged and remain dirty for retry; an explicit
`clear_ci_caches()` call reports the failure. `KeyboardInterrupt` and
`SystemExit` still propagate.

VideoToolbox pools are not flushed every 64 inputs. Those pools have their own
hard capacities and retain their reuse set until processor close. Cache
maintenance is driven by completed Core Image renders rather than source-frame
count, so a frame with multiple conversions is counted accurately and a frame
with none is not counted at all.

## Acceptance Workloads

The committed benchmark is `scripts/dev/bench_ci_cache_lifecycle.py`. Run each
measurement in a fresh process.

The default `--path host` workload drives a typed `PipelineSession` through the
real Core Image spatial stage. It is the release gate because it exercises the
same scheduler, session ownership, processor, and render scope used by public
host callers. `--path helpers` is a narrower diagnostic workload that performs
both RGB upload and readback against an IOSurface-backed NV12 destination.

The default plateau protocol is:

| Parameter | Value |
| --- | ---: |
| Geometry | 320 x 180 |
| Warmup | 256 iterations |
| Measured work | 4,096 renders |
| Sample interval | 128 iterations |
| Plateau tail | 16 samples |
| Cleanup interval | 64 completed renders |

The benchmark does not force Python garbage collection. It samples current and
peak process RSS and walks its own Mach VM regions for the IOKit, Core Image,
and IOSurface user tags. A tag with zero regions means that the kernel did not
attribute a live region to that tag at the sampling boundary; it does not turn
the tagged-region metric into an allocation counter. RSS remains the
process-wide complementary measurement. Only `KERN_INVALID_ADDRESS` is accepted
as the normal end of the Mach address map; any other region-walk status aborts
the benchmark instead of becoming a false all-zero plateau.

The plateau gate requires all of the following:

- at least 4,096 measured renders and four tail samples;
- a plateau tail spanning at least two complete 64-render cleanup intervals;
- a complete sampling interval at the end of the run;
- exactly `ceil(total renders / 64)` cache-clear calls, including final cleanup
  for a partial interval;
- tail RSS growth no greater than 4 MiB;
- tail RSS span no greater than 8 MiB;
- tail RSS slope no greater than 1 MiB per 1,000 renders;
- no more than 2 MiB resident growth, 4 MiB virtual growth, or two new regions
  for each of the IOKit, Core Image, and IOSurface VM tags.

Protocol insufficiency is a gate failure rather than a warning. The report also
records OS, architecture, Python, MLX, product revision, dirty state, every
sample, and the complete threshold set.

## Latency Comparison

The latency check compares median batch milliseconds per completed render
against an otherwise identical baseline report. Baseline compatibility is
validated for the exact Apple hardware model, architecture, OS, Python, MLX,
report schema, path, geometry, warmup, measured iterations, sample interval,
renders per iteration, and cleanup policy. The unmanaged pre-PERF-08 policy is
the sole intentional policy difference accepted against a managed candidate.
The default endpoint margin is at most 10 percent regression.

The pre-fix host path accumulated about 175.39 MiB of RSS across only 256
measured renders. The corrected path grew by about 0.28 MiB in the same
protocol. Representative pre-commit calibration latency was 1.268 ms/render
before the correction and 1.355 ms/render afterward, a 6.82 percent change
inside the endpoint margin. A full corrected run completed 4,096 measured
renders with 68 of 68 expected clears; its 16-sample tail grew by 0.30 MiB with
a slope of 0.15 MiB per 1,000 renders.

These values are calibration evidence, not universal absolute performance
targets. The machine-readable committed-tree reports are authoritative for a
specific acceptance run.

## Reproduction

Define shared paths without embedding a developer-machine location:

    export KINO_REPO=/path/to/KinoVSR
    export KINO_PYTHON=/path/to/the/project/python
    export SHARED_TEMP_DIR=/durable/shared/temp
    cd "$KINO_REPO"

Run the corrected-tree plateau gate:

    "$KINO_PYTHON" scripts/dev/bench_ci_cache_lifecycle.py \
      --assert-gates \
      --output "$SHARED_TEMP_DIR/trace_analysis/perf08_ci_host.json"

For a controlled historical latency comparison, create a clean detached
worktree for the selected baseline. The benchmark script itself comes from the
candidate tree so both processes use the same protocol implementation, while
`--product-root` selects the product implementation under test without changing
the process-launch environment:

    export BASE_REV=f1e3691
    export BASE_TREE="$SHARED_TEMP_DIR/trace_analysis/perf08-base"
    git worktree add --detach "$BASE_TREE" "$BASE_REV"

    "$KINO_PYTHON" "$KINO_REPO/scripts/dev/bench_ci_cache_lifecycle.py" \
      --product-root "$BASE_TREE" \
      --warmup 64 --iterations 256 --sample-every 64 --tail-samples 4 \
      --output "$SHARED_TEMP_DIR/trace_analysis/perf08_ci_baseline.json"

    "$KINO_PYTHON" scripts/dev/bench_ci_cache_lifecycle.py \
      --warmup 64 --iterations 256 --sample-every 64 --tail-samples 4 \
      --baseline-report \
        "$SHARED_TEMP_DIR/trace_analysis/perf08_ci_baseline.json" \
      --assert-latency \
      --output "$SHARED_TEMP_DIR/trace_analysis/perf08_ci_candidate.json"

The short historical comparison is intentionally a latency experiment, not a
plateau acceptance run. It does not meet the 4,096-render protocol minimum.
Use the default command for the memory gate. Repeat historical pairs in fresh
processes when evaluating a noisy or thermally variable machine.

## Regression Surface

Focused tests cover:

- exact cleanup at 63, 64, 65, and 128 renders;
- nested owners, owner-free work, final partial cleanup, and idempotent close;
- concurrent renders and admission gating during a clear;
- continuously overlapping renders that never quiesce at an interval boundary;
- application-lock inversion at a periodic threshold;
- failed periodic clears followed by periodic, explicit, or final retry;
- render failure versus cleanup failure exception precedence;
- autorelease-pool exit failure combined with cleanup failure;
- interrupts while waiting, after claiming a clear, and while committing a
  render completion;
- owner construction and interrupted-release failure windows;
- session, `run_plan`, file-run, and native-encoder ownership on success,
  cancellation, construction failure, and mid-stream failure;
- all actual Core Image render sites advancing exactly once and direct
  RGBAHalf transfers advancing zero times; and
- benchmark clear-count, tail-slope, and report-summary calculations.

No code under the shared reference tree is imported, compiled, or executed by
this benchmark or its regression tests.
