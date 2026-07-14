# VideoToolbox Output-Pool Benchmark

This note records the PERF-05 remediation and its controlled allocation
benchmark. It covers the native VideoToolbox pools used by typed spatial
upscale and frame-rate conversion; caller-owned retain-safe copies are a
separate ownership boundary and are intentionally not counted as native-pool
surfaces.

## Corrected Contract

CoreVideo's ordinary pool acquisition is only a reuse hint. KinoVSR now uses
`CVPixelBufferPoolCreatePixelBufferWithAuxAttributes` with
`kCVPixelBufferPoolAllocationThresholdKey`, so every owned hot-path pool has a
hard allocation ceiling and reports explicit exhaustion rather than silently
allocating a fresh buffer.

The measured stable ceilings are:

| Pool | Ceiling | Reason |
| --- | ---: | --- |
| VSR source, fast/image | 1 | Stateless modes do not retain source history |
| VSR source, balanced | 3 | Current, previous, and sequential-reference handoff |
| VSR destination, all modes | 2 | Native/scheduler output handoff |
| MLX-to-FRC upload | 2 | Previous and current source handoff |
| FRC destination | 7 | Batch four plus measured drain-time scheduler/processor/caller high-water |

FRC processes at most four destinations per native submission. The first
chunk uses sequential submission; later chunks for the same source pair use
`SequentialReferencesUnchanged`. Normal and high modes are bit-exact against
a one-call oracle for a seven-output NTSC-family source pair and its final
hold period.

Pool ownership is explicit:

- a native session owns and flushes its default pools;
- a compatible file-writer pool may replace only the destination pool;
- replacement requires actual format, geometry, and extended-pixel padding
  to meet the VideoToolbox destination descriptor;
- borrowed writer pools are never flushed by the processor;
- pool offers are cleared after preparation and on every close/failure path;
- changing geometry or profile creates new bounded pools, and closing the old
  session releases its pool references;
- `PipelineSession.process` and exported `run_plan` deep-copy native outputs by
  default, preserving the public retain-safe behavior; callers must opt into
  borrowed output explicitly.

## Controlled Protocol

The committed-tree run uses `scripts/dev/bench_vt_output_pools.py` with:

- three fresh worker processes per path and workload;
- alternating fresh/pooled execution order;
- 30 warmup outputs and 120 measured outputs per worker;
- AC power, nominal thermal state, and low-power mode disabled at every parent
  and worker boundary;
- exact product revision and dirty-diff fingerprints at the start, after every
  worker, and at the end;
- underlying `IOSurface.surfaceID` values for the pooled allocation plateau;
- explicit fresh-destination call counts for the pre-fix oracle;
- peak process RSS and per-output timing;
- an active-pixel SHA-256 equality gate across every path and run.

Workloads were balanced VSR at `256x256 -> 1024x1024` and normal FRC at
`128x96`, `24 -> 240` fps. The FRC ratio deliberately produces ten outputs per
source pair, forcing three bounded native submissions and exercising the final
hold-period drain.

The complete machine-readable report is durable at:

    $SHARED_TEMP_DIR/trace_analysis/perf05_vt_output_pools.json

The report records the exact product commit and, when applicable, a complete
working-tree fingerprint at both revision boundaries. The document does not
duplicate that fingerprint because changing the document would invalidate a
dirty-tree fingerprint.

## Candidate Calibration Results

A pre-commit calibration run produced the following representative values.
The durable JSON report is authoritative for the final committed-tree run and
contains every per-run sample, condition boundary, and revision fingerprint.

| Workload | Path | Measured fresh calls/run | Max unique pooled IOSurfaces | Cap | Median ms/output | P95 ms/output | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced VSR | Fresh oracle | 120 | n/a | n/a | 141.914 | 159.953 | 223.656 |
| Balanced VSR | Bounded pool | 0 | 2 | 2 | 141.961 | 162.133 | 223.172 |
| FRC 24 -> 240 | Fresh oracle | 120 | n/a | n/a | 4.794 | 6.158 | 107.297 |
| FRC 24 -> 240 | Bounded pool | 0 | 4 | 7 | 4.901 | 6.426 | 107.688 |

Every calibration pooled run stayed at or below its ceiling, made zero fresh
destination allocations, and matched the fresh oracle's active pixels exactly.
The fresh oracle made one explicit `CVPixelBufferCreate` call per measured
output.

Throughput and peak RSS were effectively unchanged at these small fixed
geometries: VideoToolbox processing dominates per-output time, and peak RSS
already includes native model/framework state. The material win is the hard
allocation contract: long runs no longer create one destination surface per
frame or retain an unbounded pool high-water mark. The ceiling remains valid
when processing is slower than allocation, and a retaining consumer receives
independent host-owned copies rather than forcing the native pool to grow.

## Verification Surface

Focused regressions cover:

- real CoreVideo threshold exhaustion and reuse after release;
- no fresh fallback for owned or borrowed native destination pools;
- FRC `4 + 4 + 2` chunk submission modes and normal/high bit parity;
- 30 retained host outputs surviving more than three pool capacities and
  remaining byte-stable after session close;
- actual FRC output FourCC for every accepted input layout;
- actual extended-pixel padding acceptance and rejection;
- pool-offer cleanup on success, construction failure, and unprepared close;
- FRC and VSR geometry/profile pool replacement;
- public `run_plan` retain-safe default and explicit borrowed opt-out;
- the legacy native encoder forwarding destination attributes into its writer
  pool before direct binding.

No code under the shared reference tree was imported, compiled, or executed
while implementing or measuring this correction.

## Reproduction

From the product repository:

    "$KINO_PYTHON" scripts/dev/bench_vt_output_pools.py \
      --output "$SHARED_TEMP_DIR/trace_analysis/perf05_vt_output_pools.json"

The benchmark refuses fewer than three runs, fewer than 30 warmup outputs, or
fewer than 120 measured outputs. It also aborts on power/thermal drift,
revision drift, output mismatch, any pooled fresh allocation, or a native
surface count above the declared ceiling.
