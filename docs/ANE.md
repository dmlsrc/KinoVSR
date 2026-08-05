# Running models on the Apple Neural Engine: a field guide

KinoVSR reaches the same temporal denoiser on the Apple Neural Engine
through three product paths: Core ML with `MLState`, MPSGraph compilation,
and direct `_ANEClient` execution of MPSGraph- or Core ML-compiled products.
The final MPSGraph GOP path combines the last two: MPSGraph is its compiler
and ANECIR is its executor. Getting one network working on one route is a
porting exercise. Getting it working on every route is a map of the
maze - and the maze, not the model, is where the time went. This
document records that map for anyone bringing their own model to the
ANE.

Scope and caveats up front: everything here was measured on an M1 Max
under macOS 26.x, mostly on convolutional video-restoration networks.
Much of the mechanism below is private interface - absent from SDK
headers, reached via `objc.lookUpClass` and exported-symbol contracts -
and can shift with any OS release. KinoVSR's policy is to use such
interfaces where they demonstrably work and fail loudly when they stop;
nothing falls back silently. Treat every claim as "verified here,
verify on yours." Prior art worth reading alongside this:
the community `neural-engine` notes (hollance) and the Anemll project's
LLM-on-ANE work.

Three evidence levels matter in this guide. **Public contract** means an
SDK header or documented model format promises it. **Observed internal**
means a private object, compiler product, trace, or controlled probe
exposed it on this machine. **Inferred mechanism** is the smallest
explanation that fits those observations; it is not an Apple contract.
The distinction is especially important in the lifecycle section, where
the failure is proved but the daemon's missing internal transition is not.

## 1. The territory

The ANE is a brokered system device. A daemon (`aned`) arbitrates it
across processes; programs can be loaded, evicted, and reloaded behind
your back. Apple exposes Core ML as the public way to request ANE use,
but exposes no public instruction set, stable cross-generation program
ABI, or low-level ANE programming API. On the tested stack, both the
Core ML Espresso/MIL path and MPSGraph's private MPS-dialect placement
path eventually invoked `ANECompiler.framework`.
`ANECGetMPSDialectSupportedVersion` confirms that an MPS dialect has its
own version; its existence does not document the relationship between
all compiler front ends.

The single most useful control we measured was narrower, and strong
enough: **compiler-front-end compute was not the BSVD gap**. The same
network compiled through MPSGraph and through Core ML executed 24
frames in 2.762 s vs 2.774 s once the surrounding runtimes were
stripped away. For this network and geometry, the measured wall-clock
differences between routes (up to 45%) came from the *envelope*: program
loading and readiness, request wiring, dispatch cadence, state handling,
and instrumentation. This is not a guarantee that arbitrary graphs
lower identically through both front ends; repeat the control for a new
operator mix.

## 2. The doors

**Door 0 - do not use the ANE.** On Apple Silicon the GPU is often
faster standalone for fp16 conv workloads, and some ops never place.
The ANE pays off when you need the GPU for something else (KinoVSR's
chains run denoise on ANE precisely so VideoToolbox and MLX upscalers
own the GPU - see `docs/PERFORMANCE.md`), or when you need its ~10x
power advantage (we measured ~2.5 W ANE vs ~37 W GPU power for
the same network). Composition, not raw speed, is the reason to enter
the maze.

**Door 1 - Core ML via converters.** The sanctioned route:
`coremltools` from PyTorch, `MLModel` prediction, `MLState` for
recurrence, multifunction assets for multi-phase models. The Core ML
runtime (E5RT on the tested stack) owns compilation caches, model and
state lifetimes, and scheduling. A cached BSVD `MLModel` loaded in about
0.26 s in the isolated control, and an immediate first prediction was
reliably executable; that model-load return is the readiness boundary
KinoVSR can actually depend on. State persists only when the caller
serializes predictions using the same compatible `MLState`, as the
public API requires. When this door fits your model, take it and stop
reading.

**Door 2 - Core ML via hand-authored mlpackage.** Same runtime
envelope, but you write the MIL program yourself instead of converting
(`kinovsr/native/anemil/`: a protobuf-level builder with no
`coremltools` dependency, a curated set of compiler-safe op spellings,
and a schema for the model archive). This door exists because
converters and the MIL-to-ANE translator both have ceilings: certain
op spellings crash the translator, certain geometries fail to compile
at all, and above one geometry ceiling we needed a value-exact fp32
island to keep a network correct (`kinovsr/processors/bsvd/ane.py`).
Authoring directly turns "the converter will not express this" into
"emit the one spelling that compiles."

**Door 3 - MPSGraph with ANE placement.** MPSGraph publicly admits
only a Metal device, but the shipping framework carries a compile-time
ANE placement pass behind SPI knobs
(`optimizationLevel = 1`, `preferredDevice = 2`, and a placement
report; see `kinovsr/native/mpsgraph.py`). Placement quality on
conv-heavy nets is excellent - our whole two-stage network places as
one ANE region with no per-op coaxing. What this door does NOT give
you is Core ML's envelope. MPSGraph has public executable-package
serialization and public variable/read/assign operations, and KinoVSR
uses the package format to avoid rebuilding graph structure. Reload
still specializes for the current device, however, and those public
variables are not a documented `MLState`-equivalent contract for
persistent ANE state shared across entry points. The private ANE
placement path exposes no public readiness, residency, or multifunction
lifecycle contract. Everything past "stateless graph, one process,
warm loop" therefore needs separate proof.

**Door 4 - direct execution of compiled products.** Both front ends
emit inspectable compiled products (a program plist plus weights).
`kinovsr/native/anecir.py` executes MPSGraph-compiled products through
`_ANEClient` load/evaluate/unload directly, with one program resident
and explicit lifecycle; `kinovsr/processors/bsvd/ane_direct.py` does
the analogous thing with two Core ML-compiled halves above the geometry
where Espresso cannot translate the island-free whole graph. This door
is useful when a public front end cannot express or correctly translate
the desired partition, or when the runtime above you cannot supply the
measured scheduling and lifecycle contract. You inherit the jobs Core
ML was doing: lifecycle, binding order, caching, and verification. Even
here, `aned` retains final control of the device, so "direct" does not
mean exclusive ownership.

| Door | You get | You owe |
| --- | --- | --- |
| Core ML (1, 2) | managed model/state lifecycle, multifunction, cached BSVD load ~0.26 s | converter/translator ceilings, op spellings |
| MPSGraph (3) | GPU-grade graph API, excellent placement, serialized executable packages | proof for private ANE state, readiness, residency, and entry-point behavior |
| Direct (4) | explicit load/map/evaluate/unload and concrete buffer/request ownership | every private lifecycle guard, including write verification and binding order |

## 3. Observed compiler and ABI constraints

Each item below names its measured scope. Some reproduced below both
front ends; others belong only to a compiled-product ABI or to the Core
ML translator. Do not promote the latter into hardware laws.

**The fused-bias pixel-shuffle defect.** On the probed BSVD shapes,
`conv -> pixel_shuffle` with a per-channel bias before the shuffle
computes garbage on the ANE:
the compiler fuses conv+bias, sinks the bias through the
depth-to-space, and indexes it by post-shuffle channel
(`bias[c]` instead of `bias[4c + 2(y%2) + (x%2)]`). No-bias is exact;
identity ops between bias and shuffle do not rescue (fusion sinks
through them). It reproduces through both front ends. The working fix
is structural: convolve without bias, shuffle, then add a precomputed
full-size bias tile (`pixel_shuffle_biased` in
`kinovsr/native/mpsgraph.py`).

**fp16 behaves better here than on the GPU.** On our conv stacks the
ANE route's fp16 outputs tracked fp32 references roughly an order of
magnitude closer than the same graph on Metal fp16 in the
regime where Winograd-class GPU kernels dominate. Do not assume the
GPU result is the accuracy baseline when validating an ANE port. The
probe measured outputs, not the ANE's undocumented accumulator width.

**Live-port order was lexicographic.** The input/output/state tables in
the MPSGraph-generated products we inspected sorted symbol names as
strings: `__arg0,
__arg1, __arg10, __arg11, ..., __arg2`. Runtimes above you silently
reorder buffers to match. If you bind those products yourself (door 4),
bind in live-table order (`LiveInputList` / `LiveStateList` /
`LiveOutputList`). Numeric netplist order produced immediate status
`0x1d` and untouched outputs in this runtime (`anecir.py` owns this).

**The realized ABI is not your declared ABI.** Placement prunes unused
inputs; lowering can re-inject them at the end; one of our procedures
ended up with its sixteen states bank-rotated (`8..15, 0..7`) relative
to declaration order. Record the realized ABI at lowering time into
whatever cache/contract your runtime reads
(`kinovsr/native/mpsgraph_state.py` does exactly this); never assume.

**Some state descriptors have small fields.** In the probed `anec.state`
slice/ring lowering, particular dimension fields behaved as 8-bit
values: 320 became 64 and 1920 became 128. This is not a claim about all
tensor dimensions. For these contiguous state buffers, the working fix
was to factor the same element count into safe 4-D extents as a
metadata-only reshape and validate the realized descriptor at build
time.

**State-port count is a first-order cost - twice.** Each separate
state port costs roughly 1 ms of request wiring per dispatch (a
16-port program ran 103.3 ms/call; the same math packed into 4 state
slabs ran 90.4 ms/call). Compiler pressure was also sensitive to port
count in this pair: our one-step program failed compilation outright at
16 inline states and compiled at 4. Packing sixteen logical states into
four physical slabs was the single change that reduced our final
observed backend gap to roughly 3-5%. Keep logical-vs-physical state
mapping a per-model choice
(`StateTensorSpec` in `mpsgraph_state.py`).

**But state layout is runtime-dependent.** The opposite layout was
correct on a different route: mapped multi-entry execution rejected
four big slabs at request-wiring time (memory-descriptor overflow) and
required sixteen separate ports. Neither layout is "the right one."
The runtime decides; measure per route.

**Big unrolled procedures hit a working-set wall.** A 16-step unrolled
drain procedure compiled and then hung or silently no-wrote at
production geometry; an 8-step version of the same phase is stable
(and measured 18% faster than emulating the edge with full steps).
Relatedly, three large BSVD programs alternated in the tested Core ML
phase suite while returning to a fourth failed with status `0x16`
(`kinovsr/processors/bsvd/ane_phases.py`). Internal working set is the
best explanation because shrinking or splitting the procedures moved
the failure boundary while external tensor bytes did not predict it;
the private runtime exposes no counter that proves that mechanism.

**The translator has geometry ceilings (Core ML side).** Above
resolution thresholds, MIL-to-ANE translation of our network either
failed or produced an unacceptable execution result. The practical
fixes, in escalating order, were: respell
the op, split the op, wrap the region in a value-exact fp32 island
(`processors/bsvd/ane.py`), or leave the route for door 4
(`ane_direct.py`). Treat this class of wall as a possibility on a
large-geometry model, then locate its actual threshold rather than
reusing BSVD's.

## 4. Lifecycle constraints, or: the silent no-write

The repeated failure mode in KinoVSR's private MPSGraph side-load
experiments was not an API error. A dispatch returned success while all
sentinel-filled output buffers remained untouched. This is a proved
failure signature. Its exact daemon-side cause is not.

**What the probes established.** The mapped multi-entry matrix varied
map application, fresh versus retained execution descriptors, explicit
waits, unload/reload, command-queue versus device dispatch, cache flags,
and shared events. Both fresh and retained descriptors appeared in
passing and failing combinations, so descriptor freshness was not the
root cause. Failing calls clustered around program activation or
re-entry, took reload-shaped latency, and coincided with driver-level
`0x16` messages that the high-level call did not surface. Reapplying a
map or unloading could rescue individual sequences, but no tested
combination supplied a reliable multi-phase contract across the required
transitions and reboot.

Private traces showed Core ML performing apply -> load -> ready work
during model load, while the mapped path exposed no equivalent barrier.
That makes hidden attachment/readiness/residency identity the best
explanation, not a discovered API. We did not identify the missing
token or transition. The product fix was to stop depending on it: use
one program and own explicit `_ANEClient` load/map/evaluate/unload.

**Consequences you must design for on a private route:**

- The only end-to-end signal KinoVSR found for the silent-no-write class
  is an observed write. `anecir.py` poisons probe offsets in one
  deterministic smallest output surface before every dispatch and
  verifies them afterward. Every failure observed in the matrix left
  all outputs untouched, so one witness catches that measured class;
  it is not proof against a future partial-write corruption. Verifying
  every output cost 16 ms/step in `IOSurfaceLock` traffic.
  Instrumentation is part of the system under test.
- Retaining concrete descriptors and binding maps avoids gratuitous
  identity churn, but the matrix proves it is not a readiness signal.
  Use synchronous-completion options only where pipeline logic requires
  completed results; they also did not repair the hidden lifecycle.
- On the direct ANECIR path, a model object that performed an in-process
  cold compile retained stale internal state in our probes. Discard it
  and reopen from the populated on-disk cache before evaluation.
- Cache identity must be content-derived. A constant cache key made
  Apple's compiler cache return a previously compiled *different*
  model ("Missing procedure" at run time). Hash the module content.
- Mapped-product attachment state did not survive reboot, and applying
  the entry-point map was not a complete attachment contract: a fresh
  same-process product still produced no-writes in tested sequences.
  Treat that mapped multi-program design as falsified on this OS/chip,
  not as an almost-working route that needs another warmup.

**status 0x16 is non-specific, not a diagnosis.** It appeared in
different contexts: competition from a second ANE-using process,
returning to a fourth large phase program, reusing a post-compile model
object, mixed-runtime residue (below), and oversized procedures. Those
experiments do not prove the same daemon-side cause. When you see it,
enumerate lifecycle transitions before tuning buffers.

**Process hygiene.**

- Do not mix the mapped MPSGraph side-load path and direct `_ANEClient`
  execution in one process on this tested stack. Loading and closing the
  former made later direct-client dispatches fail with explicit `0x16`.
  This observation does not imply that ordinary Core ML models cannot
  coexist in one process.
- Run large experimental compiles in a throwaway process; long probe
  sessions showed accumulating compiler/runtime failure state.
- At production geometry, treat the FIRST silent no-write as a stop
  signal. In the observed hang class, `SIGKILL` did not retire the
  process, the daemon wedged, and one wedged state kernel-panicked the
  machine on sleep. Only a reboot recovered that machine, and a wedged
  driver invalidates all timings anyway. Bound probes; stop early;
  reboot before re-measuring.

## 5. Cadence and scheduling measurements

**Chains cost max(stages) only if you let them.** A chained pipeline
(decode -> ANE denoise -> GPU upscale -> encode) costs roughly the
maximum of its stages when they overlap - and the sum when they do
not. In the tested KinoVSR chain, the variable that decided was ANE
dispatch *cadence*: a request holding multiple frames delayed the GPU
stage's opportunities to run. The same chain measured 35.7 s with
4-frame ANE dispatches (the stage sum) and 27.7 s with 1-frame
dispatches; grouped 2-frame dispatches were 10% faster per frame in
isolation and 13 s slower in the chain. KinoVSR therefore uses
one-frame steady-state requests in live chains. The Core ML backend's
specialized 8-frame fill/drain calls are bounded edge work, not its
steady-state cadence. Batch only after measuring the composed workload.

**Keep dispatches back to back.** In the BSVD cadence probe, an ANE
request issued after at least 10 ms of host idleness paid a 15-23 ms
power-state ramp. A warmup can move that cost but cannot erase it. Use
a dedicated submission worker with depth-one
pipelining - submit one job, prepare the next, join before touching
results (`kinovsr/native/dispatch.py`) - and pin it to
USER_INITIATED QoS. Background QoS measured 25% slower for this
recurrent video workload; third-party results in the other direction
did not transfer.

**Corroborate placement with three different signals.** The placement
report can print misleading per-op complaints while still placing the
whole graph. KinoVSR checks (1) the compiler's final partition report,
(2) the ANE JIT-compilation counter moving, and (3) ANE package power
under load (`kinovsr/native/sensors.py`). No one signal proves every
operation's realized runtime placement: `MLComputePlan` and placement
reports describe a plan, while power proves device activity but not
which operation caused it.

**Respect run-to-run variance.** The contended three-engine chain had
enough several-percent run-to-run spread that a single low-single-digit
delta could not establish a win. Measure an interleaved distribution on
the exact chain; do not borrow a variance percentage from another
revision. Warm and cold runs are different experiments - report both
separately because first runs can include compilation and activation.

## 6. State on the ANE

The **public** `MLState` contract is deliberately smaller than its
implementation: it is a model-created handle to zero-initialized state
buffers; predictions sharing it must be serialized; and CPU access is a
scoped `MLMultiArray` view whose address may change on every call.

Private inspection on this stack added three observed details: (a) the
BSVD state used IOSurface-backed storage that private objects also
exposed through Metal buffers; (b) its `read_state -> slice_update ->
write_state` program lowered to live-state ring-buffer reader/writer
units; and (c) Core ML retained compatible state identity across the
functions in KinoVSR's multifunction asset. Neither backing identity nor
ANE-native physical layout is a public promise. The full-range update of
the state's own read was the compiler-safe spelling for this 16-state
graph, not the only legal state program in Core ML.

`MLState` was not the missing speed mechanism. Against the already
copy-free explicit-state ABI, compiled-in state ports were only about
1.9% faster per dispatch on this network. Its larger value is the
runtime-owned lifetime and API contract, plus avoiding state
round-trips in designs that have not already eliminated them.

Getting recurrence on each door:

- **Door 1/2:** use `MLState`. This is the one door where state is a
  public product feature and the runtime owns its lifecycle contract.
- **Door 3:** MPSGraph has public variable ops, but no documented
  `MLState`-equivalent contract for the private ANE placement route. Its
  placed dialect does expose live state, and one private op-level form
  changed packaging in our probes: regions
  wrapped as the private grouped-procedure form compile into ONE ANE
  model with multiple procedures sharing live state (selected
  per-request by procedure index), while the default form emits one
  single-procedure model per region. `kinovsr/native/mpsgraph_state.py`
  keeps that behind a model-neutral contract: declare named states,
  provide ordinary graph programs, get one multi-procedure model with
  shared persistent state. The micro-contract is proven; the grouped
  full-geometry cold build remains too expensive to ship.
- **Door 4:** state is persistent IOSurfaces kept alive for the runner's
  lifetime, plus discipline: zero them only at reset boundaries, never
  while a request is mapped, and rotate surface *handles* instead of
  copying tensor bytes between frames
  (`processors/bsvd/ane_direct.py`).

## 7. Choosing your door

1. Start at door 1. If your model converts, predicts correctly, and
   the load/predict envelope fits your product, you are done. For BSVD,
   the stripped-runtime control found no compute win downstream of the
   converter; a new model still needs its own control before making that
   claim.
2. Hitting converter or translator walls (crashes, miscompiles,
   geometry ceilings, missing spellings)? Move to door 2 and author
   the MIL yourself. Budget for op-spelling archaeology; keep a
   curated list of known-good spellings like `anemil/` does.
3. Need GPU and ANE cooperating inside one graph, or a graph API for
   research iteration? Door 3 - and accept that you own persistence
   and readiness the moment you leave "one process, warm loop,
   stateless."
4. Need explicit program lifecycle, concrete IOSurface/request ownership,
   or a partition the public Core ML path cannot translate correctly?
   Door 4, with this document as the checklist of what you just signed
   up for. Core ML itself supports multifunction models; direct execution
   is not required merely because a model has more than one function.
5. At every door: corroborate placement with all available signals,
   accept only on your real pipeline (solo benchmarks do not transfer
   in either direction - we had "wins" evaporate and "costs" vanish
   when composed), and validate through at least as many
   reset/transition cycles as production will see. Our worst productization
   bug passed a two-window gate and failed on the third window.

## 8. What not to retry

Closed by measurement or falsification on this hardware/OS; recorded
so you spend your budget elsewhere:

- Mapping precompiled products into MPSGraph executables as a
  multi-program switcher (five variants; unreliable under contention;
  the required attachment/lifecycle contract was not exposed on this
  OS/chip).
- A fourth alternating large BSVD phase program on the tested Core ML
  lifecycle. This is not a universal three-model ANE limit.
- 16-step BSVD edge procedures at 640x480 on this compiler/runtime.
- Prewarm rituals as a correctness mechanism (readiness is a
  lifecycle contract; warmups just move the race).
- Per-output write verification (16 ms/step); one witness surface
  detects the request-wide no-write class actually observed, with the
  explicit partial-write caveat above.
- Multi-frame BSVD dispatch batching inside the measured live chain;
  it starved the companion engine despite winning solo.
- Treating background QoS as a portable speed hack.
- Assuming GPU fp16 output is the accuracy reference for ANE ports.

## 9. Where to look in this repository

| File | What it demonstrates |
| --- | --- |
| `kinovsr/native/mpsgraph.py` | ANE placement SPI, placement verification, safe pixel-shuffle spelling |
| `kinovsr/native/mpsgraph_state.py` | state declaration/packing, grouped multi-procedure lowering, realized-ABI recording |
| `kinovsr/native/anecir.py` | direct `_ANEClient` lifecycle, live-table binding, caching, write-verification canary |
| `kinovsr/native/anemil/` | hand-authored mlpackage/MIL, curated compiler-safe spellings |
| `kinovsr/native/dispatch.py` | depth-one dispatch pipelining, QoS pinning |
| `kinovsr/native/sensors.py` | ANE power as runtime-device corroboration |
| `kinovsr/processors/bsvd/ane.py` | Core ML + `MLState` backend, fp32 island above the translator ceiling |
| `kinovsr/processors/bsvd/ane_phases.py` | phase-specialized sparse windows, resident-program ceiling |
| `kinovsr/processors/bsvd/ane_direct.py` | direct route with rotating surface handles, alternating two-program lifecycle |
| `kinovsr/processors/bsvd/mps.py`, `mps_phases.py` | the MPSGraph backend and its one-resident-program schedule |

The short version: on BSVD, once both front ends executed equivalent
arithmetic, the remaining performance and reliability work was the
envelope - loading, readiness, residency, binding order, cadence, and
state lifetime. That is a measured control result, not a universal
silicon guarantee. Pick the door whose contract you can validate,
corroborate plans with writes and device activity rather than return
codes alone, and when something fails silently, audit the lifecycle
before changing graph math.

## Addendum: relevant GitHub repositories

This investigation did not come from one reference implementation. The
repositories below contributed different pieces of the map: public model
semantics, callable private-runtime signatures, IOSurface conventions,
MPSGraph construction patterns, or independent measurements to challenge.
A listing here does not imply that KinoVSR copied an implementation or that
another project's measurements transfer to this workload. Descriptions
reflect the repository snapshots inspected in July-August 2026; pin a commit
when reproducing anything that depends on private interfaces.

### References used directly

| Repository | What to read it for | Boundary of the evidence |
| --- | --- | --- |
| [apple/coremltools](https://github.com/apple/coremltools) | The canonical public reference for Core ML conversion, multifunction packages, `MLState`, MIL `read_state` / `coreml_update_state`, compute-plan inspection, and the protobuf schemas used by `anemil`. Start with `docs-guides/source/stateful-models.md`, `multifunction-models.md`, and the iOS 18 state-op definitions. | It defines the public model contract, not ANE placement, readiness, or performance on a particular OS and chip. Those still require a realized plan and runtime probes. |
| [pedronahum/MetalHLO](https://github.com/pedronahum/MetalHLO) | A substantial Swift MPSGraph compiler plus heterogeneous GPU/ANE experiments. Its `ANERuntime` source made the private compile/load/evaluate/unload flow, `_ANERequest` procedure index, IOSurface wrapping, and physical-layout assumptions easy to compare against live framework introspection. | It supplied API and interop clues, not a proven MPSGraph shared-state or multi-procedure contract. KinoVSR independently established the placed-dialect transformation and lifecycle behavior. |
| [christopherkarani/Espresso](https://github.com/christopherkarani/Espresso) | The closest independent example of a direct `_ANEClient` / `_ANEInMemoryModel` runtime: MIL compilation, content caching, IOSurface I/O, persistent request objects, recurrent graph fusion, and several evaluation paths are visible in `Sources/ANEInterop`. | Its transformer results showed an important control: direct dispatch itself was essentially tied with Core ML; the larger gain came from graph and recurrence redesign. It was used as a read-only reference, not as a dependency. |
| [hollance/neural-engine](https://github.com/hollance/neural-engine) | The best approachable starting point for why Core ML models do or do not use the ANE: placement limits, unsupported layers, model surgery, system-log inspection, internals, and links into reverse-engineering work. | It is primarily a conceptual and public-Core-ML guide. Some device tables and historical limitations predate current stateful models and macOS 26, so re-check current behavior. |
| [Anemll/Anemll](https://github.com/Anemll/Anemll) | A production-shaped example of Core ML LLM conversion, stateful KV caches, multifunction prefill/decode models, Swift execution, and `MLComputePlan`-based profiling. It is useful for seeing `MLState` used as a product feature rather than a toy accumulator. | Its workload, tensor geometry, quantization, and deployment envelope differ sharply from video restoration. Reuse the state and validation patterns, not its throughput expectations. |
| [ronaldoussoren/pyobjc](https://github.com/ronaldoussoren/pyobjc) | The Python bridge and metadata for Core ML, Metal, and MPS-family frameworks. KinoVSR uses the same Objective-C runtime access mechanism to reach public objects and dynamically discovered private classes without private headers. | PyObjC exposes callable Objective-C surfaces; it does not make private selectors stable or document their semantics. Resolve and validate them at runtime. |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | The GPU-side companion in KinoVSR's heterogeneous pipeline. Its allocator and zero-copy host-buffer work are relevant to IOSurface/Metal bridge experiments and to understanding when unified memory still incurs a software copy. | MLX is not an ANE runtime. Its value here is composition and shared-memory interop, not evidence about ANE compilation or state. |

### Independent implementations and surveys worth comparing

| Repository | Why it is useful | Read with this caution |
| --- | --- | --- |
| [sbryngelson/ane-guide](https://github.com/sbryngelson/ane-guide) | A broad architecture-to-runtime guide covering the datapath, memory hierarchy, compiler, program format, driver, firmware, and command protocol. Claims are usefully labeled as measured, decompile-derived, or predicted. | The direct route is private, version-fragile, and not App-Store-safe; treat its per-family tables as snapshot data. |
| [sbryngelson/ANEForge](https://github.com/sbryngelson/ANEForge) | An independent graph-to-E5RT compile/run frontend with content-addressed compilation, resident-buffer aliasing, LLM state examples, cost models, and unusually broad benchmark/probe coverage. | It is an adjacent implementation, not the source of KinoVSR's measurements. Its E5RT envelope and failure modes are not automatically the same as MPSGraph products driven through `_ANEClient`. |
| [maderix/ANE](https://github.com/maderix/ANE) | Compact Objective-C experiments for in-memory MIL, direct `_ANEClient` execution, IOSurface sharing, dynamic weights, compiler limits, power, and cross-generation behavior. It is a useful source of small probes rather than a large framework. | Many results are shape- and generation-specific, and the project explicitly describes itself as research code rather than a production runtime. |
| [slavko-at-klincov-it/ANE-Training](https://github.com/slavko-at-klincov-it/ANE-Training) | A larger C/Objective-C direct-runtime and training test bed with API discovery, hardware monitoring, dynamic spatial packing, compile-budget experiments, and energy measurements. | It reported background QoS as faster. KinoVSR measured USER_INITIATED about 25% faster on its recurrent video workload. This is the clearest reminder to treat third-party tuning advice as a hypothesis, not a setting. |
| [harsha-gouru/apple-neural-engine-notes](https://github.com/harsha-gouru/apple-neural-engine-notes) | A concise synthesis of architecture fit, hybrid-runtime design, SRAM heuristics, and the recurring conclusion that layout, dispatch, cache, and setup overhead often dominate silicon time. | It deliberately omits private-interface and reproduction detail. Use it for research direction and experiment framing, not ABI facts. |

The practical reading order is public contract first (`coremltools`), then one
application-scale state user (Anemll), then at least two independent direct
runtime implementations before touching private APIs. For every borrowed
hypothesis, keep the local controls from this guide: verify output writes,
inspect the placement plan, corroborate device activity, measure the full
pipeline, record chip/OS identity, and exercise more lifecycle transitions
than production needs.
