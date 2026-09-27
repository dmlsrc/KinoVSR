# KinoVSR

KinoVSR is an MLX-native video super-resolution and restoration toolkit
for Apple Silicon: native macOS video I/O, VideoToolbox spatial and
temporal processing, learned MLX restoration and upscaling, evaluation
tooling, and a CLI that composes them into arbitrary chains. Engine
adapters live outside this repository and load KinoVSR through the
frozen public API in [docs/API.md](docs/API.md).

## Requirements

Apple Silicon Mac, macOS 27 or later, Python 3.14+. Earlier macOS
releases are not tested and not supported. Model weights are mostly not
bundled; see [Weights](#weights) below.

## Install

```bash
uv pip install -e .
```

Optional extras draw the dependency boundaries:

- base install: the full runtime - MLX, the PyObjC frameworks, Rich.
  No NumPy, no ffmpeg.
- `[ffmpeg]`: the PyAV compatibility reader for containers the native
  reader refuses (MKV, VP9, AVI-era material).
- `[eval]`: metrics and scoring (`kinovsr metrics ...`); brings NumPy
  and the eval-only model stack.
- `[dev]`: tests, lint, benchmarks, converters, and their NumPy/OpenCV
  oracles.

## Quickstart

Native spatial upscale (VideoToolbox, no weights needed):

```bash
kinovsr run --video in.mp4 --output-dir out --upscale balanced
```

File runs carry source audio when present, and output names start with
`vsr_<input stem>_` (for example, `vsr_in_`). Use `--no-audio` for silent
output or `--output-prefix NAME` to choose the prefix yourself.

Native temporal processing - frame-rate conversion to 60 fps with the
high-quality temporal engine:

```bash
kinovsr run --video in.mp4 --output-dir out --target-fps 60 --temporal-mode high
```

On macOS 27, VideoToolbox renders motion at some frame sizes, NTSC 720x480
among them, close to a blend of the two source frames. KinoVSR checks each
size once when interpolation starts and runs those sizes edge-padded into a
size that works, then crops back.

A learned restoration chain - temporal deblock of compressed footage,
then 4x learned upscale, with recurrent windows anchored on the
source's keyframes:

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align --restore decompress_track1 --upscale realbasicvsr
```

Temporal denoise on the Neural Engine while the GPU upscales - the
accelerator split is why the chain costs little more than its slowest
stage (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md)):

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align --denoise bsvd --bsvd-backend ane --upscale balanced
```

Every run accepts `--print-config`, which prints the fully resolved
run as TOML and exits - the same file `--config` accepts back.
[docs/CONFIG.md](docs/CONFIG.md) documents the config surface: one
resolution order, the flag/TOML ownership rules, `--set`, and the
run-level tables.

## Options vocabulary

Processor options follow one shared vocabulary: `--<family>-<key>`,
where the same key (`profile`, `weights`, `strength`, `dtype`,
`window`, `trim`, `flow`, ...) means the same concept in every family.
Chain-level dials such as `--denoise-strength` distribute positionally
over a comma-chain (`--denoise mc,bsvd`); a family flag such as
`--bsvd-strength` overrides the chain value for that family. The CLI
accepts canonical vocabulary spellings only.

## Weights

Learned families declare their profiles and weight artifacts in
machine-readable manifests; [docs/PROCESSORS.md](docs/PROCESSORS.md) is
the generated matrix of every family, profile, artifact, license, and source. Most
weights are external: each family's `weights/README.md` documents how
to obtain and convert them (`kinovsr weights convert`), and
`weights/Attribution.md` credits the upstream work.

The generic converter statically scans pickle metadata, reconstructs tensors
through an exact-allowlist restricted reader with bounded zip and legacy
storage handling, and writes safetensors for runtime use. SpyNet conversion
uses that same reader; the development torch oracle uses the same static
scanner plus `torch.load(weights_only=True)`. Runtime model loading never
accepts pickle-family checkpoints.

```bash
kinovsr weights list      # what each family declares, and what is installed
kinovsr weights verify    # presence and checksums
```

## Host API

`kinovsr.api` is the supported import surface for hosts:
`process_video_file` for file-to-file runs, and
`open_pipeline`/`PipelineSession` for streaming a host's own frames
through a validated chain - bounded internal execution behind a
synchronous iterator. [docs/API.md](docs/API.md) is the contract,
including frame ownership and lifetime rules.

## Development

Install the `[dev]` extra with `uv pip install -e '.[dev]'`. The Markdown
linter is a standalone Rust tool installed outside the venv (`brew install
rumdl`, or `cargo install rumdl`); its rules live in `[tool.rumdl]` in
`pyproject.toml` so every install checks the same way. Then use the
lightweight runner for the common feedback loops:

```bash
python scripts/dev/test.py                 # quick: no integration/slow/weight tests
python scripts/dev/test.py full            # the complete suite
python scripts/dev/test.py concurrent      # the complete suite in concurrent lanes
python scripts/dev/test.py quick -x tests/media/test_timing.py
python -m ruff check .
rumdl check .
```

The quick lane includes both unit-marked and unmarked tests; it is intentionally
defined by what it excludes because the older test tree is not exhaustively
marked. The full lane remains the pre-merge correctness check.

The concurrent lane runs the same tests in about half the time: three pytest
processes at once, one of them for the Neural Engine tests and one for
VideoToolbox frame processing, then the tests marked `solo` alone. Those compare
VideoToolbox frame-rate conversion output bit for bit across two runs, and load
from other processes changes VideoToolbox's arithmetic, so they (and the full
lane) need a machine that is otherwise idle.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) - choose processors, combine stages, and
  tune a run; links to one guide per processor.
- [docs/CONFIG.md](docs/CONFIG.md) - flags, TOML, `--set`, `--print-config`.
- [docs/PROCESSORS.md](docs/PROCESSORS.md) - generated
  processor/profile/weights matrix.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) - practical backend and
  chain guidance.
- [docs/VSR_PERFORMANCE_NOTES.md](docs/VSR_PERFORMANCE_NOTES.md) - deep
  implementation reference.
- [docs/API.md](docs/API.md) - the public host API contract.
- [docs/ANE.md](docs/ANE.md) - field guide to the Apple Neural Engine:
  routes, compiler and lifecycle constraints, cadence, and state.
