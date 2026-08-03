# KinoVSR

KinoVSR is an MLX-native video super-resolution and restoration toolkit
for Apple Silicon: native macOS video I/O, VideoToolbox spatial and
temporal processing, learned MLX restoration and upscaling, evaluation
tooling, and a CLI that composes them into arbitrary chains. Engine
adapters live outside this repository and load KinoVSR through the
frozen public API in `docs/API.md`.

## Requirements

Apple Silicon Mac, macOS with VideoToolbox, Python 3.14+. Model weights
are mostly not bundled; see Weights below.

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

Native temporal processing - frame-rate conversion to 60 fps with the
high-quality temporal engine:

```bash
kinovsr run --video in.mp4 --output-dir out --target-fps 60 --temporal-mode high
```

A learned restoration chain - temporal deblock of compressed footage,
then 4x learned upscale, with recurrent windows anchored on the
source's keyframes:

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align --restore decompress_track1 --upscale realbasicvsr
```

Temporal denoise on the Neural Engine while the GPU upscales - the
accelerator split is why the chain costs little more than its slowest
stage (see `docs/PERFORMANCE.md`):

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align --denoise bsvd --bsvd-backend ane --upscale balanced
```

Every run accepts `--print-config`, which prints the fully resolved
run as TOML and exits - the same file `--config` accepts back.
`docs/CONFIG.md` documents the config surface: one resolution order,
the flag/TOML ownership rules, `--set`, and the run-level tables.

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
machine-readable manifests; `docs/PROCESSORS.md` is the generated
matrix of every family, profile, artifact, license, and source. Most
weights are external: each family's `weights/README.md` documents how
to obtain and convert them (`kinovsr weights convert`), and
`weights/Attribution.md` credits the upstream work.

```bash
kinovsr weights list      # what each family declares, and what is installed
kinovsr weights verify    # presence and checksums
```

## Host API

`kinovsr.api` is the supported import surface for hosts:
`process_video_file` for file-to-file runs, and
`open_pipeline`/`PipelineSession` for streaming a host's own frames
through a validated chain - bounded internal execution behind a
synchronous iterator. `docs/API.md` is the contract, including frame
ownership and lifetime rules.

## Documentation

- `docs/CONFIG.md` - flags, TOML, `--set`, `--print-config`.
- `docs/PROCESSORS.md` - generated processor/profile/weights matrix.
- `docs/PERFORMANCE.md` - practical backend and chain guidance.
- `docs/VSR_PERFORMANCE_NOTES.md` - deep implementation reference.
- `docs/API.md` - the public host API contract.
