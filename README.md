# KinoVSR

KinoVSR is an MLX-native video super-resolution and restoration toolkit for
Apple Silicon.

It grew out of a production VSR harness and VideoToolbox/MLX restoration stack.
The core project is now video input, restoration, upscaling, evaluation, and
native macOS video I/O; engine-specific adapters belong outside this repo and
can load KinoVSR through a future public API.

The installed console command is `kinovsr` (see `kinovsr --help`). Processor
options follow one shared vocabulary: `--<family>-<key>` where the same key
(`profile`, `weights`, `strength`, `dtype`, `window`, `trim`, `flow`, ...)
means the same concept in every processor family. Chain-level dials such as
`--denoise-strength` distribute positionally over a comma-chain
(`--denoise mc,bsvd`); a family-level flag such as `--bsvd-strength`
overrides the chain value for that family. The CLI accepts canonical
vocabulary spellings only.
