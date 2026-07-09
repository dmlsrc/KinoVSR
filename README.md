# KinoVSR

KinoVSR is an MLX-native video super-resolution and restoration toolkit for
Apple Silicon.

It grew out of a production VSR harness and VideoToolbox/MLX restoration stack.
The core project is now video input, restoration, upscaling, evaluation, and
native macOS video I/O; engine-specific adapters belong outside this repo and
can load KinoVSR through a future public API.
