"""CLI option rows owned by the RealPLKSR family."""

from kinovsr.cli.options import Opt

REALPLKSR_OPTIONS = [
    Opt(
        flag="--realplksr-scale",
        group="Spatial Upscalers",
        type=int,
        choices=(1, 2, 4),
        metavar="N",
        family="realplksr",
        key="scale",
        help="Declared 1x restore or 2x/4x upscale output scale for an explicit --realplksr-weights path; must match the checkpoint. Named profiles declare this themselves.",
    ),
    Opt(
        flag="--realplksr-weights",
        group="Spatial Upscalers",
        metavar="PATH",
        family="realplksr",
        key="weights",
        settings_backed=True,
        help="RealPLKSR weights path (.safetensors) for --upscale realplksr or --restore deh264 (or $REALPLKSR_WEIGHTS). Prefer a named profile when available.",
    ),
    Opt(
        flag="--realplksr-profile",
        group="Spatial Upscalers",
        choices=("public2x", "public2x-nn", "nomos4x"),
        family="realplksr",
        key="profile",
        stage_capabilities=("upscale",),
        help="Which RealPLKSR model for --upscale realplksr. public2x (default; 2x LayerNorm+DySample, real-world photo/JPEG, Apache-2.0), public2x-nn (same trained without noise -- for cleaner sources), nomos4x (4xNomosWebPhoto, 4x GroupNorm+PixelShuffle photo restoration, CC-BY). For same-size H.264 cleanup use --restore deh264. None are bundled; see kinovsr/processors/realplksr/weights/README.md. Ignored when --realplksr-weights is given.",
    ),
    Opt(
        flag="--realplksr-dtype",
        group="Spatial Upscalers",
        default="float16",
        choices=("float16", "float32"),
        family="realplksr",
        key="dtype",
        help="Compute/storage dtype for RealPLKSR. float16 (default) keeps norm reductions in fp32; float32 forces a full single-precision run. The 4x checkpoint's fp16 result was measured, but deh264 must be checked on the target footage.",
    ),
]

__all__ = ["REALPLKSR_OPTIONS"]
