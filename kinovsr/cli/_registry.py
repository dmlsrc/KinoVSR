"""Compose foundation, slot, conditioning, and family CLI option rows.

Family contributions live beside their processor implementations; shared
foundation and cross-family slot rows live in :mod:`foundation_options`. This
module is the single ordered composition point consumed by the parser and the
vocabulary conformance tests.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .foundation_options import (
    DEBLOCK_SLOT_OPTIONS,
    DEBLOCK_STRENGTH_OPTIONS,
    DENOISE_SLOT_OPTIONS,
    PREPROCESS_SLOT_OPTIONS,
    RUNTIME_OPTIONS,
    SOURCE_AND_OUTPUT_OPTIONS,
    UPSCALE_SLOT_OPTIONS,
)
from .options import Opt


def _load_contribution(path: str, *names: str) -> tuple[list[Opt], ...]:
    """Load data-only family rows without importing family implementations.

    A normal family submodule import executes the family package initializer
    first. Several families define their runtime there, which would initialize
    MLX or native frameworks just to build CLI help. Loading the adjacent data
    contribution by file path preserves the lazy catalog boundary while keeping
    row ownership inside the family directory.
    """
    source = Path(__file__).parents[1] / "processors" / path
    module_name = "_kinovsr_cli_options_" + path.replace("/", "_").removesuffix(".py")
    spec = spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CLI option contribution {source}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(getattr(module, name) for name in names)


(
    BASICVSRPP_RESTORE_OPTIONS,
    BASICVSRPP_UPSCALE_OPTIONS,
) = _load_contribution(
    "basicvsrpp/cli_options.py",
    "BASICVSRPP_RESTORE_OPTIONS",
    "BASICVSRPP_UPSCALE_OPTIONS",
)
BSVD_OPTIONS, BSVD_STRENGTH_OPTIONS = _load_contribution(
    "bsvd/cli_options.py", "BSVD_OPTIONS", "BSVD_STRENGTH_OPTIONS")
(DEBLOCK_MAP_OPTIONS, NOISE_MAP_OPTIONS) = _load_contribution(
    "conditioning_cli_options.py", "DEBLOCK_MAP_OPTIONS", "NOISE_MAP_OPTIONS")
(CROP_OPTIONS,) = _load_contribution("crop/cli_options.py", "CROP_OPTIONS")
(CUT_OPTIONS,) = _load_contribution("cut_detect/cli_options.py", "CUT_OPTIONS")
(DEFLICKER_OPTIONS,) = _load_contribution(
    "deflicker/cli_options.py", "DEFLICKER_OPTIONS")
(ESC_OPTIONS,) = _load_contribution("esc/cli_options.py", "ESC_OPTIONS")
(
    FASTDVDNET_OPTIONS,
    FASTDVDNET_STRENGTH_OPTIONS,
) = _load_contribution(
    "fastdvdnet/cli_options.py",
    "FASTDVDNET_OPTIONS",
    "FASTDVDNET_STRENGTH_OPTIONS",
)
FBCNN_OPTIONS, FBCNN_WEIGHT_OPTIONS = _load_contribution(
    "fbcnn/cli_options.py", "FBCNN_OPTIONS", "FBCNN_WEIGHT_OPTIONS")
MC_OPTIONS, MC_STRENGTH_OPTIONS = _load_contribution(
    "mc/cli_options.py", "MC_OPTIONS", "MC_STRENGTH_OPTIONS")
(METALFX_OPTIONS,) = _load_contribution("metalfx/cli_options.py", "METALFX_OPTIONS")
(NAFNET_OPTIONS,) = _load_contribution("nafnet/cli_options.py", "NAFNET_OPTIONS")
PVDD_OPTIONS, PVDD_STRENGTH_OPTIONS = _load_contribution(
    "pvdd/cli_options.py", "PVDD_OPTIONS", "PVDD_STRENGTH_OPTIONS")
(REALBASICVSR_OPTIONS,) = _load_contribution(
    "realbasicvsr/cli_options.py", "REALBASICVSR_OPTIONS")
(REALESRGAN_OPTIONS,) = _load_contribution(
    "realesrgan/cli_options.py", "REALESRGAN_OPTIONS")
(REALPLKSR_OPTIONS,) = _load_contribution(
    "realplksr/cli_options.py", "REALPLKSR_OPTIONS")
(REALVIFORMER_OPTIONS,) = _load_contribution(
    "realviformer/cli_options.py", "REALVIFORMER_OPTIONS")
(SAFMN_OPTIONS,) = _load_contribution("safmn/cli_options.py", "SAFMN_OPTIONS")
(SANITIZE_EDGE_OPTIONS,) = _load_contribution(
    "sanitize_edges/cli_options.py", "SANITIZE_EDGE_OPTIONS")
(SPATIAL_OPTIONS,) = _load_contribution("spatial/cli_options.py", "SPATIAL_OPTIONS")
(SQUARE_PIXELS_OPTIONS,) = _load_contribution(
    "square_pixels/cli_options.py", "SQUARE_PIXELS_OPTIONS")
(STDF_OPTIONS,) = _load_contribution("stdf/cli_options.py", "STDF_OPTIONS")
(
    TOFLOW_OPTIONS,
    TOFLOW_STRENGTH_OPTIONS,
    TOFLOW_UPSCALE_OPTIONS,
) = _load_contribution(
    "toflow/cli_options.py",
    "TOFLOW_OPTIONS",
    "TOFLOW_STRENGTH_OPTIONS",
    "TOFLOW_UPSCALE_OPTIONS",
)

REGISTRY = [
    *SOURCE_AND_OUTPUT_OPTIONS,
    *SANITIZE_EDGE_OPTIONS,
    *CROP_OPTIONS,
    *SQUARE_PIXELS_OPTIONS,
    *CUT_OPTIONS,
    *PREPROCESS_SLOT_OPTIONS,
    *BASICVSRPP_RESTORE_OPTIONS,
    *DEFLICKER_OPTIONS,
    *DEBLOCK_SLOT_OPTIONS,
    *STDF_OPTIONS,
    *FBCNN_WEIGHT_OPTIONS,
    *DEBLOCK_STRENGTH_OPTIONS,
    *DEBLOCK_MAP_OPTIONS,
    *FBCNN_OPTIONS,
    *DENOISE_SLOT_OPTIONS,
    *SPATIAL_OPTIONS,
    *MC_STRENGTH_OPTIONS,
    *FASTDVDNET_STRENGTH_OPTIONS,
    *BSVD_STRENGTH_OPTIONS,
    *TOFLOW_STRENGTH_OPTIONS,
    *PVDD_STRENGTH_OPTIONS,
    *FASTDVDNET_OPTIONS,
    *BSVD_OPTIONS,
    *TOFLOW_OPTIONS,
    *PVDD_OPTIONS,
    *NOISE_MAP_OPTIONS,
    *MC_OPTIONS,
    *TOFLOW_UPSCALE_OPTIONS,
    *NAFNET_OPTIONS,
    *UPSCALE_SLOT_OPTIONS,
    *METALFX_OPTIONS,
    *BASICVSRPP_UPSCALE_OPTIONS,
    *REALBASICVSR_OPTIONS,
    *REALESRGAN_OPTIONS,
    *REALVIFORMER_OPTIONS,
    *ESC_OPTIONS,
    *REALPLKSR_OPTIONS,
    *SAFMN_OPTIONS,
    *RUNTIME_OPTIONS,
]
