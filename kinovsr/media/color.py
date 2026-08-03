"""Source color detection + propagation.

The encoder must tag the output to match the *source* color, not a hard-coded
BT.709. Tagged containers carry explicit primaries/transfer/matrix; we read and
propagate those (reliable). Untagged containers are genuinely ambiguous -- no
tool can know for sure -- so we fall back to a documented default (BT.709,
overridable) rather than silently inheriting VideoToolbox's decode-time guess.

The container's full-range flag is a real field (present even when color is
untagged), so range is always read directly.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kinovsr.native.frameworks import CoreMedia, Quartz, av

# --- canonical CV constant triples (primaries, transfer, matrix) -------------
_709 = (Quartz.kCVImageBufferColorPrimaries_ITU_R_709_2,
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2)
_601 = (Quartz.kCVImageBufferColorPrimaries_SMPTE_C,            # NTSC 601 primaries
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,      # same SDR gamma as 709
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4)
_2020 = (Quartz.kCVImageBufferColorPrimaries_ITU_R_2020,
         Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,     # SDR 2020 uses ~709 gamma
         Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020)
_OVERRIDES = {"bt709": _709, "bt601": _601, "bt2020": _2020}
_FRAME_PRIMARIES = {
    "smpte_c": Quartz.kCVImageBufferColorPrimaries_SMPTE_C,
    "bt709": Quartz.kCVImageBufferColorPrimaries_ITU_R_709_2,
    "bt2020": Quartz.kCVImageBufferColorPrimaries_ITU_R_2020,
}
_FRAME_TRANSFERS = {
    "bt709": Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,
    # Product policy models SDR BT.2020 with the encoder-supported 709-family
    # transfer curve, matching _2020 and ffmpeg_reader._TRANSFER.
    "bt2020": Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,
}
_FRAME_MATRICES = {
    "bt601": Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4,
    "bt709": Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2,
    "bt2020": Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020,
}

# AV writer constants, matched to a CV value by shared CFString value (with a
# BT.709 fallback for anything AV doesn't expose, e.g. the SMPTE_C transfer).
# Keep this list explicit: ``dir(AVFoundation)`` asks PyObjC to enumerate the
# whole lazy framework and adds seconds to every process that imports FileSink.
def _available_av_constants(*names: str) -> tuple[Any, ...]:
    return tuple(
        value
        for name in names
        if (value := getattr(av, name, None)) is not None
    )


_AV_PRIMS = _available_av_constants(
    "AVVideoColorPrimaries_EBU_3213",
    "AVVideoColorPrimaries_ITU_R_2020",
    "AVVideoColorPrimaries_ITU_R_709_2",
    "AVVideoColorPrimaries_P3_D65",
    "AVVideoColorPrimaries_SMPTE_C",
)
_AV_TRANS = _available_av_constants(
    "AVVideoTransferFunction_IEC_sRGB",
    "AVVideoTransferFunction_ITU_R_2100_HLG",
    "AVVideoTransferFunction_ITU_R_709_2",
    "AVVideoTransferFunction_Linear",
    "AVVideoTransferFunction_SMPTE_240M_1995",
    "AVVideoTransferFunction_SMPTE_ST_2084_PQ",
)
_AV_MATS = _available_av_constants(
    "AVVideoYCbCrMatrix_ITU_R_2020",
    "AVVideoYCbCrMatrix_ITU_R_601_4",
    "AVVideoYCbCrMatrix_ITU_R_709_2",
    "AVVideoYCbCrMatrix_SMPTE_240M_1995",
)


def _match(cv_val: Any, av_list: Sequence[Any], fallback: Any) -> Any:
    return next((a for a in av_list if a == cv_val), fallback)


def read_source_color(format_desc: Any) -> dict:
    """Explicit color tags from a CM format description (None where untagged)."""
    ext = CoreMedia.CMFormatDescriptionGetExtensions(format_desc) or {}
    by = {str(k): ext[k] for k in ext}
    prim = by.get("CVImageBufferColorPrimaries")
    return {
        "primaries": prim,
        "transfer": by.get("CVImageBufferTransferFunction"),
        "matrix": by.get("CVImageBufferYCbCrMatrix"),
        "full_range": bool(by.get("FullRangeVideo", False)),
        "tagged": prim is not None,
    }


def resolve(src: dict, override: str = "auto", range_override: str = "auto") -> tuple:
    """(primaries, transfer, matrix, full_range) as CV constants.

    override 'auto' = the source's actual color: explicit container tags, or (for
    untagged sources) VideoToolbox's decode-time guess, which probe_color reads
    from a decoded frame -- so the output tag matches how the source was actually
    read, not a fixed BT.709. Falls back to BT.709 only if even the guess is
    absent. Any other override forces that colorimetry.

    range_override 'auto' = the container's full-range flag (absent = video
    range, the universal default for H.264/HEVC). 'video'/'full' force the
    interpretation for mis-flagged sources -- e.g. full-range screen recordings
    an encoder left untagged, which would otherwise be read as limited range
    (crushed shadows / blown highlights once expanded).
    """
    fr = src["full_range"] if range_override == "auto" else range_override == "full"
    if override != "auto":
        prim, trans, mat = _OVERRIDES[override]
        return prim, trans, mat, fr
    return (
        src.get("primaries") or _709[0],
        src.get("transfer") or _709[1],
        src.get("matrix") or _709[2],
        fr,
    )


def resolve_frame_spec(frame: Any) -> tuple:
    """Translate the complete typed FrameSpec tuple under the SDR color policy."""
    try:
        return (
            _FRAME_PRIMARIES[frame.color_primaries.value],
            _FRAME_TRANSFERS[frame.transfer_function.value],
            _FRAME_MATRICES[frame.color_matrix.value],
            frame.color_range.value == "full",
        )
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported typed frame color metadata: {frame!r}") from exc


def cv_triple(resolved: tuple) -> tuple:
    """Just the (primaries, transfer, matrix) CV constants (for VTPixelTransfer)."""
    return resolved[0], resolved[1], resolved[2]


def av_color_properties(resolved: tuple) -> dict:
    """AVVideoColorPropertiesKey dict for the writer, mapped from CV constants."""
    prim, trans, mat, _ = resolved
    return {
        av.AVVideoColorPrimariesKey: _match(prim, _AV_PRIMS, av.AVVideoColorPrimaries_ITU_R_709_2),
        av.AVVideoTransferFunctionKey: _match(
            trans, _AV_TRANS, av.AVVideoTransferFunction_ITU_R_709_2),
        av.AVVideoYCbCrMatrixKey: _match(mat, _AV_MATS, av.AVVideoYCbCrMatrix_ITU_R_709_2),
    }


def describe(resolved: tuple) -> str:
    prim, _, mat, fr = resolved
    p = str(prim).replace("ITU_R_", "").replace("_2", "").replace("SMPTE_C", "601-C")
    m = str(mat).replace("ITU_R_", "").replace("_4", "").replace("_2", "")
    return f"primaries={p} matrix={m} range={'full' if fr else 'video'}"
