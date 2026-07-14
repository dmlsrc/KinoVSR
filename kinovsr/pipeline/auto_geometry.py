"""Probe-time auto geometry: resolve ``bars="auto"`` / ``edges="auto"``.

The typed pipeline validates geometry at preflight, so a crop or
sanitize stage must declare literal counts before ``resolve_pipeline``
sees the config. This pass is the probe the crop family's docstring
promises: it samples early frames (the harness's recipe - frames
0,4,8,12,16,20 of the file head), runs the same detectors
(``detect_bars`` / ``detect_junk_edges``), and rewrites the stage tables
into resolved literal counts. A stage that detects nothing and would do
nothing else is removed from the pipeline, matching the harness's
"auto: nothing detected" behavior.

Detection is defined against SOURCE-resolution pixels: as the walk
passes each crop stage it applies that stage's resolved crop to the
samples (bars and aspect window alike), so a later sanitize stage
detects on the same post-crop picture the harness fed its detectors.
An auto value downstream of any non-geometry stage (a scaler, a
denoiser) is rejected - the samples no longer describe what that stage
would see.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kinovsr.config import ConfigError
from kinovsr.config.merge import split_stage_table

_log = logging.getLogger(__name__)

_GEOMETRY_FAMILIES = ("crop", "sanitize_edges")
# The harness's sample recipe: 6 frames spread over the first 24.
_SAMPLE_INDICES = frozenset(range(0, 24, 4))


def wants_auto_geometry(config: dict) -> bool:
    """Does any crop/sanitize stage table carry an ``auto`` value?"""
    for name in config.get("pipeline") or []:
        table = config.get(name) or {}
        family = table.get("processor")
        if family == "crop" and ("auto" in (table.get("bars"),
                                          table.get("trim"))):
            return True
        if family == "sanitize_edges" and table.get("edges") == "auto":
            return True
    return False


def _gather_samples(
    video: Path,
    vr: Any,
    pb: Any,
    *,
    chunk_size: int,
) -> list:
    """Decode the harness's sample frames as float32 [0,1] RGB arrays."""
    import mlx.core as mx

    samples: list = []
    index = 0
    for chunk in vr.iter_video_buffer_chunks(
            video, pb.PIX_RGBAHALF, chunk_size=chunk_size):
        for buffer in chunk:
            if index in _SAMPLE_INDICES:
                samples.append(mx.clip(
                    pb.read_pixel_buffer_rgb(buffer).astype(mx.float32)
                    / 255.0, 0, 1))
            index += 1
        if index > max(_SAMPLE_INDICES):
            return samples
    return samples


def _crop_samples(samples: list, crop: tuple[int, int, int, int]) -> list:
    from kinovsr.processors.crop.geometry import crop_rgb

    if not any(crop):
        return samples
    return [crop_rgb(sample, crop) for sample in samples]


def _stage_combined_crop(table: dict, samples: list,
                         pixel_aspect: Any) -> tuple[int, int, int, int]:
    """The full (bars + aspect) crop a resolved crop stage applies, computed
    by the crop family's own math so the probe never drifts from it."""
    from fractions import Fraction

    from kinovsr.processors.crop import FACTORY, _combined_crop

    _selector, family_settings = split_stage_table(table)
    parsed = FACTORY.parse_config(
        family_settings, capability=next(iter(FACTORY.capabilities)),
        profile=None, settings=None)
    height, width = int(samples[0].shape[0]), int(samples[0].shape[1])
    return _combined_crop(width, height, Fraction(pixel_aspect), parsed)


def resolve_auto_geometry(
    config: dict,
    *,
    video: Path,
    vr: Any,
    pixel_aspect: Any = 1,
    chunk_size: int = 8,
) -> dict:
    """Return a config copy with every auto crop/sanitize stage resolved.

    ``vr`` is the reader module the run decodes with (native or ffmpeg),
    so the probe samples through the same decoder. ``pixel_aspect`` is
    the source PAR (aspect windows on anamorphic sources fold it in).
    """
    from kinovsr.analysis.edges import detect_bars, detect_junk_edges
    from kinovsr.media import pixel_buffers as pb

    samples = _gather_samples(
        Path(video), vr, pb, chunk_size=chunk_size)
    resolved = dict(config)
    pipeline: list[str] = []
    scaled_past = None   # first non-geometry stage seen, if any
    for name in config.get("pipeline") or []:
        table = dict(config.get(name) or {})
        family = table.get("processor")
        is_auto = ((family == "crop"
                    and "auto" in (table.get("bars"), table.get("trim")))
                   or (family == "sanitize_edges"
                       and table.get("edges") == "auto"))
        if family not in _GEOMETRY_FAMILIES:
            if scaled_past is None:
                scaled_past = name
            pipeline.append(name)
            continue
        if is_auto and scaled_past is not None:
            raise ConfigError(
                f"[{name}] auto detection samples the SOURCE picture; it "
                f"cannot follow stage '{scaled_past}', which changes what "
                f"this stage would see - state literal counts or move the "
                f"stage before it")
        if family == "crop" and (table.get("bars") == "auto"
                                 or table.get("trim") == "auto"):
            if table.get("bars") == "auto":
                bars = detect_bars(samples)
                if any(bars):
                    table["bars"] = ",".join(str(b) for b in bars)
                    _log.info("[crop] auto: bars top=%d bottom=%d left=%d "
                              "right=%d px", *bars)
                else:
                    del table["bars"]
                    _log.info("[crop] auto: no bars detected")
            if table.get("trim") == "auto":
                # Junk-edge TRIM (the harness's sanitize fill="trim"):
                # detect on the post-bars picture, keep the active area
                # even (the harness's +1px bump into the content), fold
                # into bars.
                from kinovsr.config.helpers import parse_edge_counts

                bars_now = (parse_edge_counts(table["bars"])
                            if table.get("bars") else (0, 0, 0, 0))
                sub = _crop_samples(samples, bars_now)
                edges, notices = detect_junk_edges(sub)
                for note in notices:
                    _log.info("[sanitize] %s", note)
                if any(edges) and sub:
                    te = list(edges)
                    h = int(sub[0].shape[0])
                    w = int(sub[0].shape[1])
                    if (h - te[0] - te[1]) % 2:
                        te[1] += 1
                    if (w - te[2] - te[3]) % 2:
                        te[3] += 1
                    folded = tuple(b + t for b, t
                                   in zip(bars_now, te, strict=True))
                    table["bars"] = ",".join(str(b) for b in folded)
                    _log.info("[sanitize] trim: top=%d bottom=%d left=%d "
                              "right=%d px cropped off", *te)
                else:
                    _log.info("[sanitize] trim: no junk edges detected")
                del table["trim"]
            if not (table.get("bars") or table.get("trim")
                    or table.get("aspect")):
                _log.info("[crop] auto: nothing detected; stage %r removed",
                          name)
                resolved.pop(name, None)
                continue
            resolved[name] = table
        elif family == "sanitize_edges" and table.get("edges") == "auto":
            edges, notices = detect_junk_edges(samples)
            for note in notices:
                _log.info("[sanitize] %s", note)
            if any(edges):
                table["edges"] = ",".join(str(e) for e in edges)
                _log.info("[sanitize] auto: junk edges top=%d bottom=%d "
                          "left=%d right=%d px", *edges)
                resolved[name] = table
            else:
                _log.info("[sanitize] auto: no junk edges detected; "
                          "stage %r removed", name)
                resolved.pop(name, None)
                continue
        if family == "crop" and samples:
            # Later detection sees this stage's output picture, exactly as
            # the harness cropped its samples between detector passes.
            samples = _crop_samples(
                samples, _stage_combined_crop(table, samples, pixel_aspect))
        pipeline.append(name)
    resolved["pipeline"] = pipeline
    return resolved


__all__ = ["resolve_auto_geometry", "wants_auto_geometry"]
