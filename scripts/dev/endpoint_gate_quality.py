"""Native media and tolerance-based quality probes for endpoint gates."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import zlib
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from endpoint_gate_protocol import (
    _MISSING,
    QUALITY_MAX_ABS_RGB10,
    QUALITY_MIN_PSNR_DB,
    _jsonable,
    _signature_mismatches,
)


def _fraction(value: Fraction | None) -> str | None:
    return None if value is None else f"{value.numerator}/{value.denominator}"


def _codec_details(format_description: Any) -> dict[str, Any]:
    from kinovsr.native.frameworks import CoreMedia

    extensions = CoreMedia.CMFormatDescriptionGetExtensions(format_description) or {}
    bits = extensions.get("BitsPerComponent")
    details: dict[str, Any] = {
        "bits_per_component": None if bits is None else int(bits),
    }
    atoms = extensions.get("SampleDescriptionExtensionAtoms") or {}
    configuration = None
    atom_name = None
    for candidate in ("hvcC", "avcC"):
        value = atoms.get(candidate)
        if value is not None:
            atom_name = candidate
            configuration = bytes(value)
            break
    if configuration is None:
        details["configuration_atom"] = None
        return details

    details.update(
        {
            "configuration_atom": atom_name,
            "configuration_sha256": hashlib.sha256(configuration).hexdigest(),
        }
    )
    if atom_name == "hvcC" and len(configuration) >= 19:
        profile_idc = configuration[1] & 0x1F
        chroma_idc = configuration[16] & 0x03
        details["hevc"] = {
            "profile_idc": profile_idc,
            "profile": {
                1: "main",
                2: "main10",
                3: "main_still_picture",
                4: "range_extensions",
            }.get(profile_idc, "unknown"),
            "tier": "high" if configuration[1] & 0x20 else "main",
            "level_idc": configuration[12],
            "chroma_format_idc": chroma_idc,
            "chroma": {0: "monochrome", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}[chroma_idc],
            "bit_depth_luma": 8 + (configuration[17] & 0x07),
            "bit_depth_chroma": 8 + (configuration[18] & 0x07),
        }
    return details


def _track_metadata(path: Path | str) -> dict[str, Any]:
    from kinovsr.media.video_reader import (
        _first_video_track,
        _video_codec_fourcc,
        probe_color,
        probe_video,
        probe_video_timing,
    )
    from kinovsr.native.frameworks import Foundation, av

    media_path = Path(path)
    url = Foundation.NSURL.fileURLWithPath_(str(media_path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    format_description = track.formatDescriptions()[0]
    width, height, nominal_fps, _, transform, pixel_aspect = probe_video(media_path)
    timing = probe_video_timing(media_path)
    return {
        "width": width,
        "height": height,
        "nominal_fps": nominal_fps,
        "sample_count": timing.sample_count,
        "cadence": _fraction(timing.cadence),
        "first_pts": _fraction(timing.first_pts),
        "duration": _fraction(timing.duration),
        "source_tick": _fraction(timing.source_tick),
        "codec_fourcc": _video_codec_fourcc(track),
        "codec_details": _codec_details(format_description),
        "pixel_aspect": (None if pixel_aspect is None else f"{pixel_aspect[0]}/{pixel_aspect[1]}"),
        "transform": {
            key: float(getattr(transform, key)) for key in ("a", "b", "c", "d", "tx", "ty")
        },
        "color": _jsonable(probe_color(media_path)),
    }


def _encode_rgb10_frame(frame: Any) -> str:
    import numpy as np

    raw = np.ascontiguousarray(frame, dtype="<u2").tobytes()
    return base64.b64encode(zlib.compress(raw, level=6)).decode("ascii")


def _decoded_quality_probe(path: Path, indices: Sequence[int]) -> dict[str, Any]:
    """Retain compressed full RGB10 frames for tolerance-based comparison."""
    import numpy as np

    from kinovsr.media import ffmpeg_reader
    from kinovsr.media import pixel_buffers as pb

    digest = hashlib.sha256()
    samples = []
    for index in indices:
        chunks = ffmpeg_reader.iter_video_buffer_chunks(
            path,
            pb.PIX_RGBAHALF,
            chunk_size=1,
            start_frame=index,
            end_frame=index + 1,
        )
        try:
            buffer = next(iter(chunks))[0]
        except (StopIteration, IndexError) as exc:
            raise RuntimeError(f"cannot decode quality-probe frame {index}") from exc
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        rgb = np.asarray(pb.read_buffer_rgb_f32(buffer), dtype=np.float32)
        clipped = np.clip(rgb, 0.0, 1.0)
        quantized = np.rint(clipped * 1023.0).astype("<u2")
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(np.ascontiguousarray(quantized).tobytes())
        samples.append(
            {
                "index": index,
                "shape": list(quantized.shape),
                "rgb10_zlib_b64": _encode_rgb10_frame(quantized),
                "mean_rgb": [round(float(value), 8) for value in clipped.mean((0, 1))],
                "std_rgb": [round(float(value), 8) for value in clipped.std((0, 1))],
            }
        )
    return {
        "method": "decoded-rgb10-full-v4",
        "decoder": "PyAV/libav rgb48le to RGBAHalf",
        "indices": list(indices),
        "compression": "zlib",
        "diagnostic_full_frame_sha256": digest.hexdigest(),
        "samples": samples,
    }


def _quality_indices(
    *,
    warmup_frames: int,
    measured_frames: int,
    total_frames: int,
) -> list[int]:
    return sorted(
        {
            0,
            warmup_frames - 1,
            warmup_frames,
            warmup_frames + measured_frames - 1,
            total_frames - 1,
        }
    )


def _output_probe(
    path: Path,
    *,
    warmup_frames: int,
    measured_frames: int,
    total_frames: int,
) -> dict[str, Any]:
    indices = _quality_indices(
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
        total_frames=total_frames,
    )
    quality = _decoded_quality_probe(path, indices)
    return {
        "track": _track_metadata(path),
        "quality": quality,
    }


def _decode_rgb10_frame(sample: Mapping[str, Any]) -> Any:
    import numpy as np

    shape = sample.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or value < 1 for value in shape)
        or shape[2] != 3
    ):
        raise ValueError(f"invalid RGB10 frame shape {shape!r}")
    encoded = sample.get("rgb10_zlib_b64")
    if not isinstance(encoded, str):
        raise ValueError("RGB10 frame payload must be base64 text")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        raw = zlib.decompress(compressed)
    except (binascii.Error, ValueError, zlib.error) as exc:
        raise ValueError("RGB10 frame payload is not valid zlib/base64") from exc
    expected_bytes = math.prod(shape) * 2
    if len(raw) != expected_bytes:
        raise ValueError(f"RGB10 frame has {len(raw)} bytes, expected {expected_bytes} for {shape}")
    return np.frombuffer(raw, dtype="<u2").reshape(shape)


def _compare_quality(
    baseline: Any,
    current: Any,
) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(baseline, Mapping) or not isinstance(current, Mapping):
        return ["output_behavior.quality: baseline and current must be objects"], {}

    mismatches: list[str] = []
    required = (
        "method",
        "decoder",
        "indices",
        "compression",
        "diagnostic_full_frame_sha256",
        "samples",
    )
    for side, value in (("baseline", baseline), ("current", current)):
        for field in required:
            if field not in value:
                mismatches.append(f"output_behavior.quality.{field}: missing from {side}")
    if mismatches:
        return mismatches, {}

    stable = ("method", "decoder", "indices", "compression")
    mismatches.extend(
        _signature_mismatches(
            {key: baseline[key] for key in stable},
            {key: current[key] for key in stable},
            prefix="output_behavior.quality",
        )
    )
    baseline_samples = baseline["samples"]
    current_samples = current["samples"]
    if not isinstance(baseline_samples, list) or not isinstance(current_samples, list):
        mismatches.append("output_behavior.quality.samples: must be lists")
        return mismatches, {}
    if len(baseline_samples) != len(current_samples):
        mismatches.append(
            "output_behavior.quality.samples.length: "
            f"baseline={len(baseline_samples)}, current={len(current_samples)}"
        )

    import numpy as np

    comparisons = []
    for position, (left, right) in enumerate(zip(baseline_samples, current_samples, strict=False)):
        prefix = f"output_behavior.quality.samples[{position}]"
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            mismatches.append(f"{prefix}: baseline and current must be objects")
            continue
        structural = _signature_mismatches(
            {key: left.get(key, _MISSING) for key in ("index", "shape")},
            {key: right.get(key, _MISSING) for key in ("index", "shape")},
            prefix=prefix,
        )
        if structural:
            mismatches.extend(structural)
            continue
        try:
            left_frame = _decode_rgb10_frame(left)
            right_frame = _decode_rgb10_frame(right)
        except ValueError as exc:
            mismatches.append(f"{prefix}: {exc}")
            continue
        difference = np.abs(left_frame.astype(np.int32) - right_frame.astype(np.int32))
        max_abs = int(difference.max(initial=0))
        mse = float(np.mean(np.square(difference, dtype=np.float64)))
        psnr_db = None if mse == 0.0 else 20.0 * math.log10(1023.0 / math.sqrt(mse))
        sample_pass = max_abs <= QUALITY_MAX_ABS_RGB10 and (
            psnr_db is None or psnr_db >= QUALITY_MIN_PSNR_DB
        )
        comparisons.append(
            {
                "index": left["index"],
                "max_abs_rgb10": max_abs,
                "psnr_db": None if psnr_db is None else round(psnr_db, 4),
                "pass": sample_pass,
            }
        )
        if not sample_pass:
            rendered_psnr = "exact" if psnr_db is None else f"{psnr_db:.4f} dB"
            mismatches.append(
                f"{prefix}: max_abs_rgb10={max_abs} (limit "
                f"{QUALITY_MAX_ABS_RGB10}), psnr={rendered_psnr} "
                f"(minimum {QUALITY_MIN_PSNR_DB:.1f} dB)"
            )

    metrics = {
        "policy": {
            "max_abs_rgb10": QUALITY_MAX_ABS_RGB10,
            "min_psnr_db": QUALITY_MIN_PSNR_DB,
        },
        "diagnostic_full_frame_hash_match": (
            baseline["diagnostic_full_frame_sha256"] == current["diagnostic_full_frame_sha256"]
        ),
        "samples": comparisons,
    }
    return mismatches, metrics


def _compare_output_behavior(
    baseline: Any,
    current: Any,
) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(baseline, Mapping) or not isinstance(current, Mapping):
        return ["output_behavior: baseline and current must be objects"], {}
    baseline_metadata = {key: value for key, value in baseline.items() if key != "quality"}
    current_metadata = {key: value for key, value in current.items() if key != "quality"}
    mismatches = _signature_mismatches(
        baseline_metadata,
        current_metadata,
        prefix="output_behavior",
    )
    quality_mismatches, quality_metrics = _compare_quality(
        baseline.get("quality"),
        current.get("quality"),
    )
    mismatches.extend(quality_mismatches)
    return mismatches, quality_metrics
