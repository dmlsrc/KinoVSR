#!/usr/bin/env python3
"""Score VSR/denoiser variants on detected face regions.

This is a developer evaluation tool. Local video paths belong in
`scripts/vsr_eval/vsr_eval.local.toml`; the checked-in defaults and examples
stay machine-neutral.

Examples:
    scripts/vsr_eval/face_yunet_metrics.py \\
        --config scripts/vsr_eval/vsr_eval.local.toml

    scripts/vsr_eval/face_yunet_metrics.py \\
        --variants-json "$SHARED_TEMP_DIR/run/variants.json" \\
        --out-dir "$SHARED_TEMP_DIR/run/face_eval"
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2 as cv
from config import (
    config_get,
    config_section,
    default_shared_temp,
    load_config,
    resolve_path,
)

if TYPE_CHECKING:
    import numpy as np

np: Any = None


def _load_numpy() -> Any:
    global np
    if np is None:
        try:
            import numpy as _np
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "NumPy is required for scripts/vsr_eval/face_yunet_metrics.py. "
                "Install the developer extras for evaluation tools."
            ) from exc
        np = _np
    return np


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = TOOL_DIR / "weights" / "face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class FaceObs:
    frame: int
    track: int
    box: tuple[int, int, int, int]
    score: float


def read_frames(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            frames.append(rgb.astype(np.float32) * (1.0 / 255.0))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def clamp_box(
    box: tuple[float, float, float, float],
    w: int,
    h: int,
    pad: float = 0.28,
) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    cx = x + bw * 0.5
    cy = y + bh * 0.5
    side = max(bw, bh) * (1.0 + 2.0 * pad)
    x0 = max(0, int(round(cx - side * 0.5)))
    y0 = max(0, int(round(cy - side * 0.5)))
    x1 = min(w, int(round(cx + side * 0.5)))
    y1 = min(h, int(round(cy + side * 0.5)))
    return x0, y0, x1, y1


def scale_box(
    box: tuple[int, int, int, int],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    sh, sw = source_shape
    th, tw = target_shape
    if (sh, sw) == (th, tw):
        return box
    sx = tw / max(sw, 1)
    sy = th / max(sh, 1)
    x0, y0, x1, y1 = box
    return (
        max(0, min(tw, int(round(x0 * sx)))),
        max(0, min(th, int(round(y0 * sy)))),
        max(0, min(tw, int(round(x1 * sx)))),
        max(0, min(th, int(round(y1 * sy)))),
    )


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    den = area_a + area_b - inter
    return 0.0 if den <= 0 else inter / den


def detect_faces(frames: list[np.ndarray], model: Path, threshold: float, min_side: int) -> list[FaceObs]:
    if not model.exists():
        raise SystemExit(
            f"missing YuNet model: {model}\n"
            "Use --model or set face_eval.model in scripts/vsr_eval/vsr_eval.local.toml."
        )

    h, w = frames[0].shape[:2]
    det = cv.FaceDetectorYN_create(str(model), "", (w, h), threshold, 0.3, 5000)
    det.setInputSize((w, h))

    observations: list[FaceObs] = []
    active: dict[int, tuple[int, tuple[int, int, int, int]]] = {}
    next_track = 1
    max_gap = 8

    for frame_idx, rgb in enumerate(frames):
        bgr_u8 = cv.cvtColor((rgb * 255.0).clip(0, 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        _, faces = det.detect(bgr_u8)
        candidates: list[tuple[tuple[int, int, int, int], float]] = []
        if faces is not None:
            for face in faces:
                x, y, bw, bh = [float(v) for v in face[:4]]
                score = float(face[14])
                if min(bw, bh) < min_side:
                    continue
                box = clamp_box((x, y, bw, bh), w, h)
                if box[2] - box[0] < min_side or box[3] - box[1] < min_side:
                    continue
                candidates.append((box, score))

        used_tracks: set[int] = set()
        for box, score in sorted(candidates, key=lambda item: item[1], reverse=True):
            best_tid = None
            best_iou = 0.0
            for tid, (last_frame, last_box) in active.items():
                if tid in used_tracks or frame_idx - last_frame > max_gap:
                    continue
                val = iou(box, last_box)
                if val > best_iou:
                    best_iou = val
                    best_tid = tid
            if best_tid is None or best_iou < 0.10:
                best_tid = next_track
                next_track += 1
            used_tracks.add(best_tid)
            active[best_tid] = (frame_idx, box)
            observations.append(FaceObs(frame_idx, best_tid, box, score))

        active = {tid: val for tid, val in active.items() if frame_idx - val[0] <= max_gap}

    return observations


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def chroma(rgb: np.ndarray) -> np.ndarray:
    return rgb - luma(rgb)[..., None]


def box3_2d(x: np.ndarray) -> np.ndarray:
    return cv.blur(x.astype(np.float32), (3, 3), borderType=cv.BORDER_REFLECT)


def box3(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return box3_2d(x)
    return np.stack([box3_2d(x[..., c]) for c in range(x.shape[-1])], axis=-1)


def hf(x: np.ndarray) -> np.ndarray:
    return x - box3(x)


def grad_mag(y: np.ndarray) -> np.ndarray:
    gx = cv.Sobel(y.astype(np.float32), cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(y.astype(np.float32), cv.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    den = math.sqrt(float(np.dot(av, av) * np.dot(bv, bv)))
    return 0.0 if den <= 1e-12 else float(np.dot(av, bv) / den)


def ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    ax = luma(a).astype(np.float64)
    bx = luma(b).astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    ma = ax.mean()
    mb = bx.mean()
    va = ax.var()
    vb = bx.var()
    cov = ((ax - ma) * (bx - mb)).mean()
    return float(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2)))


def row_score(y: np.ndarray) -> float:
    prof = y.mean(axis=1)
    smooth = cv.blur(prof[:, None].astype(np.float32), (1, 7), borderType=cv.BORDER_REFLECT)[:, 0]
    prof = prof - smooth
    mag = np.abs(np.fft.rfft(prof))
    freqs = np.fft.rfftfreq(len(prof))
    mask = (freqs >= 1.0 / 32.0) & (freqs <= 1.0 / 3.0)
    if not np.any(mask):
        return 0.0
    den = np.sqrt(np.mean(prof * prof)) * len(prof) + 1e-8
    return float(np.max(mag[mask]) / den)


def resize_crop(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    source_shape: tuple[int, int],
    side: int = 112,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = scale_box(box, source_shape, (h, w))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((side, side, 3), dtype=np.float32)
    return cv.resize(crop, (side, side), interpolation=cv.INTER_AREA).astype(np.float32)


def crop_stats(crops: list[np.ndarray]) -> dict[str, float]:
    ys = [luma(c) for c in crops]
    hfl = [float(np.sqrt(np.mean(hf(y) ** 2))) for y in ys]
    chf = [float(np.sqrt(np.mean(hf(chroma(c)) ** 2))) for c in crops]
    grad = [float(np.mean(grad_mag(y))) for y in ys]
    rows = [row_score(y) for y in ys]
    temp = [float(np.mean(np.abs(hf(ys[i]) - hf(ys[i - 1])))) for i in range(1, len(ys))]
    return {
        "hf_luma": float(np.mean(hfl)),
        "chroma_hf": float(np.mean(chf)),
        "grad": float(np.mean(grad)),
        "row_score": float(np.mean(rows)),
        "temporal_hf_diff": float(np.mean(temp)) if temp else 0.0,
    }


def per_crop_metrics(base: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    by = luma(base)
    cy = luma(cand)
    bg = grad_mag(by)
    cg = grad_mag(cy)
    return {
        "ssim_luma": ssim_luma(base, cand),
        "grad_corr": corr(bg, cg),
        "mae_luma": float(np.mean(np.abs(cy - by))),
        "chroma_drift": float(np.sqrt(np.mean((chroma(cand) - chroma(base)) ** 2))),
    }


def aggregate_variant(
    baseline_frames: list[np.ndarray],
    candidate_frames: list[np.ndarray],
    observations: list[FaceObs],
) -> dict[str, Any]:
    n = min(len(baseline_frames), len(candidate_frames))
    obs = [o for o in observations if o.frame < n]
    if not obs:
        raise RuntimeError("no face observations overlap this candidate")

    source_shape = baseline_frames[0].shape[:2]
    base_crops = [resize_crop(baseline_frames[o.frame], o.box, source_shape) for o in obs]
    cand_crops = [resize_crop(candidate_frames[o.frame], o.box, source_shape) for o in obs]
    base_stats = crop_stats(base_crops)
    cand_stats = crop_stats(cand_crops)

    per = [per_crop_metrics(b, c) for b, c in zip(base_crops, cand_crops, strict=True)]
    out: dict[str, Any] = {
        "face_samples": len(obs),
        "tracks": len({o.track for o in obs}),
    }
    for key in per[0]:
        out[key] = float(np.mean([m[key] for m in per]))
    for key in ("hf_luma", "chroma_hf", "grad", "row_score", "temporal_hf_diff"):
        out[key] = cand_stats[key]
        out[key + "_ratio"] = cand_stats[key] / max(base_stats[key], 1e-8)

    cleanup = (
        0.28 * (1.0 - out["chroma_hf_ratio"])
        + 0.24 * (1.0 - out["temporal_hf_diff_ratio"])
        + 0.20 * (1.0 - out["row_score_ratio"])
        + 0.12 * (1.0 - out["hf_luma_ratio"])
    )
    preservation_penalty = (
        2.0 * max(0.0, 0.970 - out["grad_ratio"])
        + 2.2 * max(0.0, 0.988 - out["ssim_luma"])
        + 0.7 * max(0.0, 0.988 - out["grad_corr"])
        + 8.0 * max(0.0, out["mae_luma"] - 0.006)
        + 8.0 * max(0.0, out["chroma_drift"] - 0.006)
    )
    out["face_score"] = 100.0 * (cleanup - preservation_penalty)
    return out


def track_summary(observations: list[FaceObs]) -> list[dict[str, Any]]:
    rows = []
    for tid in sorted({o.track for o in observations}):
        os = [o for o in observations if o.track == tid]
        areas = [(o.box[2] - o.box[0]) * (o.box[3] - o.box[1]) for o in os]
        rows.append(
            {
                "track": tid,
                "frames": len(os),
                "first": min(o.frame for o in os),
                "last": max(o.frame for o in os),
                "mean_score": float(np.mean([o.score for o in os])),
                "mean_area": float(np.mean(areas)),
            }
        )
    return rows


def filter_observations_by_track_length(
    observations: list[FaceObs],
    min_track_frames: int,
) -> list[FaceObs]:
    if min_track_frames <= 1:
        return observations
    counts: dict[int, int] = {}
    for obs in observations:
        counts[obs.track] = counts.get(obs.track, 0) + 1
    keep = {track for track, count in counts.items() if count >= min_track_frames}
    return [obs for obs in observations if obs.track in keep]


def _label_panel(rgb: np.ndarray, label: str) -> np.ndarray:
    panel = np.zeros((286, 256, 3), dtype=np.uint8)
    img = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    img = cv.resize(img, (256, 256), interpolation=cv.INTER_NEAREST)
    panel[30:, :, :] = img
    cv.putText(
        panel,
        label[:34],
        (6, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv.LINE_AA,
    )
    return panel


def make_contact_sheet(
    out_path: Path,
    frames_by_variant: dict[str, list[np.ndarray]],
    observations: list[FaceObs],
    variant_order: list[str],
    max_tracks: int = 4,
) -> None:
    summaries = sorted(
        track_summary(observations),
        key=lambda r: (r["frames"], r["mean_area"]),
        reverse=True,
    )[:max_tracks]
    selected: list[FaceObs] = []
    for summary in summaries:
        os = [o for o in observations if o.track == summary["track"]]
        selected.append(os[len(os) // 2])

    panels: list[np.ndarray] = []
    source_shape = frames_by_variant[variant_order[0]][0].shape[:2]
    for obs in selected:
        for name in variant_order:
            frames = frames_by_variant[name]
            if obs.frame >= len(frames):
                continue
            crop = resize_crop(frames[obs.frame], obs.box, source_shape, side=128)
            panels.append(_label_panel(crop, f"t{obs.track} f{obs.frame} {name}"))

    if not panels:
        return
    cols = len(variant_order)
    rows = math.ceil(len(panels) / cols)
    canvas = np.zeros((rows * 286, cols * 256, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        y = (idx // cols) * 286
        x = (idx % cols) * 256
        canvas[y:y + 286, x:x + 256, :] = panel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_path), cv.cvtColor(canvas, cv.COLOR_RGB2BGR))


def load_variants_json(path: Path, base_dir: Path | None = None) -> dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"variants JSON must contain an object: {path}")
    return {str(k): resolve_path(v, base_dir or path.parent) for k, v in data.items()}


def load_variants(section: dict[str, Any], variants_json: Path | None, config_base: Path | None) -> dict[str, Path]:
    if variants_json is not None:
        return load_variants_json(variants_json)
    inline = section.get("variants")
    if isinstance(inline, dict) and inline:
        return {str(k): resolve_path(v, config_base) for k, v in inline.items()}
    raise SystemExit("provide --variants-json or [face_eval.variants] in local config")


def _normalise_config(args: argparse.Namespace) -> argparse.Namespace:
    config, config_path = load_config(args.config)
    face_eval = config_section(config, "face_eval")
    config_base = config_path.parent if config_path is not None else None

    merged = argparse.Namespace()
    merged.config = config_path
    merged.config_base = config_base
    merged.baseline = str(config_get(face_eval, args, "baseline", "none"))
    merged.model = config_get(face_eval, args, "model", DEFAULT_MODEL)
    model_base = None if args.model is not None else config_base
    merged.model = resolve_path(merged.model, model_base)

    merged.variants_json = config_get(face_eval, args, "variants_json", None)
    if merged.variants_json is not None:
        variants_base = None if args.variants_json is not None else config_base
        merged.variants_json = resolve_path(merged.variants_json, variants_base)
    merged.variants = load_variants(face_eval, merged.variants_json, config_base)

    merged.out_dir = config_get(face_eval, args, "out_dir", None)
    if merged.out_dir is None:
        merged.out_dir = default_shared_temp() / f"vsr_face_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        output_base = None if args.out_dir is not None else config_base
        merged.out_dir = resolve_path(merged.out_dir, output_base)
    merged.max_frames = int(config_get(face_eval, args, "max_frames", 180))
    merged.threshold = float(config_get(face_eval, args, "threshold", 0.30))
    merged.min_side = int(config_get(face_eval, args, "min_side", 32))
    merged.min_track_frames = int(config_get(face_eval, args, "min_track_frames", 1))
    merged.contact_sheet = bool(face_eval.get("contact_sheet", True))
    if args.no_contact_sheet:
        merged.contact_sheet = False
    return merged


def run_eval(args: argparse.Namespace) -> list[dict[str, Any]]:
    _load_numpy()
    variants = args.variants
    if args.baseline not in variants:
        raise SystemExit(f"baseline {args.baseline!r} is not configured")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_by_variant = {
        name: read_frames(path, args.max_frames)
        for name, path in variants.items()
    }
    baseline_frames = frames_by_variant[args.baseline]
    observations = detect_faces(baseline_frames, args.model, args.threshold, args.min_side)
    observations = filter_observations_by_track_length(observations, args.min_track_frames)
    if not observations:
        raise SystemExit("YuNet found no faces in baseline")

    tracks = track_summary(observations)
    rows: list[dict[str, Any]] = []
    for name, frames in frames_by_variant.items():
        row = {"variant": name, "path": str(variants[name])}
        if name == args.baseline:
            row.update({"face_samples": len(observations), "tracks": len(tracks), "face_score": 0.0})
        else:
            row.update(aggregate_variant(baseline_frames, frames, observations))
        rows.append(row)

    metric_keys = sorted({k for row in rows for k in row})
    with (args.out_dir / "face_yunet_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "face_yunet_metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "face_yunet_tracks.json").write_text(json.dumps(tracks, indent=2) + "\n", encoding="utf-8")
    if args.contact_sheet:
        make_contact_sheet(args.out_dir / "face_yunet_contact_sheet.png", frames_by_variant, observations, list(variants))

    print(f"tracks={len(tracks)} samples={len(observations)}")
    for row in sorted(
        [r for r in rows if r["variant"] != args.baseline],
        key=lambda r: r["face_score"],
        reverse=True,
    ):
        print(
            f"{row['variant']:28s} score={row['face_score']:7.2f} "
            f"ssim={row['ssim_luma']:.4f} gradR={row['grad_ratio']:.3f} "
            f"gradC={row['grad_corr']:.3f} hfR={row['hf_luma_ratio']:.3f} "
            f"chrR={row['chroma_hf_ratio']:.3f} rowR={row['row_score_ratio']:.3f} "
            f"tempR={row['temporal_hf_diff_ratio']:.3f} "
            f"mae={row['mae_luma']:.4f} cdrift={row['chroma_drift']:.4f}"
        )
    print(args.out_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="TOML/JSON config; defaults to scripts/vsr_eval/vsr_eval.local.{toml,json} if present")
    parser.add_argument("--variants-json", type=Path, help="JSON object mapping variant names to video paths")
    parser.add_argument("--baseline", help="baseline variant name, usually none or orig")
    parser.add_argument("--out-dir", help="output directory for CSV/JSON/contact sheet")
    parser.add_argument("--model", help="YuNet ONNX model path")
    parser.add_argument("--max-frames", type=int, help="maximum frames to decode per variant")
    parser.add_argument("--threshold", type=float, help="YuNet confidence threshold")
    parser.add_argument("--min-side", type=int, help="ignore detected faces smaller than this side length")
    parser.add_argument("--min-track-frames", type=int, help="ignore tracks shorter than this many detected frames")
    parser.add_argument("--no-contact-sheet", action="store_true", help="skip contact sheet image")
    return parser.parse_args()


def main() -> int:
    run_eval(_normalise_config(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
