"""kinovsr probe noise: analyze a clip's noise instead of processing it.

Per-block quantile/energy/amplitude sigma statistics, flicker density and
amplitude, temporal persistence, edge/luma correlation, a static-grain
proxy, per-channel split, the per-frame flash trace with keyframe
distances, and tool guidance naming the cleanup stage the measurements
support. Three sample windows across the trim range; prints and exits.

Extracted from the inherited harness's --probe-noise block; the flat run flag
delegates here. Results are emitted through this module's logger.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from kinovsr.media.timespec import resolve_trim

_log = logging.getLogger(__name__)


def probe_noise(video: Path, *, start_spec: str | None = None,
                end_spec: str | None = None, reader: str = "auto") -> int:
    """Run the noise probe over `video`, logging the analysis report."""
    import mlx.core as mx

    from kinovsr.analysis.noise import (
        analyze_noise,
        classify_noise_analysis,
        detect_grid_period,
        estimate_blockiness_map,
    )
    from kinovsr.analysis.quant_comb import estimate_qf
    from kinovsr.media import pixel_buffers as _pb
    from kinovsr.media import video_reader as _native_vr

    vr: Any = _native_vr
    if reader == "ffmpeg":
        from kinovsr.media import ffmpeg_reader
        vr = ffmpeg_reader
        _log.info("[reader] ffmpeg compatibility reader (forced)")
    elif reader == "auto":
        try:
            vr.probe_video(video)
        except Exception as e:
            from kinovsr.media import ffmpeg_reader
            vr = ffmpeg_reader
            _log.info(f"[reader] native reader cannot open this file "
                 f"({type(e).__name__}); using the ffmpeg compatibility reader")

    in_w, in_h, source_fps, total_frames, _transform, _par = vr.probe_video(video)
    win_start, win_end = resolve_trim(start_spec, end_spec, source_fps,
                                      total_frames)
    _pw_end = win_end if win_end is not None else total_frames
    _span = max(1, _pw_end - win_start)
    _starts = sorted({win_start + int(f * max(0, _span - 12)) for f in (0.1, 0.5, 0.9)})
    _log.info(f"[probe] noise analysis: {len(_starts)} windows of 12 frames in "
         f"[{win_start}, {_pw_end})")
    try:
        _kfl = vr.keyframe_display_indices(video) or [0]
    except Exception:
        _kfl = [0]
    _all_frames: list = []
    _mid_frames: list = []
    _all_labels: list = []
    _mc_sigs: list = []
    for ws in _starts:
        _fr = []
        for _chk in vr.iter_video_buffer_chunks(
                video, _pb.PIX_RGBAHALF, chunk_size=6,
                start_frame=ws, end_frame=min(ws + 12, _pw_end)):
            _fr += [mx.clip(_pb.read_buffer_rgb_f32(b), 0, 1) for b in _chk]
        if len(_fr) < 3:
            continue
        r = analyze_noise(_fr)
        diag = classify_noise_analysis(r)
        _all_frames.extend(_fr)
        if ws == _starts[len(_starts) // 2]:
            _mid_frames = list(_fr)
        _all_labels.extend(diag["labels"])
        _mc_sigs.append(float(r.get("mc_sigma", 0.0)))
        tr = r.pop("frame_trace")

        def _kfd(i, _ws=ws):
            pri = [k for k in _kfl if k <= _ws + i + 1]
            return _ws + i + 1 - (max(pri) if pri else 0)
        _log.info(f"[probe] window @ frame {ws}:")
        _log.info("  sigma (med/p90 per block): " + "  ".join(
            f"{k} {r[k][0]:.4f}/{r[k][1]:.4f}"
            for k in ("q50", "q75", "q90", "q95", "rms", "tail5", "tail1", "max")))
        _log.info(f"  flicker: density {r['flicker_density'][0]:.3f}/{r['flicker_density'][1]:.3f}"
             f"   amplitude {r['flicker_amplitude'][0]:.4f}/{r['flicker_amplitude'][1]:.4f}"
             f"   (fraction of pixels moving; how hard those move)")
        _log.info(f"  structure: lag2/lag1 {r.get('lag2_over_lag1', 0):.2f}   "
             f"edge/flat {r['edge_over_flat']:.2f}   luma-corr {r['luma_corr']:+.2f}   "
             f"static-frac {r['static_fraction']:.2f}   "
             f"static-spatial-hf {r['static_spatial_hf']:.4f}   "
             f"row-period {r.get('row_periodicity', 0.0):.2f}@"
             f"{r.get('row_period_px', 0.0):.0f}px")
        _log.info(f"  noise floor: mc sigma {r.get('mc_sigma', 0.0):.4f} "
             f"lag {r.get('mc_lag21', 1.0):.2f}   |   "
             f"flat sigma {r.get('flat_sigma', 0.0):.4f} "
             f"lag {r.get('flat_lag21', 1.0):.2f} "
             f"diff-corr {r.get('flat_diff_corr', 0.0):.2f}   "
             f"(mc = the map's default floor: aligned-residual noise on "
             f"all pixels; lag ~1 + sigma >= 0.03 = dense real noise)")
        _log.info(f"  spatial floor: sigma {r.get('spatial_sigma', 0.0):.4f}   "
             f"static grain ~{r.get('static_grain_sigma', 0.0):.4f}   "
             f"static-banding {r.get('static_row_periodicity', 0.0):.2f}@"
             f"{r.get('static_row_period_px', 0.0):.0f}px   "
             f"interlace {r.get('row_interlace', 0.0):.1f}x")
        _log.info(f"  channels: R {r['sigma_R']:.4f}  G {r['sigma_G']:.4f}  B {r['sigma_B']:.4f}")
        _log.info(f"  verdict: {', '.join(diag['labels'])}  risk={diag['risk']}")
        for _msg in diag["warnings"][:2]:
            _log.warning("%s", _msg)
        for _msg in diag["suggestions"][:2]:
            _log.info(f"    try: {_msg}")
        _pk = sorted(range(len(tr)), key=lambda i: -tr[i])[:3]
        _log.info(f"  frame trace: med {sorted(tr)[len(tr) // 2]:.4f}  max {max(tr):.4f}"
             "   top flashes: " + ", ".join(
                 f"diff{i + 1} {tr[i]:.4f} (kf+{_kfd(i)})" for i in _pk))
    # ---- tool guidance: name the tool the measurements support --------
    _log.info("[probe] tool guidance:")
    _comb = (estimate_qf(_all_frames) if _all_frames
             else {"qf": None, "confidence": 0.0})
    if _comb["qf"] is not None:
        _log.info(f"  JPEG ancestry: QF ~{_comb['qf']} "
             f"(confidence {_comb['confidence']:g}) -> --deblock fbcnn "
             f"(auto QF tracks it per tile)")
    else:
        _log.info("  no JPEG-family comb (native H.264/HEVC, or combs killed "
             "by a later re-encode)")
    _bp95 = None
    if _mid_frames:
        _gp = detect_grid_period(_mid_frames)
        _pnon8 = [r[0] for r in (_gp.get("px"), _gp.get("py"))
                  if r is not None and abs(r[0] - 8.0) > 0.3]
        if _pnon8:
            _pm = sum(_pnon8) / len(_pnon8)
            _log.info(f"  grid period ~{_pm:.1f} px (~{_pm / 8.0:.2f}x resize of "
                 f"an 8-grid): footage was compressed then RESIZED; the "
                 f"blockiness map tracks it (period=auto)")
        _blk = estimate_blockiness_map(_mid_frames)
        if _blk is not None:
            _bs = mx.sort(_blk.reshape(-1))
            _bp95 = float(_bs[int(0.95 * (int(_bs.shape[0]) - 1))])
            # clean modern re-encodes read p95 ~0.3 from their own light
            # grid; recommending stdf there costs quality (measured on the
            # corpus controls), so "little" extends past that baseline
            if _bp95 >= 0.6:
                _bmsg = ("strong coding grid -> --deblock stdf "
                         "--deblock-map auto (strength 0.3-0.4)")
            elif _bp95 >= 0.4:
                _bmsg = ("mild coding grid -> --deblock stdf "
                         "--deblock-map auto (strength 0.15-0.25)")
            else:
                _bmsg = ("little grid evidence (clean-encode baseline): "
                         "nothing grid-locked to deblock")
            _log.info(f"  blockiness p95 {_bp95:.2f}: {_bmsg}")
    _mc_med = sorted(_mc_sigs)[len(_mc_sigs) // 2] if _mc_sigs else 0.0
    if "dense sensor noise" in _all_labels:
        _log.info(f"  dense noise (mc floor ~{_mc_med:.3f}) -> --denoise bsvd "
             f"--noise-map auto")
    elif "sparse edge flicker" in _all_labels:
        _log.info("  sparse edge flicker -> compression cleanup first "
             "(deblock / fbcnn); plain denoise adds little")
    elif "dense motion, low noise floor" in _all_labels:
        _log.info(f"  clean motion (mc floor ~{_mc_med:.3f}) -> skip or keep "
             f"denoise minimal; heavy denoise eats texture")
    elif "static/structured grain" in _all_labels:
        _log.info("  static grain: temporal denoisers cannot remove it -> "
             "spatial cleanup or a small --noise-map-floor")
    if "interlace/field residue" in _all_labels:
        _log.info("  interlace residue -> deinterlace upstream before any "
             "denoise/deblock (temporal nets smear combing)")
    if "static row banding" in _all_labels:
        _log.info("  static row banding -> spatial-only artifact; temporal "
             "tools will not touch it")
    _bpp = None
    try:
        _sizes = vr.coded_frame_sizes(video)
    except Exception:
        _sizes = []
    if _sizes:
        _seg = [s for s in _sizes[win_start:_pw_end] if s > 0]
        if _seg:
            _bpp = 8.0 * (sum(_seg) / len(_seg)) / float(in_w * in_h)
            _bnote = ("starved encode, damage certain" if _bpp < 0.08
                      else "lean encode" if _bpp < 0.2 else "generous encode")
            _log.info(f"  bpp {_bpp:.3f} over the probed range: {_bnote} "
                 f"(last generation ONLY: high bpp proves nothing after "
                 f"a re-encode)")
    if (_comb["qf"] is None and _bp95 is not None and _bp95 < 0.4
            and _bpp is not None and _bpp < 0.12):
        _log.info("  low bpp with no measurable grid or comb (mush): auto "
             "dials have no anchor here -> manual call (bsvd for "
             "shimmer, nafnet for structure)")
    _log.info("[probe] done -- no processing performed")
    return 0


def run_probe_noise(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="kinovsr probe noise",
        description="Analyze a clip's noise instead of processing it.")
    parser.add_argument("--video", required=True, type=Path,
                        help="Video file (mp4/mov/...).")
    parser.add_argument("--start", default=None,
                        help="Probe-range start (frames, Ns, or mm:ss).")
    parser.add_argument("--end", default=None,
                        help="Probe-range end, exclusive (same forms).")
    parser.add_argument("--reader", default="auto",
                        choices=("auto", "native", "ffmpeg"),
                        help="Video reader backend (matches the run flag).")
    args = parser.parse_args(argv)
    return probe_noise(args.video, start_spec=args.start, end_spec=args.end,
                       reader=args.reader)
