#!/usr/bin/env python3
"""Verify that every collected source converts to the shipped artifact.

For each weights-src/ source with a shipped .safetensors in the repo,
run the documented conversion into a scratch dir and compare the result
against the shipped file: identical key sets, dtypes, shapes, and
bit-exact tensor values (plus a byte-identity note). Special-cased
owners use their dedicated converters (toflow .t7, dover, musiq);
realplksr and the YuNet ONNX are redistributed unmodified, so they
byte-compare directly.

Exit code is nonzero if any comparison fails. Sources that are absent
from weights-src/ or have no shipped artifact are reported as SKIP.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_log = logging.getLogger("kinovsr.dev.verify_source_conversions")

SRC = REPO / "weights-src"
OUT = Path(os.environ.get("SHARED_TEMP_DIR")
           or os.environ.get("TMPDIR") or "/tmp") / "weights_src_verify"

# (owner, asset_id, source relpath under weights-src/, mode, extra)
#   convert: kinovsr weights convert with the documented flags
#   toflow:  convert_t7_to_safetensors.py (safetensors + json graph)
#   script:  dedicated converter script taking (src, dst)
#   bytes:   redistributed unmodified; compare files directly
#   skip:    reason string in extra
BVPP_FLAGS = ["--strip-prefix", "generator."]
RECIPES = [
    ("basicvsrpp", "reds4", "basicvsrpp/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "vimeo90k_bi", "basicvsrpp/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "vimeo90k_bd", "basicvsrpp/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "ntire_vsr", "basicvsrpp/basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "decompress_track1", "basicvsrpp/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "decompress_track2", "basicvsrpp/basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "decompress_track3", "basicvsrpp/basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "denoise", "basicvsrpp/basicvsr_plusplus_denoise-28f6920c.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "deblur_dvd", "basicvsrpp/basicvsr_plusplus_deblur_dvd-ecd08b7f.pth", "convert", BVPP_FLAGS),
    ("basicvsrpp", "deblur_gopro", "basicvsrpp/basicvsr_plusplus_deblur_gopro-3c5bb9b5.pth", "convert", BVPP_FLAGS),
    ("bsvd", "c64", "bsvd/bsvd-64.pth", "convert", ["--param-key", "params"]),
    ("bsvd", "c32", "bsvd/bsvd-32.pth", "convert", ["--param-key", "params"]),
    ("dover", "dover_mobile", "dover/DOVER-Mobile.pth", "script",
     "kinovsr/eval/models/dover/convert_dover.py"),
    ("esc", "gan", "esc/ESC_Real_X4_GAN.pth", "convert", ["--param-key", "params_ema"]),
    ("esc", "mse", "esc/ESC_Real_X4_MSE.pth", "convert", ["--param-key", "params_ema"]),
    ("eval", "face_yunet", "eval/face_detection_yunet_2023mar.onnx", "bytes", None),
    ("fastdvdnet", "standard", "fastdvdnet/model.pth", "convert", []),
    ("fastdvdnet", "clipped", "fastdvdnet/model_clipped_noise.pth", "convert", []),
    ("fbcnn", "color", "fbcnn/fbcnn_color.pth", "convert", ["--strip-prefix", ""]),
    ("fbcnn", "gray", "fbcnn/fbcnn_gray.pth", "convert", ["--strip-prefix", ""]),
    ("fbcnn", "gray_double", "fbcnn/fbcnn_gray_double.pth", "convert", ["--strip-prefix", ""]),
    ("musiq", "musiq_koniq", "musiq/musiq_koniq_ckpt-e95806b9.pth", "script",
     "kinovsr/eval/models/musiq/convert_musiq.py"),
    ("nafnet", "gopro", "nafnet/NAFNet-GoPro-width64.pth", "convert", ["--strip-prefix", ""]),
    ("nafnet", "gopro32", "nafnet/NAFNet-GoPro-width32.pth", "convert", ["--strip-prefix", ""]),
    ("nafnet", "sidd", "nafnet/NAFNet-SIDD-width64.pth", "convert", ["--strip-prefix", ""]),
    ("nafnet", "sidd32", "nafnet/NAFNet-SIDD-width32.pth", "convert", ["--strip-prefix", ""]),
    ("nafnet", "reds", "nafnet/NAFNet-REDS-width64.pth", "convert", ["--strip-prefix", ""]),
    ("pvdd", "pvdd", "pvdd/pvdd_srgb_nolevel.pth", "convert", []),
    ("pvdd", "crvd", "pvdd/crvd_srgb_nolevel.pth", "convert", []),
    ("pvdd", "davis", "pvdd/davis_srgb_nolevel.pth", "convert", []),
    ("pvdd", "pvdd_level", "pvdd/pvdd_srgb_level.pth", "convert", []),
    ("pvdd", "pvdd_raw", "pvdd/pvdd_raw_nolevel.pth", "convert", []),
    ("pvdd", "pvdd_raw_level", "pvdd/pvdd_raw_level.pth", "convert", []),
    ("realbasicvsr", "x4", "realbasicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth", "convert",
     ["--only-prefix", "generator_ema.", "--strip-prefix", "generator_ema."]),
    ("realesrgan", "general", "realesrgan/realesr-general-x4v3.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "general-wdn", "realesrgan/realesr-general-wdn-x4v3.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "x4plus", "realesrgan/RealESRGAN_x4plus.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "realesrnet", "realesrgan/RealESRNet_x4plus.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "bsrgan", "realesrgan/BSRGAN.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "bsrnet", "realesrgan/BSRNet.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "x2plus", "realesrgan/RealESRGAN_x2plus.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "anime", "realesrgan/RealESRGAN_x4plus_anime_6B.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "animevideo", "realesrgan/realesr-animevideov3.pth", "convert", ["--strip-prefix", ""]),
    ("realesrgan", "esrgan", "realesrgan/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth", "convert", ["--strip-prefix", ""]),
    ("realplksr", "public2x", "realplksr/2xPublic_realplksr_dysample_layernorm_real.safetensors", "bytes", None),
    ("realplksr", "public2x-nn", "realplksr/2xPublic_realplksr_dysample_layernorm_real_nn.safetensors", "bytes", None),
    ("realplksr", "nomos4x", "realplksr/4xNomosWebPhoto_RealPLKSR.safetensors", "bytes", None),
    ("realviformer", "x4", "realviformer/weights.pth", "convert", ["--param-key", "params"]),
    ("safmn", "light", "safmn/light_safmnpp.pth", "convert", ["--param-key", "params"]),
    ("safmn", "real", "safmn/SAFMN_L_Real_LSDIR_x4.pth", "convert", ["--param-key", "params"]),
    ("safmn", "real2x", "safmn/SAFMN_L_Real_LSDIR_x2.pth", "convert", ["--param-key", "params"]),
    ("safmn", "purescale", "safmn/4x_SAFMN_PureScale.pth", "convert", ["--param-key", "params_ema"]),
    ("safmn", "purescale2x", "safmn/2x_SAFMN_PureScale.pth", "convert", ["--param-key", "params_ema"]),
    ("safmn", "purescale2x-sharp", "safmn/2x_SAFMN_PureScale_sharper.pth", "convert", ["--param-key", "params_ema"]),
    ("spynet", "stock", "spynet/spynet_20210409-c6c1bd09.pth", "script",
     "kinovsr/modeling/spynet/convert_spynet.py"),
    ("stdf", "mfqev2", "stdf/exp/MFQEv2_R3_enlarge300x/ckp_290000.pt", "convert", []),
    ("stdf", "vimeo90k", "stdf/exp/Vimeo90K_R3_enlarge300x/ckp_300000.pt", "convert", []),
    ("toflow", "denoise", "toflow/denoise.t7", "toflow", None),
    ("toflow", "deblock", "toflow/deblock.t7", "toflow", None),
    ("toflow", "sr", "toflow/sr.t7", "toflow", None),
    ("toflow", "interp", "toflow/interp.t7", "toflow", None),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_safetensors(a: Path, b: Path) -> tuple[bool, str]:
    """Same tensors bit-for-bit? Returns (ok, detail)."""
    import mlx.core as mx

    ta, tb = mx.load(str(a)), mx.load(str(b))
    if set(ta) != set(tb):
        extra = sorted(set(ta) - set(tb))[:3]
        missing = sorted(set(tb) - set(ta))[:3]
        return False, f"key sets differ (extra={extra} missing={missing})"
    for key in sorted(ta):
        x, y = ta[key], tb[key]
        if x.dtype != y.dtype:
            return False, f"{key}: dtype {x.dtype} != {y.dtype}"
        if x.shape != y.shape:
            return False, f"{key}: shape {x.shape} != {y.shape}"
        if not mx.array_equal(x, y):
            return False, f"{key}: values differ"
    note = ("bytes-identical" if sha256(a) == sha256(b)
            else "tensors identical (serialization differs)")
    return True, f"{len(ta)} tensors, {note}"


def main() -> int:
    from kinovsr.modeling.weights import load_registered
    from kinovsr.ui.logging import configure_logging

    configure_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, str]] = []   # (verdict, name, detail)

    for owner, asset_id, rel, mode, extra in RECIPES:
        name = f"{owner}/{asset_id}"
        source = SRC / rel
        artifact = load_registered(owner).weights[asset_id].path
        if mode == "skip":
            results.append(("SKIP", name, str(extra)))
            continue
        if not source.is_file():
            results.append(("SKIP", name, f"source not collected: {rel}"))
            continue
        if not artifact.is_file():
            results.append(("SKIP", name, "no shipped artifact to compare"))
            continue

        if mode == "bytes":
            ok = sha256(source) == sha256(artifact)
            results.append(("PASS" if ok else "FAIL", name,
                            "byte-identical" if ok
                            else "source differs from shipped file"))
            continue

        conv = OUT / owner / artifact.name
        conv.parent.mkdir(parents=True, exist_ok=True)
        if mode == "convert":
            from kinovsr.cli.commands.weights_convert import run_convert

            rc = run_convert([str(source), "-o", str(conv), *list(extra)])
            if rc:
                results.append(("FAIL", name, f"converter exited {rc}"))
                continue
        elif mode == "toflow":
            graph = conv.with_suffix(".json")
            proc = subprocess.run(
                [sys.executable,
                 str(REPO / "kinovsr/processors/toflow/convert_t7_to_safetensors.py"),
                 str(source), "-o", str(conv), "--graph", str(graph)],
                capture_output=True, text=True)
            if proc.returncode:
                results.append(("FAIL", name,
                                f"t7 converter: {proc.stderr.strip()[-200:]}"))
                continue
            shipped_graph = artifact.with_suffix(".json")
            if graph.read_bytes() != shipped_graph.read_bytes():
                results.append(("FAIL", name, "graph JSON differs"))
                continue
        elif mode == "script":
            proc = subprocess.run(
                [sys.executable, str(REPO / extra), str(source), str(conv)],
                capture_output=True, text=True)
            if proc.returncode:
                results.append(("FAIL", name,
                                f"converter: {proc.stderr.strip()[-200:]}"))
                continue

        ok, detail = compare_safetensors(conv, artifact)
        pinned = load_registered(owner).weights[asset_id].artifact_sha256
        if ok and pinned and "serialization differs" in detail:
            ok = False
            detail = ("tensors identical but bytes drifted from the "
                      "hash-pinned artifact - regenerate it and re-pin "
                      "the manifest")
        results.append(("PASS" if ok else "FAIL", name, detail))

    width = max(len(name) for _, name, _ in results)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for verdict, name, detail in results:
        counts[verdict] += 1
        log = {
            "PASS": _log.info,
            "SKIP": _log.warning,
            "FAIL": _log.error,
        }[verdict]
        log("%4s  %-*s  %s", verdict, width, name, detail)
    summary_log = _log.error if counts["FAIL"] else _log.info
    summary_log(
        "%s pass, %s fail, %s skip of %s",
        counts["PASS"],
        counts["FAIL"],
        counts["SKIP"],
        len(results),
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
