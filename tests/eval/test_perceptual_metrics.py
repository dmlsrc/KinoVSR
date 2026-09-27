"""kinovsr metrics perceptual: the source row and variant names."""

import json

import av
import pytest

from kinovsr.eval.perceptual_metrics import run_perceptual

pytestmark = pytest.mark.unit

W, H = 64, 48


def _clip(path, shade: int) -> None:
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=25)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    for _ in range(3):
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(bytes([shade]) * (W * H))
        for pkt in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()


def _manifest(tmp_path, entries):
    path = tmp_path / "variants.json"
    path.write_text(json.dumps({name: str(video) for name, video in entries.items()}))
    return path


def test_a_variant_named_source_is_scored_without_a_source_clip(tmp_path):
    # It used to raise UnboundLocalError: the loop read the --source frames
    # for any row named "source".
    original, variant = tmp_path / "original.mp4", tmp_path / "variant.mp4"
    _clip(original, 90)
    _clip(variant, 120)
    manifest = _manifest(tmp_path, {"source": original, "denoised": variant})
    out = tmp_path / "report"

    argv = ["--variants-json", str(manifest), "--out-dir", str(out), "--metrics", "flicker"]
    assert run_perceptual(argv) == 0
    rows = json.loads((out / "perceptual_metrics.json").read_text())
    assert [(row["variant"], row["video"]) for row in rows] == [
        ("source", str(original)),
        ("denoised", str(variant)),
    ]


def test_a_variant_named_source_is_refused_beside_a_source_clip(tmp_path):
    # The manifest's path used to label a row whose frames and VMAF shortcut
    # came from --source.
    original, other = tmp_path / "original.mp4", tmp_path / "other.mp4"
    _clip(original, 90)
    _clip(other, 120)
    manifest = _manifest(tmp_path, {"source": other})
    argv = [
        "--variants-json",
        str(manifest),
        "--source",
        str(original),
        "--out-dir",
        str(tmp_path / "report"),
        "--metrics",
        "flicker",
    ]
    with pytest.raises(SystemExit, match="reserved for --source"):
        run_perceptual(argv)
