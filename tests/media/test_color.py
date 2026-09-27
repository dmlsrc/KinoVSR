"""Media color-resolution tests."""


def test_source_range_resolve_override():
    from kinovsr.media import color

    src = {
        "primaries": None,
        "transfer": None,
        "matrix": None,
        "full_range": False,
        "tagged": False,
    }
    # auto trusts the container flag
    assert color.resolve(src, "auto", "auto")[3] is False
    assert color.resolve(dict(src, full_range=True), "auto", "auto")[3] is True
    # forcing overrides the flag in both directions
    assert color.resolve(src, "auto", "full")[3] is True
    assert color.resolve(dict(src, full_range=True), "auto", "video")[3] is False
    # range override composes with a colorimetry override
    resolved = color.resolve(src, "bt601", "full")
    assert resolved[3] is True
    assert "range=full" in color.describe(resolved)


def test_frame_spec_resolution_preserves_independent_color_fields():
    from kinovsr.media import color
    from kinovsr.native.frameworks import Quartz, av
    from kinovsr.processors import (
        ColorMatrix,
        ColorPrimaries,
        ColorRange,
        Domain,
        DType,
        FrameSpec,
        Geometry,
        Layout,
        TransferFunction,
    )

    frame = FrameSpec(
        layout=Layout.MLX_RGB_HWC,
        dtype=DType.FLOAT32,
        color_range=ColorRange.FULL,
        color_matrix=ColorMatrix.BT709,
        color_primaries=ColorPrimaries.BT2020,
        transfer_function=TransferFunction.BT2020,
        domain=Domain.UNIT,
        geometry=Geometry(8, 6),
    )
    resolved = color.resolve_frame_spec(frame)

    assert resolved == (
        Quartz.kCVImageBufferColorPrimaries_ITU_R_2020,
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        True,
    )
    properties = color.av_color_properties(resolved)
    assert properties[av.AVVideoColorPrimariesKey] == (
        Quartz.kCVImageBufferColorPrimaries_ITU_R_2020
    )
    assert properties[av.AVVideoTransferFunctionKey] == (
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2
    )
    assert properties[av.AVVideoYCbCrMatrixKey] == (Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2)


def test_av_constant_lists_cover_the_framework_metadata():
    """color.py hand-lists the AV writer constants so importing FileSink does
    not pay dir(AVFoundation)'s full lazy-framework materialization (~3.5 s,
    50k+ symbols). The names remain enumerable for free in the framework's
    metadata module, so an OS release that adds a primaries/transfer/matrix
    constant fails here instead of silently falling back to BT.709."""
    import re

    import AVFoundation._metadata as metadata

    from kinovsr.media import color
    from kinovsr.native.frameworks import av

    names = re.findall(r"\$([A-Za-z0-9_]+)", getattr(metadata, "constants", ""))
    for prefix, listed in (
        ("AVVideoColorPrimaries_", color._AV_PRIMS),
        ("AVVideoTransferFunction_", color._AV_TRANS),
        ("AVVideoYCbCrMatrix_", color._AV_MATS),
    ):
        declared = {
            value
            for name in names
            if name.startswith(prefix) and (value := getattr(av, name, None)) is not None
        }
        assert declared == set(listed), (
            f"{prefix} constants drifted from kinovsr.media.color's list"
        )


def test_a_track_without_format_descriptions_probes_as_untagged(monkeypatch, tmp_path):
    # The native probe indexed formatDescriptions()[0] and raised IndexError,
    # unlike the reader's other format-description helpers.
    from types import SimpleNamespace

    from kinovsr.media import video_reader

    def no_frames(*_args, **_kwargs):
        raise RuntimeError("no decodable frames")

    track = SimpleNamespace(formatDescriptions=list)
    monkeypatch.setattr(video_reader, "_first_video_track", lambda _asset: track)
    monkeypatch.setattr(video_reader, "iter_video_buffer_chunks", no_frames)

    src = video_reader.probe_color(tmp_path / "stub.mp4")
    assert src["tagged"] is False
    assert (src["primaries"], src["transfer"], src["matrix"]) == (None, None, None)


def _src(**tags):
    base = {"primaries": None, "transfer": None, "matrix": None, "full_range": False}
    return {**base, "tagged": True, **tags}


def test_numbered_coremedia_tokens_resolve_to_their_named_twins():
    # CoreMedia reports BT.470BG as "YCbCrMatrix#5". It used to pass through,
    # so the writer converted pixels with BT.601 but tagged BT.709, a visible
    # color shift on PAL encodes.
    import Quartz

    from kinovsr.media import color

    resolved = color.resolve(_src(matrix="YCbCrMatrix#5", primaries="ColorPrimaries#7"))
    assert resolved[2] == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4
    assert resolved[0] == Quartz.kCVImageBufferColorPrimaries_SMPTE_C
    assert (
        color.resolve(_src(matrix="YCbCrMatrix#10"))[2]
        == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020
    )


def test_matrices_the_writer_cannot_tag_fall_back_to_bt709_for_pixels_too():
    import Quartz

    from kinovsr.media import color

    for matrix in ("YCbCrMatrix#8", "ITU_R_2100_ICtCp"):
        resolved = color.resolve(_src(matrix=matrix, transfer="TransferFunction#4"))
        assert resolved[2] == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2
        assert resolved[1] == Quartz.kCVImageBufferTransferFunction_ITU_R_709_2
        assert color.av_color_properties(resolved)[color.av.AVVideoYCbCrMatrixKey] == resolved[2]


def test_smpte_240m_pixels_use_its_own_coefficients():
    # A 240M tag was converted with BT.601's coefficients, another shift.
    import Quartz

    from kinovsr.media import yuv

    assert yuv.coef_for_matrix(Quartz.kCVImageBufferYCbCrMatrix_SMPTE_240M_1995) == (0.212, 0.087)
