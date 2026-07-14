"""Media color-resolution tests."""


def test_source_range_resolve_override():
    from kinovsr.media import color

    src = {"primaries": None, "transfer": None, "matrix": None,
           "full_range": False, "tagged": False}
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
        Quartz.kCVImageBufferColorPrimaries_ITU_R_2020)
    assert properties[av.AVVideoTransferFunctionKey] == (
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2)
    assert properties[av.AVVideoYCbCrMatrixKey] == (
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2)
