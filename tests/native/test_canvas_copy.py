"""CanvasCopier pads and crops CVPixelBuffers exactly, plane by plane."""

import pytest

pytestmark = pytest.mark.integration

FORMATS = ("PIX_BGRA", "PIX_RGBAHALF", "PIX_NV12")
BYTES_PER_PIXEL = {"PIX_BGRA": (4,), "PIX_RGBAHALF": (8,), "PIX_NV12": (1, 2)}


def _buffer(format_name: str, width: int, height: int):
    from kinovsr.media import pixel_buffers as pb

    pixel_format = getattr(pb, format_name)
    return pb.make_pixel_buffer_from_attrs(
        width,
        height,
        {
            "PixelFormatType": pixel_format,
            "Width": width,
            "Height": height,
            "IOSurfaceProperties": {},
            "MetalCompatibility": True,
        },
    )


def _plane_count(buffer) -> int:
    from kinovsr.native.frameworks import Quartz

    if Quartz.CVPixelBufferIsPlanar(buffer):
        return int(Quartz.CVPixelBufferGetPlaneCount(buffer))
    return 1


def _plane(buffer, index: int):
    """(base address, bytes per row, width, height) of a locked buffer's plane."""
    from kinovsr.native.frameworks import Quartz

    if Quartz.CVPixelBufferIsPlanar(buffer):
        return (
            Quartz.CVPixelBufferGetBaseAddressOfPlane(buffer, index),
            int(Quartz.CVPixelBufferGetBytesPerRowOfPlane(buffer, index)),
            int(Quartz.CVPixelBufferGetWidthOfPlane(buffer, index)),
            int(Quartz.CVPixelBufferGetHeightOfPlane(buffer, index)),
        )
    return (
        Quartz.CVPixelBufferGetBaseAddress(buffer),
        int(Quartz.CVPixelBufferGetBytesPerRow(buffer)),
        int(Quartz.CVPixelBufferGetWidth(buffer)),
        int(Quartz.CVPixelBufferGetHeight(buffer)),
    )


def _fill(buffer, format_name: str, seed: int) -> None:
    """Deterministic content: random bytes, or finite halves in [-0.25, 1.25)."""
    import mlx.core as mx

    from kinovsr.native.frameworks import Quartz

    Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
    try:
        for index in range(_plane_count(buffer)):
            base, stride, _, height = _plane(buffer, index)
            key = mx.random.key(seed + index)
            if format_name == "PIX_RGBAHALF":
                values = mx.random.uniform(-0.25, 1.25, (stride * height // 2,), key=key)
                data = values.astype(mx.float16).view(mx.uint8)
            else:
                data = mx.random.randint(0, 256, (stride * height,), key=key).astype(mx.uint8)
            base.as_buffer(stride * height)[:] = bytes(memoryview(data))
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)


def _rows(buffer, format_name: str) -> list[list[bytes]]:
    """Each plane's active bytes, one entry per row."""
    from kinovsr.native.frameworks import Quartz

    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        planes = []
        for index, bytes_per_pixel in enumerate(BYTES_PER_PIXEL[format_name]):
            base, stride, width, height = _plane(buffer, index)
            view = base.as_buffer(stride * height)
            planes.append(
                [
                    bytes(view[y * stride : y * stride + width * bytes_per_pixel])
                    for y in range(height)
                ]
            )
        return planes
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)


@pytest.fixture
def copier():
    from kinovsr.native.canvas_copy import CanvasCopier

    instance = CanvasCopier()
    yield instance
    instance.close()


@pytest.mark.parametrize(
    ("format_name", "size", "canvas"),
    [
        *((name, (720, 480), (960, 540)) for name in FORMATS),
        ("PIX_BGRA", (719, 479), (960, 540)),
        ("PIX_RGBAHALF", (353, 287), (640, 360)),
    ],
)
def test_padding_repeats_the_last_column_and_row_and_crops_back_exactly(
    copier, format_name, size, canvas
):
    source = _buffer(format_name, *size)
    _fill(source, format_name, seed=7)

    padded = copier.copy(source, _buffer(format_name, *canvas))
    cropped = copier.copy(padded, _buffer(format_name, *size))

    source_rows = _rows(source, format_name)
    for plane, (rows, padded_rows, bytes_per_pixel) in enumerate(
        zip(source_rows, _rows(padded, format_name), BYTES_PER_PIXEL[format_name], strict=True)
    ):
        width = len(rows[0]) // bytes_per_pixel
        padded_width = len(padded_rows[0]) // bytes_per_pixel
        for y, padded_row in enumerate(padded_rows):
            row = rows[min(y, len(rows) - 1)]
            expected = row + row[-bytes_per_pixel:] * (padded_width - width)
            assert padded_row == expected, f"plane {plane} row {y}"
    assert _rows(cropped, format_name) == source_rows


def test_crop_keeps_the_top_left(copier):
    source = _buffer("PIX_BGRA", 960, 540)
    _fill(source, "PIX_BGRA", seed=11)

    cropped = copier.copy(source, _buffer("PIX_BGRA", 720, 480))

    assert _rows(cropped, "PIX_BGRA")[0] == [
        row[: 720 * 4] for row in _rows(source, "PIX_BGRA")[0][:480]
    ]


def test_copy_carries_the_source_attachments(copier):
    from kinovsr.native.frameworks import Quartz

    source = _buffer("PIX_NV12", 720, 480)
    _fill(source, "PIX_NV12", seed=3)
    Quartz.CVBufferSetAttachment(
        source,
        Quartz.kCVImageBufferYCbCrMatrixKey,
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4,
        Quartz.kCVAttachmentMode_ShouldPropagate,
    )

    padded = copier.copy(source, _buffer("PIX_NV12", 960, 540))

    matrix = Quartz.CVBufferGetAttachment(padded, Quartz.kCVImageBufferYCbCrMatrixKey, None)
    if isinstance(matrix, tuple):
        matrix = matrix[0]
    assert matrix == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4


def test_mismatched_and_unsupported_formats_are_refused(copier):
    from kinovsr.media import pixel_buffers as pb

    with pytest.raises(ValueError, match="one pixel format"):
        copier.copy(_buffer("PIX_BGRA", 64, 48), _buffer("PIX_RGBAHALF", 64, 48))

    two_component_half = int.from_bytes(b"2C0h", "big")
    attrs = {
        "PixelFormatType": two_component_half,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }
    flow = pb.make_pixel_buffer_from_attrs(64, 48, attrs)
    with pytest.raises(RuntimeError, match="does not support"):
        copier.copy(flow, pb.make_pixel_buffer_from_attrs(96, 64, attrs))
