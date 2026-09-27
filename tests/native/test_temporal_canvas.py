"""FRC frame sizes whose network renders motion wrong run edge-padded."""

import hashlib
import logging

import pytest

# ---------------------------------------------------------------------------
# Canvas choice and the per-size decision (no VideoToolbox session)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_candidates_are_the_smallest_containing_canvases_in_the_frame_orientation():
    from kinovsr.native.temporal import canvas_candidates

    assert canvas_candidates(720, 480)[0] == (960, 540)
    assert canvas_candidates(352, 288)[:2] == [(640, 360), (360, 640)]
    assert canvas_candidates(540, 720)[0] == (540, 960)
    assert canvas_candidates(640, 272)[0] == (640, 360)
    for width, height in ((720, 480), (640, 480), (480, 720), (1440, 1080)):
        candidates = canvas_candidates(width, height)
        assert (width, height) not in candidates
        assert all(w >= width and h >= height for w, h in candidates)
    assert canvas_candidates(1921, 1080) == []


@pytest.fixture
def scripted(monkeypatch):
    """VtfrcSession with native start, stop, and motion check replaced by a script."""
    from kinovsr.native import temporal

    monkeypatch.setattr(temporal, "_CANVAS_DECISIONS", {})
    events = []
    margins = {}
    unavailable = set()

    def start(self, canvas):
        self.canvas = canvas
        if canvas in unavailable:
            events.append(("unavailable", canvas))
            raise RuntimeError("startSession failed")
        events.append(("start", canvas))
        self.src_attrs = self.dst_attrs = {"PixelFormatType": 1}

    def stop(self):
        events.append(("stop", self.canvas))

    def check(self):
        events.append(("check", self.canvas))
        return margins[self.canvas]

    monkeypatch.setattr(temporal.VtfrcSession, "_start", start)
    monkeypatch.setattr(temporal.VtfrcSession, "_stop", stop)
    monkeypatch.setattr(temporal.VtfrcSession, "_motion_check", check)
    return temporal, events, margins, unavailable


@pytest.mark.unit
def test_a_size_that_passes_is_checked_once_and_kept_direct(scripted):
    temporal, events, margins, _ = scripted
    margins[None] = 20.0

    assert temporal.VtfrcSession(720, 480, 24, 60).canvas is None
    assert events == [("start", None), ("check", None)]

    events.clear()
    assert temporal.VtfrcSession(720, 480, 25, 50).canvas is None
    assert events == [("start", None)]


@pytest.mark.unit
def test_a_failing_size_keeps_the_first_canvas_that_passes(scripted, caplog):
    temporal, events, margins, unavailable = scripted
    first, second, third = temporal.canvas_candidates(720, 480)[:3]
    margins.update({None: 1.3, first: 2.0, third: 21.0})
    unavailable.add(second)

    with caplog.at_level(logging.INFO, logger="kinovsr.native.temporal"):
        session = temporal.VtfrcSession(720, 480, 24, 60)

    assert session.canvas == third
    assert events == [
        ("start", None),
        ("check", None),
        ("stop", None),
        ("start", first),
        ("check", first),
        ("stop", first),
        ("unavailable", second),
        ("stop", second),
        ("start", third),
        ("check", third),
    ]
    assert "renders motion wrong at 720x480 (1.3 dB" in caplog.text
    assert "into {}x{} (21.0 dB)".format(*third) in caplog.text

    events.clear()
    assert temporal.VtfrcSession(720, 480, 24, 60).canvas == third
    assert events == [("start", third)]


@pytest.mark.unit
def test_when_nothing_passes_every_session_checks_again_and_warns(scripted, caplog):
    temporal, events, margins, unavailable = scripted
    candidates = temporal.canvas_candidates(720, 480)
    margins[None] = 1.0
    margins.update(dict.fromkeys(candidates[1:], 2.0))
    unavailable.add(candidates[0])

    for _ in range(2):
        events.clear()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="kinovsr.native.temporal"):
            session = temporal.VtfrcSession(720, 480, 24, 60)

        assert session.canvas is None
        assert events[:3] == [("start", None), ("check", None), ("stop", None)]
        assert ("unavailable", candidates[0]) in events
        assert events[-2:] == [("stop", candidates[-1]), ("start", None)]
        assert "moving edges may look doubled" in caplog.text
    assert temporal._CANVAS_DECISIONS == {}


@pytest.mark.unit
def test_the_decision_is_per_mode_and_tiny_frames_skip_the_check(scripted):
    temporal, events, margins, _ = scripted
    margins[None] = 20.0

    temporal.VtfrcSession(720, 480, 24, 60, mode="normal")
    temporal.VtfrcSession(720, 480, 24, 60, mode="high")
    assert events.count(("check", None)) == 2

    events.clear()
    temporal.VtfrcSession(24, 24, 24, 60)
    assert events == [("start", None)]


@pytest.mark.unit
def test_an_explicit_canvas_must_hold_the_frame():
    from kinovsr.native import temporal

    with pytest.raises(ValueError, match="cannot hold 720x480"):
        temporal.VtfrcSession(720, 480, 24, 60, canvas=(640, 480))


# ---------------------------------------------------------------------------
# VideoToolbox sessions
# ---------------------------------------------------------------------------


def _content(width: int, height: int, index: int):
    """Smooth color texture moving 3 px right and 2 px down per frame."""
    import mlx.core as mx

    y, x = mx.meshgrid(
        mx.arange(height, dtype=mx.float32), mx.arange(width, dtype=mx.float32), indexing="ij"
    )
    x, y = x - 3 * index, y - 2 * index
    return mx.stack(
        [
            0.5 + 0.3 * mx.sin(x * 0.11) * mx.cos(y * 0.07),
            0.5 + 0.3 * mx.sin((x - y) * 0.05),
            0.45 + 0.25 * mx.cos(x * 0.03 + y * 0.09),
        ],
        axis=-1,
    )


def _source(width: int, height: int, index: int, format_name: str = "PIX_RGBAHALF"):
    import mlx.core as mx

    from kinovsr.media import pixel_buffers as pb

    buffer = pb.make_pixel_buffer_from_attrs(
        width,
        height,
        {
            "PixelFormatType": getattr(pb, format_name),
            "Width": width,
            "Height": height,
            "IOSurfaceProperties": {},
            "MetalCompatibility": True,
        },
    )
    rgb = _content(width, height, index)
    if format_name == "PIX_RGBAHALF":
        rgba = mx.concatenate([rgb, mx.ones((height, width, 1))], axis=-1)
        pb.write_fp16_rgba(rgba.astype(mx.float16), buffer)
    else:
        pb.upload_frame_to_buffer(mx.round(mx.clip(rgb, 0, 1) * 255).astype(mx.uint8), buffer)
    return buffer


def _digest(buffer) -> str:
    from kinovsr.native.frameworks import Quartz

    width = int(Quartz.CVPixelBufferGetWidth(buffer))
    height = int(Quartz.CVPixelBufferGetHeight(buffer))
    stride = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(stride * height)
        active = b"".join(bytes(view[y * stride : y * stride + width * 8]) for y in range(height))
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return hashlib.sha256(active).hexdigest()


def _interpolate(session, count: int, format_name: str = "PIX_RGBAHALF") -> list:
    """Feed ``count`` moving frames and drain; returns (width, height, format, digest)."""
    from kinovsr.native.frameworks import Quartz

    produced = []

    def record(outputs):
        for output in outputs:
            produced.append(
                (
                    int(Quartz.CVPixelBufferGetWidth(output)),
                    int(Quartz.CVPixelBufferGetHeight(output)),
                    int(Quartz.CVPixelBufferGetPixelFormatType(output)),
                    _digest(output),
                )
            )
            del output

    for index in range(count):
        record(session.feed(_source(session.in_w, session.in_h, index, format_name), index))
    record(session.drain())
    return produced


@pytest.mark.integration
def test_ntsc_frames_interpolate_on_a_configuration_that_renders_motion(monkeypatch, caplog):
    import re

    from kinovsr.native import temporal

    monkeypatch.setattr(temporal, "_CANVAS_DECISIONS", {})
    try:
        with caplog.at_level(logging.DEBUG, logger="kinovsr.native.temporal"):
            session = temporal.VtfrcSession(720, 480, 24, 60)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        canvas = session.canvas
        frame = (session.dst_attrs["Width"], session.dst_attrs["Height"])
    finally:
        session.close()

    threshold = temporal.MOTION_CHECK_MARGIN_DB
    if canvas is None:
        # A macOS whose NTSC network renders motion: the frame size passed.
        direct = re.search(r"FRC motion check at 720x480: ([-\d.]+) dB", caplog.text)
        assert direct is not None
        assert float(direct.group(1)) >= threshold
    else:
        # macOS 27: the frame size fails and padding into 960x540 renders motion.
        padded = re.search(
            r"wrong at 720x480 \(([-\d.]+) dB over the frame blend\); interpolating it "
            r"edge-padded into 960x540 \(([-\d.]+) dB\)",
            caplog.text,
        )
        assert canvas == (960, 540)
        assert padded is not None
        assert float(padded.group(1)) < threshold <= float(padded.group(2))
    assert frame == (720, 480)


@pytest.mark.integration
@pytest.mark.parametrize("size", [(640, 480), (320, 180)])
def test_sizes_frc_renders_correctly_keep_the_direct_path(monkeypatch, size):
    from kinovsr.native import temporal

    monkeypatch.setattr(temporal, "_CANVAS_DECISIONS", {})
    try:
        session = temporal.VtfrcSession(*size, 24, 60)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        assert session.canvas is None
    finally:
        session.close()


@pytest.mark.integration
@pytest.mark.solo
def test_a_checked_session_matches_an_unchecked_one_in_each_source_format(monkeypatch):
    # The check runs RGBAHalf frames through the session it keeps, and FRC keeps
    # the source format of its first request: a BGRA segment must get a fresh
    # processor, not error -19730. 352x288 runs padded into 640x360.
    from kinovsr.native import temporal

    monkeypatch.setattr(temporal, "_CANVAS_DECISIONS", {})
    checked = direct = None
    results = []
    try:
        checked = temporal.VtfrcSession(352, 288, 24, 60)
        direct = temporal.VtfrcSession(352, 288, 24, 60, canvas=checked.canvas)
        for format_name in ("PIX_RGBAHALF", "PIX_BGRA"):
            pair = []
            for session in (checked, direct):
                pair.append(_interpolate(session, 3, format_name))
                session.reset_temporal_context()
            results.append(pair)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        for session in (checked, direct):
            if session is not None:
                session.close()

    for after_check, without_check in results:
        assert len(after_check) == 8
        assert after_check == without_check


@pytest.mark.integration
@pytest.mark.solo
def test_padded_destination_batches_match_one_request(monkeypatch):
    from fractions import Fraction

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import temporal

    source_fps = Fraction(24000, 1001)
    target_fps = Fraction(168000, 1001)
    one = batched = None
    try:
        monkeypatch.setattr(temporal, "DESTINATION_BATCH_SIZE", 32)
        monkeypatch.setattr(temporal, "CANVAS_DST_POOL_ALLOCATION_LIMIT", 32)
        one = temporal.VtfrcSession(352, 288, source_fps, target_fps, canvas=(640, 360))
        external = pb.make_pool_from_attrs(one.dst_attrs)
        assert external is not None
        one.use_dst_pool(external)
        expected = _interpolate(one, 2)
        one.close()
        one = None

        monkeypatch.setattr(temporal, "DESTINATION_BATCH_SIZE", 4)
        monkeypatch.setattr(temporal, "CANVAS_DST_POOL_ALLOCATION_LIMIT", 4)
        batched = temporal.VtfrcSession(352, 288, source_fps, target_fps, canvas=(640, 360))
        actual = _interpolate(batched, 2)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        for session in (one, batched):
            if session is not None:
                session.close()

    assert len(actual) == 14
    assert actual == expected


@pytest.mark.integration
@pytest.mark.parametrize("format_name", ["PIX_BGRA", "PIX_NV12", "PIX_RGBAHALF"])
def test_padded_sessions_take_every_accepted_source_format(format_name):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import temporal

    try:
        session = temporal.VtfrcSession(352, 288, 24, 60, canvas=(640, 360))
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        produced = _interpolate(session, 3, format_name)
    finally:
        session.close()

    assert len(produced) == 8
    assert {entry[:3] for entry in produced} == {(352, 288, pb.PIX_RGBAHALF)}


@pytest.mark.integration
@pytest.mark.parametrize("canvas", [None, (640, 360)])
def test_a_session_keeps_its_own_copy_of_each_source(canvas):
    # FRC holds the previous source between calls; holding the caller's buffer
    # would starve a bounded upstream pool such as VT super resolution's.
    from kinovsr.native import temporal
    from kinovsr.native.frameworks import Quartz

    try:
        session = temporal.VtfrcSession(352, 288, 24, 60, canvas=canvas)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        for index in range(2):
            source = _source(352, 288, index)
            list(session.feed(source, index))
            kept = session._prev_src_pb
            assert kept is not source
            size = (Quartz.CVPixelBufferGetWidth(kept), Quartz.CVPixelBufferGetHeight(kept))
            assert size == (canvas or (352, 288))
    finally:
        session.close()


@pytest.mark.integration
def test_a_padded_session_refuses_a_source_of_another_size():
    from kinovsr.native import temporal

    try:
        session = temporal.VtfrcSession(352, 288, 24, 60, canvas=(640, 360))
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        with pytest.raises(ValueError, match="interpolates 352x288 frames"):
            list(session.feed(_source(320, 240, 0), 0))
    finally:
        session.close()


@pytest.mark.integration
def test_a_canvas_equal_to_the_frame_is_the_direct_path():
    from kinovsr.native import temporal

    try:
        session = temporal.VtfrcSession(320, 180, 24, 60, canvas=(320, 180))
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        assert session.canvas is None
        assert session._canvas_dst_pool is None
    finally:
        session.close()
