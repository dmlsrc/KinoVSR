"""FRC fan-out is bounded without changing VideoToolbox output bytes."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("needs_random", [False, True])
def test_destination_fanout_is_submitted_in_fixed_batches(
    monkeypatch, needs_random
):
    from kinovsr.native import temporal

    class Frame:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithBuffer_presentationTimeStamp_(self, buffer, pts):
            self.buffer = buffer
            self.pts = pts
            return self

    class Parameters:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithSourceFrame_nextFrame_opticalFlow_interpolationPhase_submissionMode_destinationFrames_(
            self, source, next_frame, _flow, phases, mode, destinations
        ):
            self.source = source
            self.next_frame = next_frame
            self.phases = phases
            self.mode = mode
            self.destinations = destinations
            return self

    calls = []

    class Processor:
        def processWithParameters_error_(self, params, _error):
            calls.append((len(params.destinations), params.mode))
            return True, None

    monkeypatch.setattr(temporal.vt, "VTFrameProcessorFrame", Frame)
    monkeypatch.setattr(temporal.vt, "VTFrameRateConversionParameters", Parameters)
    session = temporal.VtfrcSession.__new__(temporal.VtfrcSession)
    session.source_cadence = Fraction(24)
    session.target_cadence = Fraction(240)
    session.processor = Processor()
    session._make_dst_buffer = object
    session._submission_needs_random = needs_random

    outputs = list(
        session._process_destination_batches(
            object(),
            0,
            object(),
            1,
            list(range(10)),
            [index / 10 for index in range(10)],
            "failed",
        )
    )

    sequential = temporal.vt.VTFrameRateConversionParametersSubmissionModeSequential
    random = temporal.vt.VTFrameRateConversionParametersSubmissionModeRandom
    unchanged = (
        temporal.vt.VTFrameRateConversionParametersSubmissionModeSequentialReferencesUnchanged
    )
    assert len(outputs) == 10
    first = random if needs_random else sequential
    assert calls == [(4, first), (4, unchanged), (2, unchanged)]
    assert session._submission_needs_random is False


def _source_buffer(seed: int):
    import numpy as np

    from kinovsr.media import pixel_buffers as pb

    width, height = 128, 96
    y, x = np.mgrid[0:height, 0:width]
    rgba = np.stack(
        (
            ((x + seed * 3) % width) / (width - 1),
            ((y + seed * 5) % height) / (height - 1),
            ((x + y + seed * 7) % (width + height)) / (width + height - 1),
            np.ones_like(x),
        ),
        axis=-1,
    ).astype(np.float16)
    buffer = pb.make_pixel_buffer_from_attrs(
        width,
        height,
        {
            "PixelFormatType": pb.PIX_RGBAHALF,
            "Width": width,
            "Height": height,
            "IOSurfaceProperties": {},
            "MetalCompatibility": True,
        },
    )
    pb.write_fp16_rgba(rgba, buffer)
    return buffer


def _active_digest(buffer) -> str:
    from kinovsr.native.frameworks import Quartz

    assert not Quartz.CVPixelBufferIsPlanar(buffer)
    width = int(Quartz.CVPixelBufferGetWidth(buffer))
    height = int(Quartz.CVPixelBufferGetHeight(buffer))
    row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(row_bytes * height)
        active = b"".join(
            bytes(view[row * row_bytes : row * row_bytes + width * 8]) for row in range(height)
        )
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return hashlib.sha256(active).hexdigest()


def _output_digests(outputs) -> list[str]:
    digests = []
    for output in outputs:
        digests.append(_active_digest(output))
        del output
    return digests


def _consume_pair(session, first, second) -> list[str]:
    assert list(session.feed(first, 0)) == []
    return [
        *_output_digests(session.feed(second, 1)),
        *_output_digests(session.drain()),
    ]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["normal", "high"])
def test_batched_ntsc_fanout_is_bit_exact_to_one_call(monkeypatch, mode):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import temporal

    source_fps = Fraction(24000, 1001)
    target_fps = Fraction(168000, 1001)
    first, second = _source_buffer(1), _source_buffer(2)
    baseline = batched = None
    try:
        baseline = temporal.VtfrcSession(128, 96, source_fps, target_fps, mode=mode)
        external = pb.make_pool_from_attrs(baseline.dst_attrs)
        assert external is not None
        baseline.use_dst_pool(external)
        monkeypatch.setattr(temporal, "DESTINATION_BATCH_SIZE", 32)
        expected = _consume_pair(baseline, first, second)
        baseline.close()
        baseline = None

        monkeypatch.setattr(temporal, "DESTINATION_BATCH_SIZE", 4)
        batched = temporal.VtfrcSession(128, 96, source_fps, target_fps, mode=mode)
        actual = _consume_pair(batched, first, second)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        if baseline is not None:
            baseline.close()
        if batched is not None:
            batched.close()

    assert len(actual) == 14
    assert actual == expected


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["normal", "high"])
def test_post_cut_random_submission_matches_fresh_session(mode):
    from kinovsr.native import temporal

    before_first = _source_buffer(1)
    before_last = _source_buffer(2)
    after_first = _source_buffer(31)
    after_second = _source_buffer(32)
    reused = fresh = None
    try:
        reused = temporal.VtfrcSession(
            128,
            96,
            24,
            48,
            mode=mode,
        )
        assert _output_digests(reused.feed(before_first, 0)) == []
        _output_digests(reused.feed(before_last, 1))
        _output_digests(reused.drain())
        reused.reset_temporal_context()
        assert _output_digests(reused.feed(after_first, 2)) == []
        actual = _output_digests(reused.feed(after_second, 3))

        fresh = temporal.VtfrcSession(
            128,
            96,
            24,
            48,
            mode=mode,
        )
        assert _output_digests(fresh.feed(after_first, 2)) == []
        expected = _output_digests(fresh.feed(after_second, 3))
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        if reused is not None:
            reused.close()
        if fresh is not None:
            fresh.close()

    assert len(actual) == 2
    assert actual == expected


@pytest.mark.integration
def test_geometry_and_profile_change_gets_a_new_bounded_pool():
    from kinovsr.native import temporal
    from kinovsr.native.frameworks import Quartz

    first = second = None
    try:
        first = temporal.VtfrcSession(128, 96, 24, 60, mode="normal")
        first_pool = first._dst_pool
        first_attrs = dict(Quartz.CVPixelBufferPoolGetAttributes(first_pool))
        assert (
            first_attrs[Quartz.kCVPixelBufferPoolMinimumBufferCountKey]
            == temporal.DST_POOL_ALLOCATION_LIMIT
        )
        first.close()
        first = None

        second = temporal.VtfrcSession(160, 120, 24, 60, mode="high")
        assert second._dst_pool is not first_pool
        pixel_attrs = dict(Quartz.CVPixelBufferPoolGetPixelBufferAttributes(second._dst_pool))
        assert pixel_attrs[Quartz.kCVPixelBufferWidthKey] == 160
        assert pixel_attrs[Quartz.kCVPixelBufferHeightKey] == 120
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()

    assert second._dst_pool is None
