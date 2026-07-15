"""Native ImageIO status failures are operational filesystem errors."""

from __future__ import annotations

import contextlib

import pytest

from kinovsr.media import images

pytestmark = pytest.mark.unit


def _patch_image_boundary(monkeypatch):
    monkeypatch.setattr(images, "autorelease_pool", contextlib.nullcontext)
    monkeypatch.setattr(images, "_url", lambda path: object())
    monkeypatch.setattr(images, "_mx_to_cgimage", lambda image: object())


def test_save_image_destination_failure_is_os_error(monkeypatch, tmp_path):
    _patch_image_boundary(monkeypatch)
    monkeypatch.setattr(
        images.Quartz,
        "CGImageDestinationCreateWithURL",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(OSError, match="cannot create image destination"):
        images.save_image(object(), tmp_path / "frame.png")


def test_save_image_finalize_failure_is_os_error(monkeypatch, tmp_path):
    _patch_image_boundary(monkeypatch)
    monkeypatch.setattr(
        images.Quartz,
        "CGImageDestinationCreateWithURL",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        images.Quartz,
        "CGImageDestinationAddImage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        images.Quartz,
        "CGImageDestinationFinalize",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(OSError, match="failed to write image"):
        images.save_image(object(), tmp_path / "frame.png")


def test_native_bitmap_context_failure_is_operational(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(
        images.Quartz, "CGBitmapContextCreate", lambda *args: None)
    with pytest.raises(RuntimeError, match="CGBitmapContextCreate"):
        images._mx_to_cgimage(mx.zeros((4, 4, 3), dtype=mx.uint8))
