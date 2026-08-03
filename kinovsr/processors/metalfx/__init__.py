"""MetalFX spatial upscaling as a processor family.

MTLFXSpatialScaler is Apple's game-upscaler network run as a single
Metal encode. On video it synthesizes high-frequency texture the source
never had - crisper edges than any resampler, with a content-dependent
crawl on stochastic texture (sand, brick, foliage). Like any learned
model that trade cuts both ways; what earns it a place in the catalog
is cost: measured ~1 ms/frame at 320x180 -> 4x including readback into
MLX, roughly 20x cheaper than the VideoToolbox HQ scaler.

The scaler runs in fp16 (RGBA16Float textures both sides), so the MLX
float chain crosses without quantizing through 8-bit. Values are
display-referred [0, 1] RGB and the scaler is configured for its
perceptual (sRGB-encoded) color mode to match.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.errors import MediaError
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings

_SCALES = (2, 3, 4)
_DEFAULT_SCALE = 2


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedMetalFxInput:
    rgba: bytes
    width: int
    height: int
    dtype: Any


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedMetalFxOutput:
    rgba: memoryview
    width: int
    height: int
    dtype: Any


class MetalFxSpatialUpscaler:
    """feed()/flush() driver for the MetalFX spatial scaler.

    Stateless per-frame upscale: each frame is emitted immediately, so
    feed()/flush() mirror the other per-frame upscalers and the harness
    wiring stays parallel. The Metal device, scaler, and textures are
    created on the first frame (geometry comes from the frame itself)
    and reused for the stream; a mid-stream geometry change raises.
    """

    def __init__(self, scale: int = _DEFAULT_SCALE):
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}")
        self.scale = scale
        self._state: tuple | None = None   # (queue, scaler, in_tex, out_tex, readback, w, h)

    def _setup(self, width: int, height: int) -> tuple:
        import Metal
        import MetalFX

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None or not (
                MetalFX.MTLFXSpatialScalerDescriptor.supportsDevice_(device)):
            raise MediaError(
                "MetalFX spatial scaling is not supported on this device")
        fmt = Metal.MTLPixelFormatRGBA16Float
        descriptor = MetalFX.MTLFXSpatialScalerDescriptor.alloc().init()
        descriptor.setColorTextureFormat_(fmt)
        descriptor.setOutputTextureFormat_(fmt)
        descriptor.setInputWidth_(width)
        descriptor.setInputHeight_(height)
        descriptor.setOutputWidth_(width * self.scale)
        descriptor.setOutputHeight_(height * self.scale)
        descriptor.setColorProcessingMode_(
            MetalFX.MTLFXSpatialScalerColorProcessingModePerceptual)
        scaler = descriptor.newSpatialScalerWithDevice_(device)
        if scaler is None:
            raise MediaError(
                f"MetalFX refused a {width}x{height} -> {self.scale}x "
                f"fp16 spatial scaler")

        def texture(w: int, h: int, usage: int, *, private: bool):
            d = Metal.MTLTextureDescriptor.\
                texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                    fmt, w, h, False)
            d.setStorageMode_(Metal.MTLStorageModePrivate if private
                              else Metal.MTLStorageModeShared)
            d.setUsage_(usage)
            return device.newTextureWithDescriptor_(d)

        in_tex = texture(width, height, scaler.colorTextureUsage(),
                         private=False)
        out_tex = texture(width * self.scale, height * self.scale,
                          scaler.outputTextureUsage(), private=True)
        scaler.setInputContentWidth_(width)
        scaler.setInputContentHeight_(height)
        scaler.setColorTexture_(in_tex)
        scaler.setOutputTexture_(out_tex)
        readback = device.newBufferWithLength_options_(
            width * self.scale * height * self.scale * 8,
            Metal.MTLResourceStorageModeShared)
        return (device.newCommandQueue(), scaler, in_tex, out_tex,
                readback, width, height)

    def prepare_input(self, rgb: Any) -> _PreparedMetalFxInput:
        import mlx.core as mx

        frame = rgb[0] if rgb.ndim == 4 else rgb
        height, width = int(frame.shape[0]), int(frame.shape[1])
        clipped = mx.clip(frame[..., :3].astype(mx.float32), 0.0, 1.0)
        rgba = mx.contiguous(
            mx.concatenate(
                [clipped, mx.ones((height, width, 1))], axis=-1
            ).astype(mx.float16)
        )
        mx.eval(rgba)
        return _PreparedMetalFxInput(
            rgba=bytes(memoryview(rgba)),
            width=width,
            height=height,
            dtype=frame.dtype,
        )

    def _feed_prepared(
        self,
        prepared: _PreparedMetalFxInput,
        token: Any,
    ) -> _PreparedMetalFxOutput:
        import Metal

        height, width = prepared.height, prepared.width
        if self._state is None:
            self._state = self._setup(width, height)
        queue, scaler, in_tex, out_tex, readback, w, h = self._state
        if (width, height) != (w, h):
            raise MediaError(
                f"frame geometry changed mid-stream: scaler is bound to "
                f"{w}x{h}, got {width}x{height}")

        in_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            Metal.MTLRegionMake2D(0, 0, width, height), 0,
            prepared.rgba, width * 8)

        ow, oh = width * self.scale, height * self.scale
        cmd = queue.commandBuffer()
        scaler.encodeToCommandBuffer_(cmd)
        blit = cmd.blitCommandEncoder()
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_(
            out_tex, 0, 0, Metal.MTLOriginMake(0, 0, 0),
            Metal.MTLSizeMake(ow, oh, 1), readback, 0, ow * 8, ow * oh * 8)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.error() is not None:
            raise MediaError(f"MetalFX encode failed: {cmd.error()}")

        raw = memoryview(
            readback.contents().as_buffer(ow * oh * 8)
        ).cast("B")
        return _PreparedMetalFxOutput(
            rgba=raw,
            width=ow,
            height=oh,
            dtype=prepared.dtype,
        )

    def prepare_output(self, output: _PreparedMetalFxOutput) -> Any:
        import mlx.core as mx

        raw = mx.array(output.rgba)
        sr = mx.view(raw, mx.float16).reshape(
            output.height, output.width, 4
        )[..., :3]
        sr = mx.clip(sr, 0.0, 1.0).astype(output.dtype)
        mx.eval(sr)
        return sr

    def feed(self, rgb: Any, token: Any = None) -> list:
        if isinstance(rgb, _PreparedMetalFxInput):
            return [(self._feed_prepared(rgb, token), token)]
        prepared = self.prepare_input(rgb)
        output = self._feed_prepared(prepared, token)
        return [(self.prepare_output(output), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._state = None


@dataclasses.dataclass(frozen=True, slots=True)
class MetalFxStageConfig:
    scale: int


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, MetalFxStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(config.scale))
    return dataclasses.replace(spec, frame=frame)


class MetalFxFactory:
    name = "metalfx"
    execution_affinity = "metalfx:{stage}"
    execution_input_affinity = "mlx"
    execution_output_affinity = "mlx"
    execution_resources = ("gpu", "memory_bandwidth")
    execution_native_slots = 2

    # No profiles: the family has no weights to pick between - the model
    # ships inside the OS framework. The only knob is the scale factor.
    capabilities = {
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_produces,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> MetalFxStageConfig:
        reject_unknown_keys(raw, ("scale",))
        scale = typed_value(raw, "scale", int)
        if scale is None:
            scale = _DEFAULT_SCALE
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}")
        return MetalFxStageConfig(scale=scale)

    def build(self, config: MetalFxStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        return FeedFlushProcessor(
            lambda: MetalFxSpatialUpscaler(scale=config.scale))


FACTORY = MetalFxFactory()

__all__ = [
    "FACTORY",
    "MetalFxFactory",
    "MetalFxSpatialUpscaler",
    "MetalFxStageConfig",
]
