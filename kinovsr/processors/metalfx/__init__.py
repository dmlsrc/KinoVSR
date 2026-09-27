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

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, Protocol, cast

import Metal
import MetalFX
import mlx.core as mx

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.media.buffer import mlx_array_from_buffer
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
# Apple GPUs cap a 2D texture at 16384 px on a side, and Metal's descriptor
# validation aborts the process past it instead of returning nil.
_MAX_TEXTURE_SIDE = 16384
_DEFAULT_SCALE = 2


class _BufferContents(Protocol):
    def as_buffer(self, count: int) -> memoryview: ...


class _MetalBuffer(Protocol):
    def contents(self) -> _BufferContents: ...


class _Texture(Protocol):
    def replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
        self,
        region: object,
        level: int,
        data: bytes,
        bytes_per_row: int,
    ) -> None: ...


class _BlitEncoder(Protocol):
    def copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_(
        self,
        texture: object,
        source_slice: int,
        source_level: int,
        source_origin: object,
        source_size: object,
        buffer: object,
        destination_offset: int,
        destination_bytes_per_row: int,
        destination_bytes_per_image: int,
    ) -> None: ...

    def endEncoding(self) -> None: ...


class _CommandBuffer(Protocol):
    def blitCommandEncoder(self) -> _BlitEncoder: ...

    def commit(self) -> None: ...

    def waitUntilCompleted(self) -> None: ...

    def error(self) -> object | None: ...


class _CommandQueue(Protocol):
    def commandBuffer(self) -> _CommandBuffer: ...


class _SpatialScaler(Protocol):
    def colorTextureUsage(self) -> int: ...

    def outputTextureUsage(self) -> int: ...

    def setInputContentWidth_(self, width: int) -> None: ...

    def setInputContentHeight_(self, height: int) -> None: ...

    def setColorTexture_(self, texture: object) -> None: ...

    def setOutputTexture_(self, texture: object) -> None: ...

    def encodeToCommandBuffer_(self, command_buffer: object) -> None: ...


type _ScalerState = tuple[_CommandQueue, _SpatialScaler, _Texture, object, _MetalBuffer, int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedMetalFxInput:
    rgba: bytes
    width: int
    height: int
    dtype: mx.Dtype


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedMetalFxOutput:
    rgba: memoryview
    width: int
    height: int
    dtype: mx.Dtype


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
        self._state: _ScalerState | None = None  # (queue, scaler, in_tex, out_tex, readback, w, h)

    def _setup(self, width: int, height: int) -> _ScalerState:

        if max(width, height) * self.scale > _MAX_TEXTURE_SIDE:
            raise MediaError(_oversize_message(width, height, self.scale))
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None or not (MetalFX.MTLFXSpatialScalerDescriptor.supportsDevice_(device)):
            raise MediaError("MetalFX spatial scaling is not supported on this device")
        fmt = Metal.MTLPixelFormatRGBA16Float
        descriptor = MetalFX.MTLFXSpatialScalerDescriptor.alloc().init()
        descriptor.setColorTextureFormat_(fmt)
        descriptor.setOutputTextureFormat_(fmt)
        descriptor.setInputWidth_(width)
        descriptor.setInputHeight_(height)
        descriptor.setOutputWidth_(width * self.scale)
        descriptor.setOutputHeight_(height * self.scale)
        descriptor.setColorProcessingMode_(MetalFX.MTLFXSpatialScalerColorProcessingModePerceptual)
        scaler = cast(_SpatialScaler | None, descriptor.newSpatialScalerWithDevice_(device))
        if scaler is None:
            raise MediaError(
                f"MetalFX refused a {width}x{height} -> {self.scale}x fp16 spatial scaler"
            )

        def texture(w: int, h: int, usage: int, *, private: bool) -> _Texture:
            d = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                fmt, w, h, False
            )
            d.setStorageMode_(
                Metal.MTLStorageModePrivate if private else Metal.MTLStorageModeShared
            )
            d.setUsage_(usage)
            made = device.newTextureWithDescriptor_(d)
            if made is None:
                raise MediaError(f"Metal could not allocate a {w}x{h} fp16 texture for MetalFX")
            return cast(_Texture, made)

        in_tex = texture(width, height, scaler.colorTextureUsage(), private=False)
        out_tex = texture(
            width * self.scale, height * self.scale, scaler.outputTextureUsage(), private=True
        )
        scaler.setInputContentWidth_(width)
        scaler.setInputContentHeight_(height)
        scaler.setColorTexture_(in_tex)
        scaler.setOutputTexture_(out_tex)
        size = width * self.scale * height * self.scale * 8
        readback = device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
        queue = device.newCommandQueue()
        if readback is None or queue is None:
            raise MediaError(
                f"Metal could not allocate the MetalFX readback ({size} bytes) or queue"
            )
        return (
            cast(_CommandQueue, queue),
            scaler,
            in_tex,
            out_tex,
            cast(_MetalBuffer, readback),
            width,
            height,
        )

    def prepare_input(self, rgb: mx.array) -> _PreparedMetalFxInput:

        frame = rgb[0] if rgb.ndim == 4 else rgb
        height, width = int(frame.shape[0]), int(frame.shape[1])
        clipped = mx.clip(frame[..., :3].astype(mx.float32), 0.0, 1.0)
        rgba = mx.contiguous(
            mx.concatenate([clipped, mx.ones((height, width, 1))], axis=-1).astype(mx.float16)
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
    ) -> _PreparedMetalFxOutput:

        height, width = prepared.height, prepared.width
        if self._state is None:
            self._state = self._setup(width, height)
        queue, scaler, in_tex, out_tex, readback, w, h = self._state
        if (width, height) != (w, h):
            raise MediaError(
                f"frame geometry changed mid-stream: scaler is bound to "
                f"{w}x{h}, got {width}x{height}"
            )

        in_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            Metal.MTLRegionMake2D(0, 0, width, height), 0, prepared.rgba, width * 8
        )

        ow, oh = width * self.scale, height * self.scale
        cmd = queue.commandBuffer()
        if cmd is None:
            raise MediaError("Metal could not create a MetalFX command buffer")
        scaler.encodeToCommandBuffer_(cmd)
        blit = cmd.blitCommandEncoder()
        if blit is None:
            raise MediaError("Metal could not create the MetalFX readback encoder")
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_(
            out_tex,
            0,
            0,
            Metal.MTLOriginMake(0, 0, 0),
            Metal.MTLSizeMake(ow, oh, 1),
            readback,
            0,
            ow * 8,
            ow * oh * 8,
        )
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.error() is not None:
            raise MediaError(f"MetalFX encode failed: {cmd.error()}")

        raw = memoryview(readback.contents().as_buffer(ow * oh * 8)).cast("B")
        return _PreparedMetalFxOutput(
            rgba=raw,
            width=ow,
            height=oh,
            dtype=prepared.dtype,
        )

    def prepare_output(self, output: _PreparedMetalFxOutput) -> mx.array:

        raw = mlx_array_from_buffer(output.rgba)
        sr = mx.view(raw, mx.float16).reshape(output.height, output.width, 4)[..., :3]
        sr = mx.clip(sr, 0.0, 1.0).astype(output.dtype)
        mx.eval(sr)
        return sr

    def feed(
        self, rgb: mx.array | _PreparedMetalFxInput, token: object = None
    ) -> list[tuple[mx.array | _PreparedMetalFxOutput, object]]:
        if isinstance(rgb, _PreparedMetalFxInput):
            return [(self._feed_prepared(rgb), token)]
        prepared = self.prepare_input(rgb)
        output = self._feed_prepared(prepared)
        return [(self.prepare_output(output), token)]

    def flush(self) -> list[tuple[mx.array | _PreparedMetalFxOutput, object]]:
        return []

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._state = None


@dataclasses.dataclass(frozen=True, slots=True)
class MetalFxStageConfig:
    scale: int


def _oversize_message(width: int, height: int, scale: int) -> str:
    return (
        f"metalfx {scale}x output {width * scale}x{height * scale} exceeds the "
        f"{_MAX_TEXTURE_SIDE} px Metal texture limit; use a smaller --metalfx-scale"
    )


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, MetalFxStageConfig)
    g = spec.frame.geometry
    if max(g.width, g.height) * config.scale > _MAX_TEXTURE_SIDE:
        raise ValueError(_oversize_message(g.width, g.height, config.scale))
    frame = dataclasses.replace(spec.frame, geometry=g.scaled(config.scale))
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
    capabilities: ClassVar[dict[Capability, CapabilitySpec]] = {
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
        raw: Mapping[str, object],
        *,
        capability: Capability,  # noqa: ARG002 - protocol signature
        profile: str | None,  # noqa: ARG002 - protocol signature
        settings: Settings,  # noqa: ARG002 - protocol signature
    ) -> MetalFxStageConfig:
        reject_unknown_keys(raw, ("scale",))
        scale = typed_value(raw, "scale", int)
        if scale is None:
            scale = _DEFAULT_SCALE
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}")
        return MetalFxStageConfig(scale=scale)

    def build(self, config: MetalFxStageConfig, *, context: PipelineContext) -> FeedFlushProcessor:  # noqa: ARG002 - protocol signature
        return FeedFlushProcessor(lambda: MetalFxSpatialUpscaler(scale=config.scale))


FACTORY = MetalFxFactory()

__all__ = [
    "FACTORY",
    "MetalFxFactory",
    "MetalFxSpatialUpscaler",
    "MetalFxStageConfig",
]
