"""Scene-cut detector for resetting VSR's prev-frame chain on hard cuts.

Pure MLX on the CPU. VSR's `balanced` mode propagates previous source +
output frames into the upscaler for temporal coherence; across a hard cut
that's the wrong context and produces ghosting around the cut frame. For
continuous generated clips with no edits, mode="off" is correct. For
arbitrary video input on edited footage, enable one of the detection
modes so VSR resets at cut boundaries.

Three algorithms:
  simple  Downsampled-pixel mean absolute difference. ~1ms/frame. Catches
          hard cuts cleanly; doesn't flag dissolves (which don't ghost in
          VSR anyway because the temporal coherence kinda matches).
  hist    Per-channel 32-bin histogram chi-squared distance. ~3ms/frame.
          More robust to fast motion than simple-pixel diff.
  vtme    Trackability, not similarity: neighbor incoherence of the
          VTMotionEstimationSession 16x16 multipass block-motion field
          (macOS 26+), matched on z-normalized luma and scaled by the
          frame diagonal. Consecutive frames stay coherent even under
          fast motion and global gain swings; across a cut the matcher
          free-associates. On the deflicker-class oscillation fixture
          (alternating +-30% gain) simple and hist false-fire on EVERY
          pair while this mode stays quiet, only hard clipped flashes
          still read as boundaries (arguably correct for temporal
          state). ~7-10ms/frame on the media engine, zero GPU cost.

False positives are cheap (one frame of "no temporal context", visually
invisible). False negatives let a cut ghost - tune the threshold down.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import mlx.core as mx

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.processors.boundaries import Boundary, BoundaryKind
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.processors.units import FrameUnit
from kinovsr.settings import Settings


def _to_uint8_rgb(frame: Any) -> Any:
    """Coerce any frame format (uint8 RGB, uint8 RGBA, fp16 RGBA, fp32 RGBA)
    to a uint8 RGB mlx array for histogram / thumbnail work.
    """
    f = frame if isinstance(frame, mx.array) else mx.array(frame)
    if str(f.dtype).split(".")[-1] in ("float16", "float32"):
        return mx.clip(f[..., :3] * 255.0, 0, 255).astype(mx.uint8)
    if f.shape[-1] == 4:
        return f[..., :3]
    return f


def _frame_thumbnail(frame: Any, target_size: int = 32) -> Any:
    rgb = _to_uint8_rgb(frame)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    step_h = max(1, h // target_size)
    step_w = max(1, w // target_size)
    return mx.contiguous(rgb[::step_h, ::step_w])


def _channel_histogram(channel: Any, bins: int) -> Any:
    """Per-channel uint8 histogram, `bins` equal-width bins over [0, 256).

    mx has no bincount, so count via a one-hot sum: value v lands in bin
    v * bins // 256, which is np.histogram's binning for range=(0, 256).
    """
    idx = (channel.astype(mx.int32) * bins) // 256          # (H, W) in [0, bins)
    onehot = idx.reshape(-1, 1) == mx.arange(bins).reshape(1, -1)
    return mx.sum(onehot, axis=0)                            # (bins,) counts


def _frame_histogram(frame: Any, bins: int = 32) -> Any:
    rgb = _to_uint8_rgb(frame)
    hists = [_channel_histogram(rgb[..., c], bins) for c in range(3)]
    return mx.concatenate(hists).astype(mx.float32)


def _normalized_gray(frame: Any) -> Any:
    """Z-normalized luma replicated to RGB for gain-invariant matching.

    Global flicker (deflicker-class alternating gains) shifts luma
    wholesale; SAD block matching is not illumination-invariant, so
    without this a bright/dark swing reads as an incoherent field and
    false-fires the detector. Mean/std normalization cancels global
    gain and offset; hard clipping (a blown flash frame) still destroys
    structure and legitimately reads as a boundary."""
    rgb = _to_uint8_rgb(frame).astype(mx.float32)
    luma = rgb @ mx.array([0.2126, 0.7152, 0.0722])
    z = (luma - mx.mean(luma)) / (mx.sqrt(mx.var(luma)) + 1e-6)
    g = mx.clip(0.5 + 0.18 * z, 0.0, 1.0)
    out = mx.stack([g, g, g], axis=-1)
    mx.eval(out)
    return out


def _block_incoherence(blocks: Any) -> float:
    """Mean Euclidean difference between adjacent block vectors, px."""
    dh = blocks[:, 1:, :] - blocks[:, :-1, :]
    dv = blocks[1:, :, :] - blocks[:-1, :, :]
    total = (float(mx.sum(mx.sqrt(mx.sum(dh * dh, axis=-1))))
             + float(mx.sum(mx.sqrt(mx.sum(dv * dv, axis=-1)))))
    count = (dh.shape[0] * dh.shape[1]) + (dv.shape[0] * dv.shape[1])
    return total / max(count, 1)


class CutDetector:
    """Detects hard cuts between consecutive frames.

    Modes:
        "off"     no-op
        "simple"  downsampled-pixel MAD. threshold ~0.2-0.35 typical.
        "hist"    per-channel histogram chi-squared. threshold ~0.4-0.8.
        "vtme"    block-motion-field incoherence / frame diagonal
                  (macOS 26+). threshold ~0.05-0.09 typical; the gate
                  measured non-cut pairs at or under ~0.06 and true cuts
                  at or over ~0.088 across 854x480 and 426x240.

    Always returns False on the first frame (no previous to compare).
    """

    def __init__(self, mode: str, threshold: float):
        self.mode = mode
        self.threshold = float(threshold)
        self._prev: Any | None = None
        self._engine: Any = None

    def _vtme_stat(self, frame: Any) -> float | None:
        curr = _normalized_gray(frame)
        if self._engine is None:
            from kinovsr.modeling.vtme_flow import VtmeFlowEngine

            h, w = int(curr.shape[0]), int(curr.shape[1])
            self._engine = VtmeFlowEngine(w, h, block=16, multipass=True)
        if self._prev is None:
            self._prev = curr
            return None
        blocks = self._engine.compute_blocks(curr, self._prev)
        self._prev = curr
        diagonal = (self._engine.w ** 2 + self._engine.h ** 2) ** 0.5
        return _block_incoherence(blocks) / diagonal

    def is_cut(self, frame: Any) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "simple":
            curr = _frame_thumbnail(frame)
            if self._prev is None:
                self._prev = curr
                return False
            diff = mx.abs(curr.astype(mx.int16) - self._prev.astype(mx.int16))
            mad = (mx.mean(diff.astype(mx.float32)) / 255.0).item()
            self._prev = curr
            return bool(mad > self.threshold)
        if self.mode == "hist":
            curr = _frame_histogram(frame)
            if self._prev is None:
                self._prev = curr
                return False
            a, b = self._prev, curr
            eps = 1e-6
            chi2 = mx.sum((a - b) ** 2 / (a + b + eps))
            norm = (chi2 / (mx.sum(a) + eps)).item()
            self._prev = curr
            return bool(norm > self.threshold)
        if self.mode == "vtme":
            stat = self._vtme_stat(frame)
            return stat is not None and stat > self.threshold
        raise ValueError(f"Unknown cut-detect mode: {self.mode!r}")

    def reset_state(self) -> None:
        """Restart the comparison (post-cut); native sessions are kept."""
        self._prev = None

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()
        self._prev = None


# ===========================================================================
# Processor family: a HARD_CUT boundary emitter
# ===========================================================================

_MODES = ("simple", "hist", "vtme")

# Per-mode defaults when the user does not pass a threshold: the modes'
# statistics live on different scales. simple/hist keep the historical
# 0.25; vtme's 0.07 sits inside the measured non-cut/cut gap at both
# probed sizes (max non-cut ~0.059, min cut ~0.088).
_DEFAULT_THRESHOLDS = {"simple": 0.25, "hist": 0.25, "vtme": 0.07}


@dataclasses.dataclass(frozen=True, slots=True)
class CutDetectStageConfig:
    detect: str
    threshold: float


def _passthrough(spec: StreamSpec, config: object) -> StreamSpec:
    return spec


class CutDetectProcessor:
    """Passthrough stage that attaches a HARD_CUT boundary to the first
    unit of each new shot. The scheduler owns everything downstream of
    the mark: it flushes buffered tails before the cut and resets
    stateful stages so recurrence never bridges the discontinuity."""

    def __init__(self, config: CutDetectStageConfig) -> None:
        self._config = config
        self._detector: CutDetector | None = None
        self._index = 0

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        if self._config.detect == "vtme":
            # Fail at pipeline assembly, not on the first frame.
            from kinovsr.modeling.vtme_flow import VtmeFlowEngine

            VtmeFlowEngine.ensure_available()
        self._detector = CutDetector(self._config.detect,
                                     self._config.threshold)

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        index = self._index
        self._index += 1
        if self._detector.is_cut(unit.payload):
            yield unit.with_boundary(Boundary(
                BoundaryKind.HARD_CUT, source_index=index))
        else:
            yield unit

    def reset(self, boundary: Boundary,
              context: PipelineContext) -> None:
        # Post-cut the previous frame belongs to the old shot; restart the
        # comparison so the next frame cannot re-trigger against it. The
        # detector keeps its native session (geometry is unchanged).
        self._detector.reset_state()

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        return ()

    def close(self, context: PipelineContext) -> None:
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()


class CutDetectFactory:
    name = "cut_detect"

    capabilities = {
        Capability.PREPROCESS: CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_passthrough,
            emits_boundaries=(BoundaryKind.HARD_CUT,),
            temporal_mode=TemporalMode.CAUSAL,
            stateful=True,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, object],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> CutDetectStageConfig:
        reject_unknown_keys(raw, ("detect", "threshold"))
        detect = typed_value(raw, "detect", str, "simple")
        if detect not in _MODES:
            raise ValueError(
                f"detect must be one of {_MODES}; a chain without cut "
                f"detection simply omits this stage")
        threshold = typed_value(raw, "threshold", float)
        if threshold is None:
            threshold = _DEFAULT_THRESHOLDS[detect]
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        return CutDetectStageConfig(detect=detect, threshold=threshold)

    def build(self, config: CutDetectStageConfig, *,
              context: PipelineContext) -> CutDetectProcessor:
        return CutDetectProcessor(config)


FACTORY = CutDetectFactory()
