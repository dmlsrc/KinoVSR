"""Typed pipeline failures.

The taxonomy distinguishes what the user can fix (config, media) from
genuine runtime faults. Everything raised before frame processing carries
enough structure to say exactly what to change: :class:`StreamEdgeError`
names both sides of the failing edge and lists field-level violations.
"""

from .specs import FieldViolation, StreamSpec, describe_spec


class PipelineError(Exception):
    """Base of all typed pipeline failures."""


class StageConfigError(PipelineError):
    """A stage's configuration is invalid; message carries the stage name."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class UnknownStageError(StageConfigError):
    """The stage names an unknown family, capability, or profile."""


class MediaError(PipelineError):
    """The input or output endpoint cannot handle the media as declared."""


class StreamEdgeError(PipelineError):
    """A chain edge does not validate: the producer's stream contract
    violates the consumer's constraint. Names both sides and every
    mismatched field, so any invalid ordering fails with a precise reason
    before processing starts."""

    def __init__(
        self,
        upstream: str,
        downstream: str,
        violations: tuple[FieldViolation, ...],
        produced: StreamSpec | None = None,
    ) -> None:
        self.upstream = upstream
        self.downstream = downstream
        self.violations = violations
        self.produced = produced
        lines = [
            f"invalid edge {upstream} -> {downstream}:",
        ]
        lines += [
            f"  {v.field}: {downstream} accepts {v.accepted}; {upstream} produces {v.actual}"
            for v in violations
        ]
        if produced is not None:
            lines.append(f"  (upstream stream: {describe_spec(produced)})")
        super().__init__("\n".join(lines))


class PipelineRuntimeError(PipelineError):
    """A stage failed while frames were flowing; message carries the stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


def append_cleanup_context(primary: BaseException, cleanup: BaseException) -> None:
    """Chain a cleanup failure onto ``primary``'s context without cycles.

    Appends ``cleanup`` at the end of ``primary``'s ``__context__`` chain and
    cuts any link that would lead back into either chain, so the active
    exception is never retained as a back-link."""
    cursor = primary
    primary_ids = {id(primary)}
    while cursor.__context__ is not None:
        next_error = cursor.__context__
        if id(next_error) in primary_ids:
            cursor.__context__ = None
            break
        primary_ids.add(id(next_error))
        cursor = next_error
    tail = cursor

    if id(cleanup) in primary_ids:
        return
    cursor = cleanup
    cleanup_ids = {id(cleanup)}
    while cursor.__context__ is not None:
        next_error = cursor.__context__
        if id(next_error) in primary_ids or id(next_error) in cleanup_ids:
            cursor.__context__ = None
            break
        cleanup_ids.add(id(next_error))
        cursor = next_error
    tail.__context__ = cleanup


__all__ = [
    "MediaError",
    "PipelineError",
    "PipelineRuntimeError",
    "StageConfigError",
    "StreamEdgeError",
    "UnknownStageError",
    "append_cleanup_context",
]
