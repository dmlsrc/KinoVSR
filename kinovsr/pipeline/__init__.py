"""Pipeline orchestration: chain resolution, validation, and scheduling.

The builder (:mod:`kinovsr.pipeline.builder`) resolves a composed user
config against the processor catalog and preflight-validates the whole
chain by threading a ``StreamSpec`` from input endpoint to output
endpoint. The scheduler drives resolved chains over frames; the session
(:mod:`kinovsr.pipeline.session`) is the host surface over both; the
file endpoints (:mod:`kinovsr.pipeline.run`) ground a chain against
real video files.
"""

from .builder import (
    INPUT_ENDPOINT,
    OUTPUT_ENDPOINT,
    BuildPlan,
    OutputEndpointSpec,
    ResolvedStage,
    build_processors,
    resolve_pipeline,
)
from .execution import (
    AffinityKey,
    BufferingSpec,
    ExecutionSpec,
    Ordering,
    PhysicalOperation,
    ResourceClaim,
    ResourceKind,
)
from .leases import (
    Completion,
    Envelope,
    PayloadLease,
    StorageDescriptor,
    StorageKind,
)
from .run import FileRunResult, FileSink, FileSource, run_file
from .scheduler import ChainRun, run_chain, run_plan
from .session import PipelineSession, open_pipeline

__all__ = [
    "INPUT_ENDPOINT",
    "OUTPUT_ENDPOINT",
    "AffinityKey",
    "BufferingSpec",
    "BuildPlan",
    "ChainRun",
    "Completion",
    "Envelope",
    "ExecutionSpec",
    "FileRunResult",
    "FileSink",
    "FileSource",
    "Ordering",
    "OutputEndpointSpec",
    "PayloadLease",
    "PhysicalOperation",
    "PipelineSession",
    "ResolvedStage",
    "ResourceClaim",
    "ResourceKind",
    "StorageDescriptor",
    "StorageKind",
    "build_processors",
    "open_pipeline",
    "resolve_pipeline",
    "run_chain",
    "run_file",
    "run_plan",
]
