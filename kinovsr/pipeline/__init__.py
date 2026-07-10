"""Pipeline orchestration: chain resolution, validation, and scheduling.

The builder (:mod:`kinovsr.pipeline.builder`) resolves a composed user
config against the processor catalog and preflight-validates the whole
chain by threading a ``StreamSpec`` from input endpoint to output
endpoint. The scheduler (M3 step 4) drives resolved chains over frames.
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

__all__ = [
    "INPUT_ENDPOINT",
    "OUTPUT_ENDPOINT",
    "BuildPlan",
    "OutputEndpointSpec",
    "ResolvedStage",
    "build_processors",
    "resolve_pipeline",
]
