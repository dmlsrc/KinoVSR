"""Pipeline orchestration: chain resolution, validation, and scheduling.

The builder (:mod:`kinovsr.pipeline.builder`) resolves a composed user
config against the processor catalog and preflight-validates the whole
chain by threading a ``StreamSpec`` from input endpoint to output
endpoint. The scheduler (M3 step 4) drives resolved chains over frames;
the file endpoints (:mod:`kinovsr.pipeline.run`, M4 step 0) ground a
chain against real video files.
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
from .run import FileRunResult, FileSink, FileSource, run_file
from .scheduler import ChainRun, run_chain, run_plan

__all__ = [
    "INPUT_ENDPOINT",
    "OUTPUT_ENDPOINT",
    "BuildPlan",
    "OutputEndpointSpec",
    "ResolvedStage",
    "ChainRun",
    "FileRunResult",
    "FileSink",
    "FileSource",
    "build_processors",
    "resolve_pipeline",
    "run_chain",
    "run_file",
    "run_plan",
]
