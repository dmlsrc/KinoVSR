"""Regenerate kinovsr/native/anemil/coreml_spec.fds from coremltools.

Dev-side only, run once per schema refresh - that is, only if anemil ever
needs an op from a Core ML opset newer than the vendored descriptor set
carries. Collects the transitive FileDescriptorProto closure of
``CoreML.Specification.Model`` (dependencies first, so a DescriptorPool can
add them in order) and writes it as a serialized FileDescriptorSet plus a
provenance sidecar with the source version, file list, and content hash.

The vendored .fds is stable DATA: unlike protoc-generated _pb2 modules it
carries no generated-code/runtime version coupling, which is the whole
reason ``kinovsr.native.anemil.schema`` loads schemas dynamically. After a
refresh, importing the schema re-asserts the baked enum values against the
new descriptor set, and both ANE conversion paths must be re-verified.

Requires an installed coremltools (NOT a project dependency; any recent
version on any machine works - the output is Apple's public schema, not
coremltools code).

Run: "$KINOVSR_PYTHON" scripts/dev/make_coreml_fds.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

_log = logging.getLogger("kinovsr.dev.make_coreml_fds")


def default_out() -> Path:
    import kinovsr.native.anemil as anemil

    return Path(anemil.__file__).resolve().parent / "coreml_spec.fds"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .fds path (default: the vendored file inside "
             "kinovsr/native/anemil). Point elsewhere to compare a fresh "
             "capture against the vendored bytes without touching them.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import coremltools
    from coremltools.proto import Model_pb2
    from google.protobuf import descriptor_pb2

    ordered = []
    seen = set()

    def visit(file_descriptor):
        if file_descriptor.name in seen:
            return
        seen.add(file_descriptor.name)
        for dependency in file_descriptor.dependencies:
            visit(dependency)
        proto = descriptor_pb2.FileDescriptorProto()
        file_descriptor.CopyToProto(proto)
        ordered.append(proto)

    visit(Model_pb2.Model.DESCRIPTOR.file)

    fds = descriptor_pb2.FileDescriptorSet()
    fds.file.extend(ordered)
    payload = fds.SerializeToString(deterministic=True)
    out = arguments.out or default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    out.with_suffix(".provenance.json").write_text(json.dumps({
        "source": f"coremltools {coremltools.__version__}",
        "generated": str(date.today()),
        "files": [p.name for p in ordered],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }, indent=2) + "\n")
    _log.info("wrote %s (%d bytes, %d proto files)",
              out, len(payload), len(ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
