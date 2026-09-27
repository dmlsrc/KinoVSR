"""Output-filename helpers for file endpoints.

Foundation-level naming utilities shared by the file-to-file entry points:
they turn a user-supplied prefix into a shell-friendly output stem without
reaching into orchestration or CLI code.
"""

from pathlib import Path


def sanitize_output_prefix(prefix: str | None) -> str:
    """Keep generated filenames shell-friendly while preserving readable prefixes."""
    prefix = (prefix or "kinovsr").strip()
    if not prefix:
        prefix = "kinovsr"
    sanitized = []
    for char in prefix:
        if char.isalnum() or char in ("-", "_", "."):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("._") or "kinovsr"


def default_output_prefix(video: Path | str) -> str:
    """Name a file run after its input, keeping the same safe stem policy."""
    return "vsr_" + sanitize_output_prefix(Path(video).stem)
