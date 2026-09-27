"""Exact static pickle-global scanning for tensor checkpoint gates."""

import pickletools
import zipfile
from pathlib import Path


class PickleScanError(RuntimeError):
    """Checkpoint pickle metadata cannot be scanned conservatively."""


_STRING_OPCODES = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
    "SHORT_BINSTRING",
    "BINSTRING",
}
_UNRESOLVED_STACK_GLOBAL = "<unresolved> STACK_GLOBAL"
_UNRESOLVED_EXTENSION = "<unresolved> pickle-extension"
_UNKNOWN = object()
_MARK = pickletools.markobject
_STORAGE_TOKENS = frozenset(
    {
        "FloatStorage",
        "HalfStorage",
        "BFloat16Storage",
        "DoubleStorage",
        "LongStorage",
        "IntStorage",
        "ShortStorage",
        "CharStorage",
        "ByteStorage",
        "BoolStorage",
    }
)
_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch", "Size"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    *(("torch", token) for token in _STORAGE_TOKENS),
}


def _is_allowed_global(module: str, name: str) -> bool:
    """Return whether the restricted reader can resolve this exact global."""
    return (module, name) in _ALLOWED_GLOBALS


def _global_reference(module: object, name: object) -> str:
    if not isinstance(module, str) or not isinstance(name, str):
        return _UNRESOLVED_STACK_GLOBAL
    return f"{module} {name}"


def _split_global_reference(reference: str) -> tuple[str, str] | None:
    module, separator, name = reference.partition(" ")
    if not separator or not module or not name or " " in name:
        return None
    return module, name


def _pop_stack(stack: list[object], opcode: str) -> object:
    if not stack:
        raise PickleScanError(f"{opcode} underflowed the symbolic pickle stack")
    return stack.pop()


def _apply_stack_effect(
    stack: list[object],
    opcode: pickletools.OpcodeInfo,
) -> None:
    before = opcode.stack_before
    if pickletools.stackslice in before:
        try:
            marker = len(stack) - 1 - stack[::-1].index(_MARK)
        except ValueError as exc:
            raise PickleScanError(f"{opcode.name} has no MARK") from exc
        fixed = before.index(pickletools.markobject)
        del stack[marker:]
        for _ in range(fixed):
            _pop_stack(stack, opcode.name)
    else:
        for _ in before:
            _pop_stack(stack, opcode.name)
    stack.extend(_UNKNOWN for _ in opcode.stack_after)


def _scan_one_pickle(data: bytes) -> tuple[set[str], int]:
    references: set[str] = set()
    stack: list[object] = []
    memo: dict[int, object] = {}
    consumed = 0
    try:
        for opcode, argument, position in pickletools.genops(data):
            name = opcode.name
            if name in _STRING_OPCODES:
                stack.append(str(argument))
            elif name == "GLOBAL":
                module, separator, global_name = str(argument).partition(" ")
                references.add(
                    _global_reference(module, global_name)
                    if separator
                    else _UNRESOLVED_STACK_GLOBAL
                )
                stack.append(_UNKNOWN)
            elif name == "STACK_GLOBAL":
                stack_name = _pop_stack(stack, name)
                stack_module = _pop_stack(stack, name)
                references.add(_global_reference(stack_module, stack_name))
                stack.append(_UNKNOWN)
            elif name in {"BINPUT", "LONG_BINPUT", "PUT"}:
                if not stack:
                    _pop_stack(stack, name)
                assert argument is not None  # pickletools gives PUT opcodes an int
                memo[int(argument)] = stack[-1]
            elif name == "MEMOIZE":
                if not stack:
                    _pop_stack(stack, name)
                memo[len(memo)] = stack[-1]
            elif name in {"BINGET", "LONG_BINGET", "GET"}:
                assert argument is not None  # pickletools gives GET opcodes an int
                index = int(argument)
                if index not in memo:
                    raise PickleScanError(f"{name} references missing memo entry {index}")
                stack.append(memo[index])
            elif name == "MARK":
                stack.append(_MARK)
            elif name == "POP":
                _pop_stack(stack, name)
            elif name == "POP_MARK":
                try:
                    marker = len(stack) - 1 - stack[::-1].index(_MARK)
                except ValueError as exc:
                    raise PickleScanError("POP_MARK has no MARK") from exc
                del stack[marker:]
            elif name == "DUP":
                if not stack:
                    _pop_stack(stack, name)
                stack.append(stack[-1])
            elif name == "INST":
                module, separator, global_name = str(argument).partition(" ")
                references.add(
                    _global_reference(module, global_name)
                    if separator
                    else _UNRESOLVED_STACK_GLOBAL
                )
                _apply_stack_effect(stack, opcode)
            elif name in {"EXT1", "EXT2", "EXT4"}:
                references.add(_UNRESOLVED_EXTENSION)
                _apply_stack_effect(stack, opcode)
            else:
                _apply_stack_effect(stack, opcode)
            if name == "STOP":
                assert position is not None  # genops over bytes reports positions
                consumed = position + 1
                break
    except PickleScanError:
        raise
    except Exception as exc:
        raise PickleScanError(str(exc)) from exc
    if not consumed:
        raise PickleScanError("missing STOP opcode")
    return references, consumed


def scan_pickle_globals(data: bytes) -> set[str]:
    """List globals referenced by one pickle without unpickling it."""
    references, _consumed = _scan_one_pickle(data)
    return references


def suspicious_globals(references: set[str]) -> list[str]:
    """Return globals outside the restricted reader's exact allowlist."""
    return sorted(
        reference
        for reference in references
        if (
            (parsed := _split_global_reference(reference)) is None
            or not _is_allowed_global(*parsed)
        )
    )


def scan_checkpoint_globals(
    source: Path | str,
    *,
    max_pickle_bytes: int = 64 * 1024 * 1024,
) -> set[str]:
    """Scan zip metadata or up to five metadata pickles in a legacy stream."""
    path = Path(source)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                matches = [
                    info
                    for info in archive.infolist()
                    if info.filename == "data.pkl" or info.filename.endswith("/data.pkl")
                ]
                if len(matches) != 1:
                    detail = "no" if not matches else "multiple"
                    raise PickleScanError(f"checkpoint carries {detail} data.pkl members")
                info = matches[0]
                if info.file_size > max_pickle_bytes:
                    raise PickleScanError(
                        f"checkpoint pickle is {info.file_size} bytes; limit is {max_pickle_bytes}"
                    )
                return scan_pickle_globals(archive.read(info))

        with path.open("rb") as handle:
            data = handle.read(max_pickle_bytes + 1)
        references: set[str] = set()
        offset = 0
        for index in range(5):
            if offset == len(data):
                break
            try:
                current, consumed = _scan_one_pickle(data[offset:])
            except PickleScanError as exc:
                raise PickleScanError(
                    f"cannot scan legacy metadata pickle {index + 1}: {exc}"
                ) from exc
            references.update(current)
            offset += consumed
            if offset > max_pickle_bytes:
                raise PickleScanError(
                    f"legacy checkpoint metadata exceeds {max_pickle_bytes} bytes"
                )
        return references
    except PickleScanError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PickleScanError(f"cannot scan checkpoint {path}: {exc}") from exc


__all__ = [
    "PickleScanError",
    "scan_checkpoint_globals",
    "scan_pickle_globals",
    "suspicious_globals",
]
