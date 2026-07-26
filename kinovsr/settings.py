"""Environment-derived settings - single source of truth.

Anywhere in the package that needs an environment-backed value (a UX mode,
a scratch directory, a legacy weight override) imports :class:`Settings`
and reads it from there. Nothing else should call ``os.environ.get(...)``.

Project-specific env vars use the ``KINOVSR_*`` prefix. Legacy per-family
weight overrides (``NAFNET_WEIGHTS``, ``TOFLOW_WEIGHTS``, ...) predate this
module and are honored as compatibility fallbacks; they are declared here
and nowhere else.

Each :class:`Settings` field declares its env source(s) in
``metadata["env"]``. A single string is one source; a list is a fallback
chain (try the first; if unresolved, try the next).

**Substitution is explicit, never implicit:** ``{{VAR}}`` references are
looked up in ``os.environ``; entries with no ``{{}}`` are literal values.
A bare string like ``"/tmp"`` in a fallback list reads as a literal path,
not an env-var name, so a chain like ``["{{SHARED_TEMP_DIR}}", "/tmp"]``
cannot accidentally treat the fallback as a variable name. If any
``{{VAR}}`` in a template is unset or empty, the entry is unresolved and
the next fallback is tried.

**Boolean env parsing is truthy-string.** ``""``, ``"0"``, ``"false"``,
``"no"``, and ``"off"`` (case-insensitive) are false; anything else is
true. This deliberately preserves the historical behavior of
``KINOVSR_VERBOSE``, which treated any non-empty value as enabled.

**CLI overrides are taken verbatim.** No ``{{VAR}}`` substitution runs on
values parsed from ``--kebab-case`` flags; type conversion still applies.

:meth:`Settings.from_env` and :func:`add_argparse_args` are generic loops
over ``dataclasses.fields(Settings)``: adding a setting is one field
declaration, and the env read, the CLI flag, and the parsing all follow.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# ===========================================================================
# String helpers - env-template resolution and type-driven parsing.
# ===========================================================================


_ENV_TEMPLATE_VAR = re.compile(r"\{\{(\w+)\}\}")

_FALSE_WORDS = frozenset({"", "0", "false", "no", "off"})


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() not in _FALSE_WORDS


def _resolve_env_entry(entry: str) -> str | None:
    """Resolve an env-source entry to a string value, or ``None``.

    Entries containing ``{{VAR}}`` substitute each ``VAR`` from
    ``os.environ``; if any referenced variable is unset or empty the
    entry is unresolved and the caller tries the next one. Entries
    without ``{{}}`` are literals and resolve to themselves.

    Only called from :meth:`Settings.from_env`. CLI-supplied values never
    pass through here.
    """
    if "{{" not in entry:
        return entry
    parts: list[str] = []
    last = 0
    for m in _ENV_TEMPLATE_VAR.finditer(entry):
        parts.append(entry[last:m.start()])
        val = os.environ.get(m.group(1))
        if not val:
            return None
        parts.append(val)
        last = m.end()
    parts.append(entry[last:])
    return "".join(parts)


def _parser_for(annotation: Any) -> Callable[[str], Any]:
    """Return a callable that parses a string into the field's type.

    Order matters: ``bool`` before ``Path`` before numerics, and the
    fallthrough is ``str`` (weight-override fields are variant tokens or
    paths, so they stay strings). Path parsing runs ``expanduser`` so
    ``~`` resolves; no ``{{VAR}}`` substitution happens here.
    """
    s = annotation if isinstance(annotation, str) else str(annotation)
    if "bool" in s:
        return _parse_bool
    if "Path" in s:
        return lambda raw: Path(raw).expanduser()
    if "int" in s:
        return int
    if "float" in s:
        return float
    return str


# ===========================================================================
# Settings - the dataclass, its fields, and its constructors.
# ===========================================================================


@dataclass(frozen=True)
class Settings:
    # ---- UX ---------------------------------------------------------------

    # Keep native VideoToolbox/AVFoundation logs on stderr (historically any
    # non-empty KINOVSR_VERBOSE enabled this; see kinovsr/native/vsr.py).
    verbose: bool = field(default=False, metadata={"env": "{{KINOVSR_VERBOSE}}"})

    quiet: bool = field(default=False, metadata={"env": "{{KINOVSR_QUIET}}"})

    # ---- Scratch / performance ---------------------------------------------

    # Durable scratch root for eval sweeps, fixtures, and diagnostics.
    shared_temp_dir: Path = field(
        default_factory=lambda: Path("/tmp"),
        metadata={"env": ["{{SHARED_TEMP_DIR}}", "{{TMPDIR}}", "/tmp"]},
    )

    # Cap MLX's allocator cache (GiB). ``None`` keeps the MLX default.
    mlx_cache_limit_gb: float | None = field(
        default=None,
        metadata={"env": "{{KINOVSR_MLX_CACHE_LIMIT_GB}}"},
    )

    # Where converted CoreML models are cached (one set per SpyNet geometry).
    cache_dir: str = field(
        default="~/Library/Caches/KinoVSR",
        metadata={"env": ["{{KINOVSR_CACHE_DIR}}", "{{XDG_CACHE_HOME}}/KinoVSR",
                          "~/Library/Caches/KinoVSR"]},
    )

    # Which SpyNet implementation runs: "auto" (Neural Engine convolutions
    # with an automatic fall back to MLX), "ane" (require the ANE path), or
    # "mlx" (pure MLX only).
    spynet_backend: str = field(
        default="auto", metadata={"env": "{{SPYNET_BACKEND}}"})

    # Which BSVD implementation runs: "mlx" (the reference GPU path,
    # default - it is the faster standalone) or "ane" (one Core ML dispatch
    # per step on the Neural Engine; explicitly opt-in, fails loudly when
    # unavailable). The ANE path pays off in chains where other stages need
    # the GPU it vacates.
    bsvd_backend: str = field(
        default="mlx", metadata={"env": "{{BSVD_BACKEND}}"})

    # Which implementation serves large BSVD ANE graphs: "auto" selects the
    # direct private route where the Core ML graph would need its fp32 island,
    # "off" keeps Core ML, "require" refuses fallback when direct dispatch is
    # unavailable, and "force" enables direct dispatch at every geometry for
    # probes and testing.
    bsvd_direct: str = field(
        default="auto", metadata={"env": "{{KINOVSR_BSVD_DIRECT}}"})

    # ---- Legacy per-family weight overrides ---------------------------------
    #
    # Variant token or filesystem path; ``None`` means "use the family
    # default". These are compatibility fallbacks consumed where a stage is
    # constructed (`explicit argument or settings field`); new configuration
    # should prefer stage tables and profiles over environment variables.

    basicvsrpp_weights: str | None = field(
        default=None, metadata={"env": "{{BASICVSRPP_WEIGHTS}}"})
    basicvsrpp_restore_weights: str | None = field(
        default=None, metadata={"env": "{{BASICVSRPP_RESTORE_WEIGHTS}}"})
    bsvd_weights: str | None = field(
        default=None, metadata={"env": "{{BSVD_WEIGHTS}}"})
    esc_weights: str | None = field(
        default=None, metadata={"env": "{{ESC_WEIGHTS}}"})
    # Canonical family name is fastdvdnet (planning 07-cli-vocabulary.md);
    # the env var keeps its legacy compatibility spelling.
    fastdvdnet_weights: str | None = field(
        default=None, metadata={"env": "{{FASTDVD_WEIGHTS}}"})
    fbcnn_weights: str | None = field(
        default=None, metadata={"env": "{{FBCNN_WEIGHTS}}"})
    nafnet_weights: str | None = field(
        default=None, metadata={"env": "{{NAFNET_WEIGHTS}}"})
    pvdd_weights: str | None = field(
        default=None, metadata={"env": "{{PVDD_WEIGHTS}}"})
    realbasicvsr_weights: str | None = field(
        default=None, metadata={"env": "{{REALBASICVSR_WEIGHTS}}"})
    realesrgan_weights: str | None = field(
        default=None, metadata={"env": "{{REALESRGAN_WEIGHTS}}"})
    realplksr_weights: str | None = field(
        default=None, metadata={"env": "{{REALPLKSR_WEIGHTS}}"})
    realviformer_weights: str | None = field(
        default=None, metadata={"env": "{{REALVIFORMER_WEIGHTS}}"})
    safmn_weights: str | None = field(
        default=None, metadata={"env": "{{SAFMN_WEIGHTS}}"})
    spynet_weights: str | None = field(
        default=None, metadata={"env": "{{SPYNET_WEIGHTS}}"})
    stdf_weights: str | None = field(
        default=None, metadata={"env": "{{STDF_WEIGHTS}}"})
    toflow_weights: str | None = field(
        default=None, metadata={"env": "{{TOFLOW_WEIGHTS}}"})
    toflow_graph: str | None = field(
        default=None, metadata={"env": "{{TOFLOW_GRAPH}}"})
    toflow_sr_weights: str | None = field(
        default=None, metadata={"env": "{{TOFLOW_SR_WEIGHTS}}"})
    toflow_sr_graph: str | None = field(
        default=None, metadata={"env": "{{TOFLOW_SR_GRAPH}}"})

    # --- Constructors -------------------------------------------------------

    @classmethod
    def from_env(cls) -> Settings:
        """Build a ``Settings`` from environment variables.

        For each field whose ``metadata["env"]`` lists one or more entries:
        try each via :func:`_resolve_env_entry` in order, parse the first
        resolved value via the field's type annotation, and pass it to
        ``cls``. Fields with no resolved entry keep their defaults.
        """
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            env_spec = f.metadata.get("env")
            if env_spec is None:
                continue
            env_list = [env_spec] if isinstance(env_spec, str) else env_spec
            for entry in env_list:
                resolved = _resolve_env_entry(entry)
                if resolved is None:
                    continue
                kwargs[f.name] = _parser_for(f.type)(resolved)
                break
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a new Settings with non-None overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **clean)


# ===========================================================================
# Process-default accessor - the bridge for legacy call sites.
#
# Stage construction paths that predate the config layer read their env
# fallbacks through here instead of os.environ, so the environment surface
# stays declared in one file. New code should thread an explicit Settings
# instead of reaching for the default.
# ===========================================================================


_DEFAULT: Settings | None = None


def default_settings() -> Settings:
    """Return the process-wide ``Settings.from_env()``, built once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Settings.from_env()
    return _DEFAULT


def set_default_settings(settings: Settings) -> None:
    """Publish a fully resolved ``Settings`` as the process-wide default.

    An invocation resolves settings from the environment, the config table,
    and the command line in that order; code that cannot be handed the
    instance directly (deep inside a model implementation, below the layers
    that thread it) reads it through :func:`default_settings`, which would
    otherwise see environment values only and silently ignore a flag.
    """
    global _DEFAULT
    _DEFAULT = settings


def _reset_default_settings() -> None:
    """Drop the cached default so tests can re-read a patched environment."""
    global _DEFAULT
    _DEFAULT = None


# ===========================================================================
# CLI bridge - auto-generate --kebab-case flags from Settings fields.
# ===========================================================================


def add_argparse_args(parser: argparse.ArgumentParser,
                      skip: frozenset[str] | set[str] = frozenset()) -> None:
    """Add a ``--kebab-case`` flag for every Settings field.

    Flags default to ``None`` (unset). After parsing, apply them with
    :func:`settings_from_args` on top of a base ``Settings`` (typically
    ``Settings.from_env()``). Booleans get a ``--flag`` / ``--no-flag``
    pair so they can be turned off explicitly. CLI values are verbatim:
    no ``{{VAR}}`` substitution.

    ``skip`` names fields whose flag another parser group already owns
    (the CLI registry defines richer per-family weight flags whose dest
    IS the Settings field); adding them twice would be an argparse
    conflict.
    """
    group = parser.add_argument_group("settings overrides (override env vars)")
    for f in fields(Settings):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        annotation = f.type if isinstance(f.type, str) else str(f.type)
        if "bool" in annotation:
            group.add_argument(flag, dest=f.name, action="store_true", default=None)
            group.add_argument(
                f"--no-{f.name.replace('_', '-')}",
                dest=f.name, action="store_false", default=None)
        else:
            group.add_argument(flag, dest=f.name, type=_parser_for(f.type), default=None)


def settings_from_args(args: argparse.Namespace, base: Settings) -> Settings:
    """Apply argparse overrides on top of a base Settings."""
    return base.with_overrides(
        **{f.name: getattr(args, f.name, None) for f in fields(Settings)})
