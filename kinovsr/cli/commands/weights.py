"""The ``kinovsr weights`` command: list and verify manifest-covered weights.

Read-only by design (planning: automatic downloading stays disabled until
the download-policy questions are answered). Output renders through the
shared Rich console; exit codes make ``verify`` scriptable:

- 0: everything present that must be (external absences are informational);
- 1: a bundled asset is missing or any recorded artifact hash mismatches.
"""

from __future__ import annotations

import argparse

from kinovsr.modeling.weights import (
    ManifestError,
    load_registered,
    registered_owners,
    verify_manifest,
)
from kinovsr.ui.console import get_console


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinovsr weights",
        description="List or verify manifest-covered model weights.")
    sub = parser.add_subparsers(dest="action", required=True)
    for action, description in (
            ("list", "profiles and weight assets per owner"),
            ("verify", "check presence and recorded artifact hashes")):
        action_parser = sub.add_parser(action, help=description)
        action_parser.add_argument(
            "owners", nargs="*", metavar="OWNER",
            help="family/component names (default: every registered owner)")
    return parser


def _select_owners(requested: list[str]) -> tuple[str, ...]:
    if not requested:
        return registered_owners()
    known = set(registered_owners())
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ManifestError(
            f"no manifest registered for {unknown[0]!r} "
            f"(registered: {', '.join(sorted(known))})")
    return tuple(requested)


def _list(owners: tuple[str, ...]) -> int:
    console = get_console()
    for owner in owners:
        manifest = load_registered(owner)
        console.print(f"[bold]{manifest.name}[/bold] ({manifest.kind})")
        for profile in manifest.profiles.values():
            capabilities = "/".join(profile.capabilities) or "-"
            console.print(
                f"  profile {profile.name}: {capabilities}; "
                f"weights {', '.join(profile.weights)}"
                + (f"; defaults {profile.defaults}" if profile.defaults
                   else ""))
        for asset in manifest.weights.values():
            status = "present" if asset.path.is_file() else "missing"
            console.print(
                f"  weights {asset.asset_id}: {asset.distribution}, "
                f"{status}, license {asset.license or 'unrecorded'} "
                f"({asset.path.name})")
    return 0


def _verify(owners: tuple[str, ...]) -> int:
    console = get_console()
    failures = 0
    for owner in owners:
        manifest = load_registered(owner)
        for report in verify_manifest(manifest):
            broken = ((report.distribution == "bundled"
                       and not report.present)
                      or report.hash_ok is False)
            failures += broken
            style = "bold red" if broken else None
            console.print(
                f"{report.owner}/{report.asset_id}: "
                f"{report.distribution}, "
                f"{'present' if report.present else 'missing'}; "
                f"{report.note}",
                style=style)
    console.print("verify: FAIL" if failures else "verify: ok",
                  style="bold red" if failures else "bold green")
    return 1 if failures else 0


def run_weights_command(argv: list[str]) -> int:
    if argv and argv[0] == "convert":
        # The torch-checkpoint re-serializer keeps its own parser (it
        # predates this command and its flags are documented in every
        # family weights README).
        from .weights_convert import run_convert

        return run_convert(argv[1:])
    args = _build_parser().parse_args(argv)
    try:
        owners = _select_owners(args.owners)
        if args.action == "list":
            return _list(owners)
        return _verify(owners)
    except ManifestError as exc:
        get_console().print(f"weights error: {exc}", style="bold red")
        return 2
