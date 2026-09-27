"""Subcommand implementations for the installed kinovsr entry point.

Each module exposes run_<name>_command(argv) -> int; kinovsr.cli.main
routes the first positional token here. Both the flat processing CLI and
typed configuration route through the same pipeline.
"""
