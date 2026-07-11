"""Subcommand implementations for the installed kinovsr entry point.

Each module exposes run_<name>_command(argv) -> int; kinovsr.cli.main
routes the first positional token here. The flat processing CLI stays
the default invocation shape until step 6 moves it onto the typed
pipeline.
"""
