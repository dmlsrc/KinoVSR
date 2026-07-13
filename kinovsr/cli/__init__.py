"""KinoVSR command-line interface.

- :mod:`kinovsr.cli.options` - the option vocabulary contract (Opt, shared keys);
- :mod:`kinovsr.cli._registry` - composed foundation and family option data;
- :mod:`kinovsr.cli.args` - parser assembly and cross-option validation;
- :mod:`kinovsr.cli.config` - args + TOML into a resolved :class:`Invocation`;
- :mod:`kinovsr.cli.main` - the installed console entry point.

The console script references ``kinovsr.cli.main:main`` directly; this
package deliberately re-exports nothing so ``python -m kinovsr.cli.main``
does not double-import the module.
"""
