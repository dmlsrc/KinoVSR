# Configuring runs: flags, TOML, --set, --print-config

There are two ways to author a run - CLI flags, or a TOML config with a
`pipeline` list - and one resolution order behind both. Flags are not a
second config system: they are a layer merged onto the same resolved
config a TOML file expresses. `kinovsr run --print-config` shows the
result of that merge for any invocation, as TOML, and its output is a
complete reproducible run.

## The resolution order

```text
settings:   dataclass defaults < environment < --base-config < --config < settings flags < --set
stage:      family/profile defaults < merged stage table < flag dials < --set
run-level:  defaults < --base-config < --config < run flags < --set
```

Family and profile defaults live inside each family's config parser -
they are applied when the stage is parsed, not written into your config.
Everything after them is concrete data you can see with `--print-config`.

## Two ownership rules

**The stage list has exactly one owner.** If a config declares
`pipeline`, flags may not create, remove, or reorder stages. Every
compositional flag - the stage selectors (`--denoise`, `--deblock`,
`--restore`, `--upscale`, `--nafnet`, `--level`, `--deflicker`), the
geometry selectors (`--crop-*`, `--sanitize-edges`, `--square-pixels`),
`--target-fps`, `--conform-cfr`, `--preprocess-order`, and
`--denoise-first` - is refused with an error naming the conflict.
Everything else still applies.

**Dials merge onto stages by target, not by path.** A family dial such
as `--bsvd-strength` applies to every stage whose processor is `bsvd`,
whether the chain came from flags or TOML. A chain dial such as
`--denoise-strength` distributes positionally across the stages filling
that capability. Targeting respects capability: a dial declared for a
family's upscale capability does not land on its restore stages. A set
dial that matches no stage is an error, never a silent no-op.

## --set: the instance-addressed escape hatch

```bash
kinovsr run --config run.toml --set denoise_bsvd.strength=0.35
```

`--set stage.key=value` targets one named stage where a family dial
targets every matching stage. Values are typed exactly like TOML
(`0.35` is a float, `true` a boolean, `[1,2]` an array, quoted strings
stay strings), applied after everything else, and never expand
environment templates. Repeated `--set` flags apply in order.

Structural problems - an unknown stage name, malformed syntax, setting
through a scalar - fail immediately. A key the family itself does not
accept fails when the run opens, with an error listing the family's
accepted keys. That boundary is deliberate: family key knowledge lives
in exactly one place, the family parsers, so `--print-config` output is
guaranteed structurally loadable but not guaranteed family-key valid.

## --print-config: resolve, print, exit

```bash
kinovsr run --video in.mp4 --output-dir out --denoise bsvd --print-config > run.toml
kinovsr run --config run.toml
```

The printed TOML round-trips: feeding it back reproduces the identical
resolved config with no additional flags, because run-level options are
part of the config surface. It is also how you discover `--set` targets
- stage names like `denoise_bsvd` are visible in the output, not
guessed. Printing probes the source (stage assembly depends on the
probed geometry), so `--video` must point at a readable file.

## Run-level tables

The 27 run-level options map onto three tables:

```toml
[input]
video = "in.mp4"
start = 0
gop_align = true

[output]
output_dir = "out"
audio = true
encode_quality = 0.65

[diagnostics]
noise_map_debug = false
```

## A complete config

```toml
pipeline = ["denoise_bsvd", "upscale"]

[input]
video = "in.mp4"
gop_align = true

[output]
output_dir = "out"

[denoise_bsvd]
processor = "bsvd"
capability = "denoise"
strength = 0.3

[upscale]
processor = "videotoolbox"
capability = "upscale"
profile = "balanced"
```

The equivalent flag invocation - and the way to generate the file above
without writing it by hand:

```bash
kinovsr run --video in.mp4 --output-dir out --gop-align --denoise bsvd --bsvd-strength 0.3 --upscale balanced --print-config
```

## Rules worth knowing

- Unknown top-level tables are rejected; an unlisted stage table must
  declare `processor = "<family>"`.
- A flag passed at its registry default is a no-op by design - it must
  not clobber a TOML value. `--set` is the explicit way to force a
  default over a config.
- `{{VAR}}` template expansion applies only to declared environment
  sources in settings; TOML values, CLI values, and `--set` values are
  always literal.
- Stage settings and shared dial names use one vocabulary: the same key
  (`strength`, `weights`, `profile`, `window`, `trim`, `flow`, ...)
  means the same concept in every family. `docs/PROCESSORS.md` lists
  every family, profile, and weight artifact.
