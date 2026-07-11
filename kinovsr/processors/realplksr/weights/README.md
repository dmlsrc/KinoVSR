# RealPLKSR weights

Not bundled -- download; the `.safetensors` are gitignored. These are
`--realplksr-weights <token>` tokens (default `public2x`). The scale (2x/4x),
width, block count, LayerNorm/GroupNorm variant, and DySample vs PixelShuffle head
are all auto-detected from the checkpoint at load.

All three are trained by Philip Hofmann (Phhofm) and published as `.safetensors`
directly on his releases, so no `.pth` conversion is needed -- download the
`.safetensors` asset and rename it to the token filename below.

| token | scale | variant | download | license |
| --- | --- | --- | --- | --- |
| public2x (default) | 2x | LayerNorm + DySample, real-world photo/JPEG | https://github.com/Phhofm/models/releases/download/2xPublic_realplksr_dysample_layernorm_real/2xPublic_realplksr_dysample_layernorm_real.safetensors | Apache-2.0 |
| public2x-nn | 2x | same, trained without noise (cleaner sources) | https://github.com/Phhofm/models/releases/download/2xPublic_realplksr_dysample_layernorm_real_nn/2xPublic_realplksr_dysample_layernorm_real_nn.safetensors | Apache-2.0 |
| nomos4x | 4x | GroupNorm + PixelShuffle, photo restoration | https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.safetensors | CC-BY-4.0 |

Download -- the output name must match `net.py`'s `_VARIANTS`:

```bash
curl -L -o kinovsr/processors/realplksr/weights/2xpublic_realplksr_dysample_layernorm_real.safetensors \
  https://github.com/Phhofm/models/releases/download/2xPublic_realplksr_dysample_layernorm_real/2xPublic_realplksr_dysample_layernorm_real.safetensors
```

`nomos4x` -> `4xnomoswebphoto_realplksr.safetensors`, `public2x-nn` ->
`2xpublic_realplksr_dysample_layernorm_real_nn.safetensors`, likewise.

If you point `--realplksr-weights` at another RealPLKSR checkpoint that ships only
as `.pth` (a plain BasicSR/neosr/traiNNer state dict, not nested under `params`),
convert it without a param-key unwrap:

```bash
kinovsr weights convert model.pth -o weights/model.safetensors
```

## Precision

The `public2x` LayerNorm checkpoints are fp16-safe. The `nomos4x` GroupNorm
checkpoint is flagged fp16-unsafe upstream (spandrel `supports_half=False`): the
partial-large-kernel activations reach ~1150, so a naive fp16 GroupNorm variance
(`mean(x^2)` ~ 1.3e6) overflows fp16 and silently zeros the hot channels. The MLX
port runs just the GroupNorm/LayerNorm reductions in fp32 (the `--realplksr-dtype
float16` default), which fixes exactly that -- measured visually lossless (~72 dB
vs fp32) while keeping the convs and activations in fp16. Use `--realplksr-dtype
float32` to force a full single-precision run.

## Architecture

RealPLKSR = the GAN-stable rework of PLKSR (Lee et al., arXiv 2404.11848). The
stability changes (GroupNorm/LayerNorm, Mish, dropout) were released by the neosr
author on the PLKSR repo issue #4 and endorsed by the paper author. Reimplemented
in MLX from the spandrel reference architecture as a spec; no upstream code is
bundled.

Source: https://github.com/Phhofm/models , architecture
https://github.com/chaiNNer-org/spandrel (PLKSR/RealPLKSR).
