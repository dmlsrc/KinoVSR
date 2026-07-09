# Attribution

`kinovsr/safmn/` is an independent MLX reimplementation of the SAFMN
super-resolution family, written from the published architectures as a spec -- no
upstream code is bundled. The model weights are the upstream project's,
redistributed under its license (see `weights/README.md` for the downloads).

## SAFMN

Long Sun, Jiangxin Dong, Jinshan Pan et al. -- "Spatially-Adaptive Feature
Modulation for Efficient Image Super-Resolution" (ICCV 2023).
https://github.com/sunny2109/SAFMN

The two ported variants are the project's challenge winners: `light_SAFMN++`
(1st place, fidelity track, AIS 2024 Real-Time 4K Super-Resolution of compressed
AVIF images) and `Real_SAFMN++` / SAFMN-L (1st place, AIM 2025 Efficient
Perceptual Super-Resolution).

Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0). Copyright the
SAFMN authors. You may not use these files except in compliance with the License;
the full text is at https://www.apache.org/licenses/LICENSE-2.0 and in the
upstream repository's `LICENSE`. Unless required by applicable law or agreed to
in writing, the software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND.

## PureScale 2.0 (the `purescale*` variant tokens)

The `purescale`, `purescale2x`, and `purescale2x-sharp` tokens load checkpoints
by **limitlesslab** from the AI-upscaling-models project, release "PureScale 2.0":
https://github.com/limitlesslab/AI-upscaling-models
https://github.com/limitlesslab/AI-upscaling-models/releases/tag/PureScale2

These are SAFMN-L models retrained from scratch on the author's own curated
real-world dataset with a modified SAFM branch (adaptive average pooling and
bicubic upsampling in place of max pooling and nearest) that eliminates the
stock architecture's known transient block-lattice artifact. This port
implements that modified branch as its own MLX code, selected per variant; no
code from the release is bundled.

**License caution -- these weights are NOT Apache-2.0.** The PureScale release
is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

- **NonCommercial: the weights may only be used for non-commercial purposes.**
  If you use this software commercially, do not use the `purescale*` variants.
- Attribution: credit limitlesslab / AI-upscaling-models when sharing results
  produced with these weights.
- ShareAlike: if you adapt and share the weights themselves, the adaptation
  must carry the same license.

The weights are not distributed with this repository; users download them from
the author's release and convert locally (see `weights/README.md`).
