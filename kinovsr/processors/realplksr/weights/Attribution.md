# Attribution

[`kinovsr/processors/realplksr/`](../) is an independent MLX reimplementation of RealPLKSR,
written from the spandrel reference architecture as a spec -- no upstream code is
bundled. The model weights are the upstream project's, redistributed under their
licenses (see [README.md](README.md) beside this file for the downloads).

## PLKSR / RealPLKSR

Dongheon Lee, Yongjun Cho et al. -- "Partial Large Kernel CNNs for Efficient
Super-Resolution" (IEEE Access; arXiv 2404.11848).
<https://github.com/dslisleedh/PLKSR>

RealPLKSR is the GAN-stable rework (GroupNorm/LayerNorm, Mish, dropout, optional
DySample head) released by the neosr author on the PLKSR repository (issue #4,
"Making PLKSR stable for real-world SISR") and endorsed by the paper's author. The
reference implementation used as the porting spec is spandrel's
(`chaiNNer-org/spandrel`, `architectures/PLKSR/__arch/RealPLKSR.py` and the
`DySample` helper). PLKSR / RealPLKSR / spandrel are all MIT-licensed.

DySample: Wenze Liu et al. -- "Learning to Upsample by Learning to Sample"
(ICCV 2023; arXiv 2308.15085). <https://github.com/tiny-smart/dysample>

## Weights (the variant tokens)

Trained and released by Philip Hofmann (Phhofm), <https://github.com/Phhofm/models>:

- `public2x`, `public2x-nn` (2xPublic_realplksr_dysample_layernorm_real[_nn]):
  Apache License 2.0 (SPDX: Apache-2.0).
- `nomos4x` (4xNomosWebPhoto_RealPLKSR): Creative Commons Attribution 4.0
  International (SPDX: CC-BY-4.0). Attribution: Philip Hofmann.

The weights are downloaded, not bundled (see [README.md](README.md) beside this file). Both licenses
permit commercial use with attribution; retain this file and the upstream credit.
