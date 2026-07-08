# Attribution

`videotoolbox/musiq/` is an independent MLX reimplementation of MUSIQ,
written from the published paper and from reading the IQA-PyTorch
reference implementation as a specification -- no upstream code is bundled
or executed.

## MUSIQ

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang -- "MUSIQ:
Multi-scale Image Quality Transformer" (ICCV 2021).
https://github.com/google-research/google-research/tree/master/musiq
Original code and released model checkpoints: Apache License 2.0
(SPDX: Apache-2.0).

## Checkpoint lineage

The converted weights derive from the Google Research release via the
PyTorch conversion distributed by IQA-PyTorch (Chaofeng Chen,
https://github.com/chaofengc/IQA-PyTorch, musiq_koniq_ckpt-e95806b9.pth).
The IQA-PyTorch codebase is under the NTU S-Lab License 1.0; it was used
here only as a reading reference for numerical semantics. The weight
values themselves are the Google Apache-2.0 release, reshaped.
