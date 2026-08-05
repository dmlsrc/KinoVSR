# Attribution

[`kinovsr/eval/models/dover/`](../) is an independent MLX reimplementation of
DOVER-Mobile, written from the published paper and from reading the
official implementation as a specification -- no upstream code is
bundled or executed.

## DOVER

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan
Wang, Wenxiu Sun, Qiong Yan, Weisi Lin -- "Exploring Video Quality
Assessment on User Generated Contents from Aesthetic and Technical
Perspectives" (ICCV 2023).
<https://github.com/VQAssessment/DOVER>

The reference implementation and the released model checkpoints are
under the NTU S-Lab License 1.0, which permits redistribution and use
for **non-commercial purposes only**; commercial use requires prior
permission from the authors. That restriction carries over to the
converted weights (`weights/dover_mobile.safetensors`), which are a
reshaped copy of the released DOVER-Mobile checkpoint. This directory's
reimplemented *code* is original to this project, but running it with
those weights is bound by the S-Lab terms.

## Backbone lineage

DOVER-Mobile's two branches are ConvNeXt-V2 femto backbones (Woo et
al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
Autoencoders", CVPR 2023) inflated to 3D, fine-tuned by the DOVER
authors on the DIVIDE-3k / LSVQ video-quality corpora.
