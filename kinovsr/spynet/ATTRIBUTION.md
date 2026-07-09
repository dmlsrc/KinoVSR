# Attribution

`kinovsr/spynet/` holds the STOCK SpyNet optical-flow weights consumed by
the shared MLX SpyNet implementation in `kinovsr/vsr_blocks.py`
(`spynet_flow`). The implementation is an independent MLX port written from
the published architecture and the BasicSR reference as a spec; the weights
are the upstream project's, redistributed under its license, converted to
safetensors (NHWC conv layout, ImageNet normalization constants embedded as
`spynet.mean` / `spynet.std`).

These are the stock flow-estimation weights, deliberately NOT a SpyNet
extracted from a super-resolution checkpoint: end-to-end SR training
repurposes SpyNet into a feature matcher and its accuracy as a motion
measurement collapses.

## SpyNet

Anurag Ranjan, Michael J. Black -- "Optical Flow Estimation using a Spatial
Pyramid Network" (CVPR 2017). https://github.com/anuragranj/spynet

## Weights

`weights/spynet_stock_20210409.safetensors` is converted from
`spynet_20210409-c6c1bd09.pth` as distributed by BasicSR (XPixelGroup),
https://github.com/XPixelGroup/BasicSR, licensed under the Apache License
2.0 (SPDX: Apache-2.0). The original SpyNet models were released by the
authors for research purposes; see the SpyNet repository for their terms.
