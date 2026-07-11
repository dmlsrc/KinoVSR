# SpyNet weights

The stock SpyNet optical-flow weights consumed by the shared MLX SpyNet
implementation in `kinovsr/vsr_blocks.py` (`spynet_flow`). See
[`../ATTRIBUTION.md`](../ATTRIBUTION.md) for the port provenance and license
posture.

Bundled file:

| file | source `.pth` | download | sha256 (source `.pth`) |
| --- | --- | --- | --- |
| `spynet_stock_20210409.safetensors` | `spynet_20210409-c6c1bd09.pth` | https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth | `c6c1bd09b52d05ba17f3e701f549d6faf5e314aabce8ae462c1c171a8d6c4914` |

License: Apache-2.0, as distributed by BasicSR (XPixelGroup). The original
SpyNet models were released by the authors for research use; see the SpyNet
repository (https://github.com/anuragranj/spynet) for their terms.

The safetensors is the converted form of the source `.pth`: NHWC convolution
layout, with the ImageNet normalization constants embedded as `spynet.mean` and
`spynet.std` so the runtime needs no external preprocessing constants. These are
the stock flow-estimation weights, deliberately not a SpyNet extracted from a
super-resolution checkpoint (end-to-end SR training repurposes SpyNet into a
feature matcher and degrades its accuracy as a motion measurement).
