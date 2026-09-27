# DOVER-Mobile weights

`dover_mobile.safetensors` (39 MB) is converted from the official
DOVER-Mobile checkpoint and is **not bundled** in the repo; download
and convert once as below. The weights are under the NTU S-Lab
License 1.0 (non-commercial) -- see [Attribution](Attribution.md).

| File | Size | sha256 |
| ---- | ---- | ------ |
| `DOVER-Mobile.pth` (source) | 43 MB | `81b487be2aa4b3dd6920afa2e92294ed8fdd46a306911f75ecc8e6938a670884` |
| `dover_mobile.safetensors` (converted) | 39 MB | `9d8f65fd9f2810cb0f31897e187d2094c888c90c2c90a1e0b5fc8c4243fc2ff2` |

## Download

```bash
curl -L -o DOVER-Mobile.pth \
  https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth
shasum -a 256 DOVER-Mobile.pth   # expect 81b487be2aa4...
```

## Convert

```bash
python kinovsr/eval/models/dover/convert_dover.py DOVER-Mobile.pth
# writes dover_mobile.safetensors into this directory
shasum -a 256 kinovsr/eval/models/dover/weights/dover_mobile.safetensors
# expect 9d8f65fd9f28...
```

Conversion needs torch and is layout-specific (channels-last conv3d
kernels, per-block depthwise stacks whose temporal extent alternates 1
and 3, pre-transposed linears, dropped ImageNet classifier heads) --
the generic `kinovsr weights convert` re-serializer is not
sufficient; use [`convert_dover.py`](../convert_dover.py).
