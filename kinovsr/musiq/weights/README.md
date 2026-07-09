# MUSIQ Weights

Not bundled -- download + convert; the `.safetensors` is gitignored. The
source checkpoint filename embeds the first 8 hex of its sha256 (torch-hub
convention), which matches the hash below.

| file | source | sha256 | license |
| --- | --- | --- | --- |
| musiq_koniq_ckpt-e95806b9.pth | https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_koniq_ckpt-e95806b9.pth | `e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa` | Apache-2.0 (Google Research MUSIQ lineage; see ../ATTRIBUTION.md) |

Converted output (for reference after a correct conversion):

| file | sha256 |
| --- | --- |
| musiq_koniq.safetensors | `e86f16e456c1ecc119bee6d44b88e7444b21d65629454937cffb9e8f07ccb7e0` |

Download, verify, convert:

```bash
mkdir -p weights-src
curl -L -o weights-src/musiq_koniq_ckpt-e95806b9.pth \
  https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_koniq_ckpt-e95806b9.pth
shasum -a 256 weights-src/musiq_koniq_ckpt-e95806b9.pth
python kinovsr/musiq/convert_musiq.py \
  weights-src/musiq_koniq_ckpt-e95806b9.pth
```

The net-specific converter is used instead of the general
`scripts/pth_to_safetensors.py` because MUSIQ's conversion is not a plain
re-serialization: the reference applies weight standardization to the conv
weights at every forward, and since weights are fixed at inference the
converter folds it in (unbiased std, eps 1e-5) and transposes the conv
layout OIHW -> OHWI for MLX. A generically-converted checkpoint would load
but score garbage.
