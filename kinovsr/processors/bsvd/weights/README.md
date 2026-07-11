# BSVD Weights

Not bundled -- download + convert; the `.safetensors` are gitignored. The
original upstream SharePoint checkpoint link is currently dead, so these hashes
document the local mirror copies used while the upstream issue is pending.

| token | source `.pth` | sha256 |
| --- | --- | --- |
| c64 | https://raw.githubusercontent.com/gmlwns2000/sharkshark-4k/master/src/upscale/model/bsvd/bsvd-64.pth | `1d8f7252765cf8f826d03f23c4a4b51413c51b649f0aed6994f149cbe5d6ad8f` |
| c32 | https://raw.githubusercontent.com/gmlwns2000/sharkshark-4k/master/src/upscale/model/bsvd/bsvd-32.pth | `e91af9682e2c3f71684559dd95defd243cae811375db238a71caf42850418c14` |

Download the source `.pth` files to a scratch directory, verify the hashes, then
convert them:

```bash
mkdir -p weights-src
curl -L -o weights-src/bsvd-64.pth \
  https://raw.githubusercontent.com/gmlwns2000/sharkshark-4k/master/src/upscale/model/bsvd/bsvd-64.pth
shasum -a 256 weights-src/bsvd-64.pth
kinovsr weights convert \
  weights-src/bsvd-64.pth \
  --param-key params \
  -o kinovsr/processors/bsvd/weights/bsvd_64.safetensors

curl -L -o weights-src/bsvd-32.pth \
  https://raw.githubusercontent.com/gmlwns2000/sharkshark-4k/master/src/upscale/model/bsvd/bsvd-32.pth
shasum -a 256 weights-src/bsvd-32.pth
kinovsr weights convert \
  weights-src/bsvd-32.pth \
  --param-key params \
  -o kinovsr/processors/bsvd/weights/bsvd_32.safetensors
```

The c64 checkpoint matches the public unblind test config shape:
`chns=[64,128,256]`, `mid_ch=64`, `interm_ch=64`, `act=relu6`, `norm=none`,
and first conv input channels = 4 (RGB + sigma map). The c32 mirror checkpoint
uses the same unblind layout at half width (`chns=[32,64,128]`,
`mid_ch=32`, first conv input channels = 4), but its provenance is weaker than
c64 because the upstream README only advertises the c64 checkpoint.
