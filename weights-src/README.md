# Upstream source checkpoints

The collected, hash-verified upstream files every shipped `.safetensors`
in this repo was converted from. One subfolder per weights owner, files
under their upstream names. Everything here except this README is
untracked (multi-GB); this file exists so git keeps the folder and so
the layout is documented.

`kinovsr weights convert` looks here first: an input that does not
exist as given is resolved as `weights-src/<path>` and then as a unique
`weights-src/**/<name>` match, so the documented conversion commands
work with bare filenames from the repo root:

```bash
kinovsr weights convert fbcnn_color.pth \
  -o kinovsr/processors/fbcnn/weights/fbcnn_color.safetensors --strip-prefix ''
```

State `-o` when converting from here; the converter refuses to drop its
default output into the collection.

## Contents

Per-checkpoint provenance (source URL, sha256, license) lives in each
owner's `manifest.toml`; conversion commands live in each owner's
`weights/README.md`. Every file placed here was sha256-verified against
the manifest's recorded upstream hash at collection time.

| folder | files | conversion |
| --- | --- | --- |
| basicvsrpp/ | 10 mmediting .pth (4 SR, 6 restore) | `--strip-prefix 'generator.'` |
| bsvd/ | bsvd-64.pth, bsvd-32.pth | `--param-key params` per its README |
| dover/ | DOVER-Mobile.pth | [`kinovsr/eval/models/dover/convert_dover.py`](../kinovsr/eval/models/dover/convert_dover.py) |
| esc/ | ESC_Real_X4_{GAN,MSE}.pth | `--param-key params_ema` |
| eval/ | face_detection_yunet_2023mar.onnx | none (redistributed unmodified) |
| fastdvdnet/ | model.pth, model_clipped_noise.pth | default (`--strip-prefix 'module.'`) |
| fbcnn/ | fbcnn_color/gray/gray_double.pth | `--strip-prefix ''` |
| musiq/ | musiq_koniq_ckpt-e95806b9.pth | [`kinovsr/eval/models/musiq/convert_musiq.py`](../kinovsr/eval/models/musiq/convert_musiq.py) |
| nafnet/ | 5 NAFNet-*.pth | `--strip-prefix ''` |
| pvdd/ | 6 .pth (renamed from `<subdir>/g_net_100000.pth`) | no flags |
| realbasicvsr/ | the mmediting c64b20 .pth | `--only-prefix generator_ema. --strip-prefix generator_ema.` |
| realesrgan/ | 10 .pth (9 profiles + wdn companion) | `--strip-prefix ''` |
| realplksr/ | 3 upstream .safetensors | none (shipped files are byte-identical renames) |
| realviformer/ | weights.pth | `--param-key params` |
| safmn/ | 3 stock + 3 PureScale .pth | `--param-key params` (stock) / `params_ema` (PureScale) |
| spynet/ | spynet_20210409-c6c1bd09.pth | [`kinovsr/modeling/spynet/convert_spynet.py`](../kinovsr/modeling/spynet/convert_spynet.py) |
| stdf/ | exp/<config>/ckp_*.pt (exp.zip members) | default (`--strip-prefix 'module.'`) |
| toflow/ | denoise/deblock/sr/interp.t7 | [`kinovsr/processors/toflow/convert_t7_to_safetensors.py`](../kinovsr/processors/toflow/convert_t7_to_safetensors.py) |

Not here:
[`kinovsr/eval/weights/niqe_pristine_reds.safetensors`](../kinovsr/eval/weights/niqe_pristine_reds.safetensors)
is a first-party fitted artifact with no upstream source (see the
[eval weights README](../kinovsr/eval/weights/README.md)).

Conversion parity is re-checkable end to end:

```bash
python scripts/dev/verify_source_conversions.py
```

It converts every source present here with the documented recipe and
compares the result tensor-for-tensor against the shipped artifact.
