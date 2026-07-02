# SAFMN weights

Not bundled -- download + convert; the `.safetensors` are gitignored. These are
`--safmn-weights <token>` tokens (default `light`). sha256 is of the source `.pth`:

| token | download | sha256 |
| --- | --- | --- |
| light (light_SAFMN++, fidelity 4x, trained on compressed content) | https://github.com/sunny2109/SAFMN/raw/main/AIS2024-RTSR/pretrained_model/light_safmnpp.pth | `a542c92072cb25adab1f9cc5209d4f4f4ca8549db084e6703d2e032357cd50a7` |
| real (SAFMN_L_Real_LSDIR, real-world perceptual 4x, Real-ESRGAN degradation) | https://huggingface.co/Meloo/SAFMN/resolve/main/SAFMN_L_Real_LSDIR_x4.pth | `f1ac0ee3ee143fbbc49aff6584cb7b48b71ba8d1961dfb8ad076b98ca4799ae2` |
| real2x (same family, 2x output -- the HD -> 4K class tool) | https://huggingface.co/Meloo/SAFMN/resolve/main/SAFMN_L_Real_LSDIR_x2.pth | `a0d838b197dcaedb58cb317bcc3bd2704cd4733083e1eb10d7eac56eb2ea6820` |

Convert -- the output name must match `net.py`'s `_VARIANTS`:

```bash
curl -L -O https://huggingface.co/Meloo/SAFMN/resolve/main/SAFMN_L_Real_LSDIR_x4.pth
python scripts/pth_to_safetensors.py SAFMN_L_Real_LSDIR_x4.pth --param-key params \
  -o LTX_2_MLX/videotoolbox/safmn/weights/safmn_l_real_lsdir_x4.safetensors
```

(`light_safmnpp.pth` -> `light_safmnpp.safetensors` the same way.) `--param-key
params` matters: these checkpoints carry BOTH `params` and `params_ema` and they
differ; every SAFMN reference loader for these models (`app.py`,
`inference/inference_real_safmn.py`, the AIS2024 script) loads `['params']`.
Variant, width, block count, and scale are auto-detected from the checkpoint at
load, so other SAFMN checkpoints (e.g. the DF2K x2/x3/x4 bicubic-fidelity models
at https://huggingface.co/Meloo/SAFMN) also load when passed as a path -- though
the bicubic-trained ones are benchmark models, not suited to real video. The
GitHub v0.1.0 release's `SAFMN_L_Real_LSDIR_x4-v2.pth` is byte-identical to the
HF file above (same sha256) -- there is no newer model.

Do NOT use the AIM 2025 challenge checkpoint (`SAFMN-L.pth` from the challenge folder of
the GitHub repo, also named "SAFMN-L"): it was tuned on synthetically degraded stills for
no-reference perceptual metrics and hallucinates crusty texture over motion blur on real
video.

Source: https://github.com/sunny2109/SAFMN , https://huggingface.co/Meloo/SAFMN
