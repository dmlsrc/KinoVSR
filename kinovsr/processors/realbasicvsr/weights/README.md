# RealBasicVSR weights

[`realbasicvsr_x4.safetensors`](realbasicvsr_x4.safetensors) (~24MB) IS bundled in the repo. Provenance / re-download --
direct download and the sha256 of the source `.pth`:

| variant | download | sha256 |
| --- | --- | --- |
| realbasicvsr_x4 | <https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth> | `52f77c2c835aaa3fe675b3959b2f85010a6c6f63f77f7e279394646e55a4e376` |

Convert:

```bash
curl -L -O https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth
kinovsr weights convert realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth \
  -o kinovsr/processors/realbasicvsr/weights/realbasicvsr_x4.safetensors \
  --only-prefix 'generator_ema.' --strip-prefix 'generator_ema.'
```

The mmediting checkpoint nests under `state_dict` and carries BOTH
`generator.` and `generator_ema.` copies of every parameter with the same
names but different values; the bundled artifact is the EMA set (verified
tensor-for-tensor against the source). Converting with
`--strip-prefix 'generator.'` would silently produce the non-EMA model.

Source: Chan et al., "Investigating Tradeoffs in Real-World Video Super-Resolution"
(CVPR 2022) -- <https://github.com/ckkelvinchan/RealBasicVSR>
