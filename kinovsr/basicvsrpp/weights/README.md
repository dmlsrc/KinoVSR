# BasicVSR++ weights (not bundled)

Not committed (too large) -- download + convert; the `.safetensors` are gitignored.
`--spatial-mode basicvsrpp --basicvsrpp-variant <token>` (default `vimeo90k_bd`).

Direct download and the sha256 of the source `.pth`:

| variant | download | sha256 |
| --- | --- | --- |
| reds4 | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth | `db622b2fd4caae0a4c63ab5e54f1cfef7a62a0f3b8ad101aba2eae068d928549` |
| vimeo90k_bd | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth | `ab315ab1d83d09834d43c8ba17d019282a93f0cbf40bfd49be99dfcebf4c12eb` |
| vimeo90k_bi | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth | `4ef437e27e7ed468853d9bad2e7a02e50fd4582c986a5cb4231054b0021b2e81` |
| ntire_vsr | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth | `1ff352921112b84cacc79b23df502a2319d5714d6b26730248839a6e4074c285` |

Convert:

```bash
curl -L -O https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth
python scripts/pth_to_safetensors.py basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth \
  -o kinovsr/basicvsrpp/weights/basicvsrpp_vimeo90k_bd.safetensors --strip-prefix 'generator.'
```

The mmediting checkpoints nest under `state_dict` and prefix keys with `generator.`;
output names must match `net.py`'s `_VARIANTS` (`basicvsrpp_<token>.safetensors`).

## 1x-restoration checkpoints (`--restore <token>`)

The same BasicVSR++ architecture with `is_low_res_input=False` (the net downsamples
the input 4x, propagates at 1/4 res, upsamples back to the SAME size). Used as a
temporal restoration PREPROCESSOR, not an upscaler -- `--restore <token>`, default
`decompress_track1`. Output names must match `net.py`'s `_RESTORE_VARIANTS`
(`basicvsrpp_<token>.safetensors`). Convert the same way (`--strip-prefix 'generator.'`).

| token | task | download | sha256 |
| --- | --- | --- | --- |
| decompress_track1 | NTIRE'21 Compressed Video, Track 1 (c128n25) | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth | `7b2eba02a24989bfbf8b2ed4a06c8e6fd5dbeb193b1178ef7171cd1c455ddb0f` |
| decompress_track2 | NTIRE'21 Compressed Video, Track 2 | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth | `eeae05e655b9849d57a7bb732004ebeb75d376cb0bf76792f18ca0f9eb491774` |
| decompress_track3 | NTIRE'21 Compressed Video, Track 3 | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth | `6daf4a405b0ff7221e3ac39b0a5c788468ae17661c577a3353b9fd477d0c983a` |
| denoise | temporal video denoise (c64n15) | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_denoise-28f6920c.pth | `28f6920c4681bd8468a831d479459872c7f020d95c40eb3ced4745670bf2d596` |
| deblur_dvd | real handheld-video deblur (DVD) | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_deblur_dvd-ecd08b7f.pth | `ecd08b7f938c54f5cd8b2e62cbf4867a1b9e0500ce728bd981a6b99ea1ea9dbf` |
| deblur_gopro | synthetic GoPro deblur | https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_deblur_gopro-3c5bb9b5.pth | `3c5bb9b57b74b268624236ae01e7dc5f5d76ca0cdd3577f2f94d92943900f714` |

Notes:
- The NTIRE decompress + ntire_vsr checkpoints are meant to run through an 8-way
  geometric self-ensemble at inference (their config declares
  `SpatialTemporalEnsemble`; the mmagic re-port dropped it as dead config). Opt in
  with `--restore-ensemble` (1x) / `--basicvsrpp-ensemble` (SR) -- 8x compute, a
  mild artifact-reducer; not applied by default.
- Arbitrary input resolutions are handled by padding to a multiple of 4 and cropping
  back (the fix the author gives in ckkelvinchan/BasicVSR_PlusPlus issue #24).
- decompress_track2 is the most aggressive synthesizer and over-textures flat
  directional surfaces (road/asphalt "alligator skin"); prefer track1/track3 or
  reduce `--restore-strength` there.

Source: https://github.com/open-mmlab/mmagic and
https://github.com/ckkelvinchan/BasicVSR_PlusPlus -- BasicVSR++ (Chan et al.,
CVPR 2022); the NTIRE 2021 champions for Video SR and Compressed Video Enhancement.
Apache-2.0.
