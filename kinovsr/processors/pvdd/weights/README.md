# PVDD Weights

Not bundled -- download + convert; the `.safetensors` are gitignored. The six
released checkpoints share one `pvdd0815` architecture (num_in and the level flag
are inferred from the weights at load).

## Source

The pretrained checkpoints ship as a single archive from the official PVDD
release ("Trained Weights"):

- Google Drive: https://drive.google.com/drive/folders/1qEmupCR4JcaPNky3B5ldRN88t8K6CGaG
- The archive unpacks to six `g_net_100000.pth` files, one per subdir.

| variant token | subdir / source `.pth` | domain | level | sha256 |
| --- | --- | --- | --- | --- |
| `pvdd` | `pvdd_srgb_nolevel/g_net_100000.pth` | sRGB | blind | `c76b91c66a6e923e03a8f4847af61968b889c34911ee964a2894506d169d0eff` |
| `crvd` | `crvd_srgb_nolevel/g_net_100000.pth` | sRGB | blind | `a327de3ba8c6b82f6deb78c4930ff077954b920c9258e7ba7f71c7cba68cd9af` |
| `davis` | `davis_srgb_nolevel/g_net_100000.pth` | sRGB (synthetic AWGN) | blind | `410065f431b36ceb1328ea5d58a7c09838a81e09ea5b14d516070a3a8c6c2b94` |
| `pvdd_level` | `pvdd_srgb_level/g_net_100000.pth` | sRGB | noise-variance map | `fae9ba86080df49f2182079b4ac8efce89b537dcbb8b3db6f54b41f2674773df` |
| `pvdd_raw` | `pvdd_raw_nolevel/g_net_100000.pth` | packed Bayer (4ch) | blind | `12cabfbfb807a1dbae74a5d5d11e3cc89e8d3a1795bd84a4ef62c553d6116fe4` |
| `pvdd_raw_level` | `pvdd_raw_level/g_net_100000.pth` | packed Bayer (4ch) | noise-variance map | `b4bfbb04176feff5e4e3cf99556c992d28244e63b060c92239ad1cd68ce6b703` |

## Convert

Verify the sha256, then convert each `.pth` with the safe generic converter
(static pickle scan + `weights_only=True`; nothing in the checkpoint is executed).
The MLX loader keeps the torch NCHW layout in the file and transposes conv weights
to OHWI at load, so no conversion flags are needed.

```bash
kinovsr weights convert \
  weights-src/pvdd_srgb_nolevel.pth \
  -o kinovsr/processors/pvdd/weights/pvdd_srgb_nolevel.safetensors
# ... repeat for the other five, keeping the filenames in the table above.
```

## Notes

- The blind (`nolevel`) checkpoints need no side input. The `level` checkpoints
  take a per-pixel noise-**variance** map (not sigma); the reference S/M/L presets
  are 0.000687765 / 0.002191 / 0.005470, exposed as `LEVEL_PRESETS`.
- The `raw` checkpoints expect packed 4-channel Bayer, not sRGB video; they are
  loadable but need a raw pipeline to drive.
- The unused `feat_STTB.*` pre-attention weights (commented out in the reference
  forward) are dropped at load.
