# MUSIQ weights

Not bundled (104 MB fp32). Convert the released koniq checkpoint:

    python LTX_2_MLX/videotoolbox/musiq/convert_musiq.py \
        /path/to/musiq_koniq_ckpt-e95806b9.pth

The source checkpoint is distributed by the IQA-PyTorch project
(chaofengc/IQA-PyTorch, release tag v0.1-weights), converted from the
original Google Research MUSIQ TensorFlow checkpoints (Apache-2.0). See
ATTRIBUTION.md.
