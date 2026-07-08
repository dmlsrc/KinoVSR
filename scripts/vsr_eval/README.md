# VSR Evaluation Helpers

These scripts are developer-only tooling for comparing denoiser/VSR outputs.
They are intentionally config-driven so local clip paths and output artifact
names stay out of git.

## Tools

- `face_yunet_metrics.py`: detects faces in a baseline video with OpenCV YuNet,
  tracks those face boxes, and scores each candidate on face-region cleanup
  versus preservation. It writes CSV/JSON plus an optional contact sheet.
- `perceptual_metrics.py`: scores a manifest of variant videos (plus the
  source) with MUSIQ, DOVER-Mobile (tech/aes/fused), NIQE, flicker on
  source-static pixels, moving-content deviation (the content-erasure
  guard: opinion metrics score erased movers the same or better, this
  column catches them), and VMAF-vs-source, printing one ranked table
  and writing CSV/JSON. The module docstring says what each column
  means and which are ranking columns versus tripwires/anchors.
- `run_denoise_sweep.py`: runs `scripts/vsr_harness.py` across configured clips
  and variants, writes per-run logs/manifests, then optionally invokes the face
  and perceptual evaluators and aggregates the metrics.
- `musiq_score.py`, `dover_score.py`, `niqe.py`: the individual metric CLIs
  behind `perceptual_metrics.py`, for scoring ad-hoc files.

## Local Config

Copy `vsr_eval.example.toml` to `vsr_eval.local.toml` and put machine-local
paths there:

```bash
cp scripts/vsr_eval/vsr_eval.example.toml scripts/vsr_eval/vsr_eval.local.toml
```

Then run:

```bash
scripts/vsr_eval/run_denoise_sweep.py --config scripts/vsr_eval/vsr_eval.local.toml
```

For standalone face scoring, set `[face_eval.variants]` or
`face_eval.variants_json` in the local config and run:

```bash
scripts/vsr_eval/face_yunet_metrics.py --config scripts/vsr_eval/vsr_eval.local.toml
```

## Weights

The YuNet ONNX used by default is vendored under `weights/` because it is tiny
and MIT licensed. Override `face_eval.model` in local config if you want to test
a different detector.
