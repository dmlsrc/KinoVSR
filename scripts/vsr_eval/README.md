# VSR Evaluation Helpers

These scripts are developer-only tooling for comparing denoiser/VSR outputs.
They are intentionally config-driven so local clip paths and output artifact
names stay out of git.

## Tools

- `face_yunet_metrics.py`: detects faces in a baseline video with OpenCV YuNet,
  tracks those face boxes, and scores each candidate on face-region cleanup
  versus preservation. It writes CSV/JSON plus an optional contact sheet.
- `run_denoise_sweep.py`: runs `scripts/vsr_harness.py` across configured clips
  and variants, writes per-run logs/manifests, then optionally invokes the face
  evaluator and aggregates the metrics.

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
