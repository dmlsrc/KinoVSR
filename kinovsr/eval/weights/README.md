# YuNet Face Detector Weights

`face_detection_yunet_2023mar.onnx` is bundled for
`kinovsr metrics faces`. It is small (~227 KB) and MIT
licensed, so keeping it in the repo avoids a local Hugging Face cache dependency
for face-region denoiser evaluation.

| file | source | sha256 | xet hash | license |
| --- | --- | --- | --- | --- |
| face_detection_yunet_2023mar.onnx | https://huggingface.co/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | `d3fbb6028fc86125755b70f69f428ebbf39fbb01cfa5df3e3dbe1563148ae349` | MIT |

The license text and attribution are in `../ATTRIBUTION.md`.

Source: OpenCV Zoo `models/face_detection_yunet`; upstream model source is
https://github.com/ShiqiYu/libfacedetection.train.

To refresh from Hugging Face:

```bash
hf download opencv/opencv_zoo
```

With the standard Hugging Face cache layout, that places the snapshot under
`$HF_HOME/hub/models--opencv--opencv_zoo` when `HF_HOME` is set. If `HF_HOME` is
unset, Hugging Face uses `~/.cache/huggingface/hub`; `HF_HUB_CACHE` can override
the hub cache path directly.
