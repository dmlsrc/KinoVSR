# Attribution

`kinovsr/eval/` contains the evaluation helpers for VSR denoiser comparisons. The
YuNet model weights are redistributed under the upstream license (see `README.md`
in this directory for provenance and hashes).

## NIQE pristine model

`niqe_pristine_reds.safetensors` is a first-party artifact: multivariate-
Gaussian statistics fitted by this repo's own tool from pristine frames.
It redistributes no upstream weights. The method is:

Anish Mittal, Rajiv Soundararajan, Alan C. Bovik -- "Making a 'Completely
Blind' Image Quality Analyzer." IEEE Signal Processing Letters, 2013.

## YuNet Face Detector

Wei Wu, Hanyang Peng, Shiqi Yu -- "Yunet: A tiny millisecond-level face detector."
https://github.com/ShiqiYu/libfacedetection.train

Bundled model source: https://huggingface.co/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Licensed under the MIT License (SPDX: MIT):

    MIT License

    Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
