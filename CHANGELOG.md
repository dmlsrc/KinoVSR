# Changelog

All notable user-facing changes to KinoVSR are recorded here. The format follows Keep a
Changelog, and versions follow semantic versioning: before 1.0, a minor release may break
compatibility. [docs/API.md](docs/API.md) defines the supported Python surface, `kinovsr.api`;
every other module is internal and can change in any release.

## 0.2.0 - 2026-10-18

This release is mostly about reading and writing real-world files. The FFmpeg reader now gives
frames their display order and timestamps in packed-bitstream DivX and Xvid AVIs, H.264 in AVI
with B frames, and FLV files with zeroed composition offsets. A read that starts partway through
a file begins on the requested frame in every container, MPEG program streams are read with the
FFmpeg reader by default, and rotated or mirrored files keep their orientation on PyAV 19. Output
video is written in one-second movie fragments under a visible `.partial.mp4` name, so a stopped
or crashed run leaves a playable file. RealPLKSR gains `deh264`, a same-size H.264 artifact
restoration profile. KinoVSR now requires Python 3.15, MLX 0.32.4, and, for its optional extras,
PyAV 19.0.1, and `--denoise-first` is gone; read Breaking changes first.

### Breaking changes

- KinoVSR now requires Python 3.15, MLX 0.32.4 or later, and protobuf 7.36.2 or later. MLX
  releases before 0.32.3 hold Python's GIL while `mx.synchronize()` and `mx.clear_streams()` wait
  for the GPU, which can deadlock against the release of a zero-copy output frame. They also leak
  the evaluated arrays a compiled function with a multi-output operation captured when that
  function is dropped, which happens to BasicVSR++'s deformable alignment whenever the compile
  cache evicts it.
- The `ffmpeg`, `eval`, and `dev` extras require PyAV 19.0.1 or later, which bundles FFmpeg 9.
  PyAV 19 changed interfaces the FFmpeg reader and the evaluation readers use, so earlier PyAV
  releases are no longer supported.
- `--denoise-first` is removed. It only swapped the deblock and denoise stages, and did nothing
  when deblocking was off. To run denoise first, pass `--preprocess-order denoise,deblock`, or
  list the stages in that order in a config file's `pipeline`.

### Added

- RealPLKSR's `deh264` profile restores H.264 compression artifacts at the original size, one
  frame at a time. Select it with `--restore deh264`, alone or ahead of an upscaler;
  `--restore-strength` blends its output with the input, from 0 (the input) to 1 (the full model
  output). In a config file it is a stage with `processor = "realplksr"`,
  `capability = "restore"`, and `profile = "deh264"`. The checkpoint is Philip Hofmann's
  1xDeH264_realplksr under CC-BY-4.0, an external download that the
  [RealPLKSR guide](docs/processors/realplksr.md) and the family's weights README explain how to
  install. `--realplksr-scale 1` declares an explicit 1x checkpoint.
- The FFmpeg reader reads raw H.264 and HEVC elementary streams. Streams that carry no
  timestamps, and MPEG-1 and MPEG-2 streams whose muxer gave two pictures one PES timestamp, are
  now numbered by coded picture in decode order.
- Python additions, all in internal modules outside the supported `kinovsr.api` surface:
  - `kinovsr.media.reader.auto_reader(path)` returns the reader that automatic selection uses
    for a file.
  - `kinovsr.media.ffmpeg_reader.iter_coded_pictures` reports each frame's picture type and,
    where libavcodec exports it, its mean block quantizer. Frames are numbered and placed exactly
    as the frame readers deliver them, and no pixels are converted.
  - `kinovsr.native.writer.AVWriter` takes `fragment_interval` in seconds, 1 by default, where
    `None` writes a plain MPEG-4. `AVWriter.finish(early=True)` ends the file at the last video
    frame written instead of draining the rest of the source audio.
  - `AVWriter.append_mlx_rgb`, `AVWriter.prepare_mlx_rgb`, and
    `kinovsr.media.yuv.frame_to_yuv422_10` take `signed=True` for model output in [-1, 1].
  - `kinovsr.native.anemil.runtime.ModelRunner` takes `ane_compiler_options`, a
    `KEY=VALUE,KEY=VALUE` string of Neural Engine compiler flags that it passes through a private
    Core ML interface. Core ML caches each compile by program and options.
  - `kinovsr.native.anemil.runtime.publish` publishes a cache entry through a staging directory
    of the run's own.
  - `kinovsr.native.sensors.Reading.gpu_power_reasons` gives the GPU power controller's residency
    as shares of the time the GPU was busy: `PERF` and `PRFBOOST` while it runs as fast as its
    load asks, and `PPMTARGE`, `PPMLIMIT`, `POWER`, `PWR_ZONE`, and others while the power or
    thermal controller holds it below that.
  - `kinovsr.native.sensors.SensorSampler` takes `groups` and `utilization`, so a caller that
    polls many times a second can read the GPU Stats group alone, without the IOAccelerator or
    SMC reads. `kinovsr.native.sensors.SystemPower` reads the SMC system power rail alone.

### Changed

- Output video is written in one-second movie fragments. A file that is never closed plays up to
  its last complete fragment, and other programs can open it while it is written. Closing it
  turns it into a plain MPEG-4 in place, with a few hundred unused bytes per fragment.
- The output video is written as `<name>.partial.mp4` beside its destination and renamed when the
  run finishes; 0.1.0 wrote a hidden temporary file and deleted it on any failure. A run that
  fails or is interrupted after writing frames now closes the video at its last frame and keeps
  it under the partial name. A later run takes the next free `.partial-N` name instead of
  overwriting it. A partial video without frames is removed, and the run's other outputs roll
  back as before.
- Automatic reader selection, which `--reader auto` and `process_video_file` without a reader
  use, reads MPEG program streams with the FFmpeg reader when the `ffmpeg` extra is installed.
  These are VCD and DVD `.mpg` and `.vob` files, recognized by their content rather than their
  name. AVFoundation can stop short of the last frames of a video-only program stream, and drops
  a frame or fails partway where a muxer stamped a PES timestamp on the wrong picture; the FFmpeg
  reader delivers every frame of both. Other files keep the native reader, and `--reader native`
  now always uses it.
- MLX frames are converted to the encoder's 10-bit 4:2:2 planes in one compiled graph that reads
  each frame once, where 0.1.0 first made two float32 copies of every frame. The planes are
  byte-identical to 0.1.0's; [Performance](#performance) has the speed and energy.
- Core ML models that KinoVSR compiles for the Neural Engine are moved into its cache instead of
  copied, and SpyNet's flow levels load from the cache. Core ML keys its Neural Engine build by
  the path a model loads from, so a later run no longer rebuilds every SpyNet level when it
  starts; [Performance](#performance) has the timing.
- GPU watts in `kinovsr.native.sensors` come from IOReport's "GPU Energy" channel, which updates
  at every sample, instead of "GPU0", which changes about once a second and made one-second
  readings alternate high and low. "GPU0" stands in where "GPU Energy" is missing. The
  `system_w` reading is documented as about the previous second's average, as every SMC power
  key reports, not an instantaneous value.

### Performance

Measured on an Apple M1 Max under macOS 27, each against 0.1.0.

| Change | Workload | Result |
| --- | --- | --- |
| One-graph YUV conversion | Conversion and plane copy of one 2560x1920 frame | 2.39 ms, against 4.07 ms |
| One-graph YUV conversion | HEVC Main 4:2:2 10-bit encode of 1,080 frames at 2560x1920, as fast as it runs | 108 frames a second, against 91 |
| One-graph YUV conversion | SoC energy per frame in that encode | 0.112 J, against 0.153 J |
| One-graph YUV conversion | GPU energy per frame in that encode | 0.019 J, against 0.047 J |
| One-graph YUV conversion | DRAM and DCS energy per frame in that encode | 0.065 J, against 0.084 J |
| SpyNet models loaded from the cache | First start of a later run at 640x352 | 0.09 s, against 0.53 to 0.58 s |

### Removed

- `--denoise-first`; use `--preprocess-order`.

### Fixed

- The FFmpeg reader gives frames their display order and timestamps in streams whose packets are
  stamped in decode order: DivX and Xvid AVIs written as a packed bitstream, H.264 in AVI with B
  frames, and FLV files whose muxer zeroed H.264's composition offsets. 0.1.0 mislabeled their
  frames, and the file pipeline refused the source. A slot whose packet holds no picture, such as
  the one-byte delay frame DivX writes after the first I frame, repeats the picture before it, as
  players show such a drop frame.
- A read that starts partway through a file (`--start`) begins on the requested frame in every
  container. FFmpeg's seek in MPEG program and transport streams, RealMedia files, and raw
  elementary streams can land past the keyframe the start needs, and the timestamps it estimates
  after a seek differ from those of a read from the beginning. 0.1.0 could therefore fail with
  "video decode delivered sample 53745; expected 53740" or return another frame under the
  requested number.
- The FFmpeg reader now seeks only in containers that store every frame's timestamp (MP4 and MOV,
  Matroska and WebM, AVI, FLV, ASF, DV, and MXF), and it checks where each seek lands, because
  with B frames an MP4 seek can land on the keyframe shown after its target. Other containers are
  demuxed from the beginning without decoding. Audio windows follow the same rules, which also
  ends the decode errors a program stream's audio window raised.
- The native reader checks where a trimmed range lands and rereads from the beginning when the
  range starts late or delivers nothing. An open-ended read no longer stops at the asset's
  duration, which for an MPEG program stream ends before its last frames and dropped them.
- Rotated and mirrored files read by the FFmpeg reader keep their orientation on PyAV 19, which
  made a frame's side data a mapping. 0.1.0's probe iterated it as a list of entries and fell
  back to upright, so on PyAV 19 a rotated or mirrored file lost its rotation or mirroring
  whenever the FFmpeg reader read it: a Matroska file, a file in another container AVFoundation
  cannot open, or any file read with `--reader ffmpeg`.
- Video files whose metadata tags are not UTF-8, such as the Mac Roman copyright sign that old
  QuickTime writers stored, open in the FFmpeg reader and the evaluation readers. PyAV 18 refused
  them while opening.
- Audio read through the FFmpeg reader no longer fails with "audio stream ... overlaps by N
  samples" where a packet's timestamp overlaps the samples before it: the first decoded samples
  are kept and the overlap is trimmed. Vorbis audio whose granule positions run one frame ahead
  and then return, which also failed, is placed at its decoded position, and a sustained gap
  still becomes silence.
- Since macOS 27.0.1, AVAudioFile reports a read at or past the end of the audio as an
  end-of-file error, where macOS 27.0 returned no error, and the native audio source raised on
  it. A source whose audio ends before its video could fail when the run reached the end of the
  audio; processing KinoVSR's own output again, when its ALAC audio is a whole number of packets,
  failed this way. The source now takes that error as the end of the audio and returns only the
  frames that exist.
- A source without an audio track, or with an empty one, no longer opens an audio reader. An
  audio track whose samples carry no usable presentation timestamp is now an error; 0.1.0 placed
  it as if it started at time zero.
- A BSVD run that stops early, for example at `--max-frames`, no longer fails while it shuts
  down: the queue that completes its background work stays open until every stage has closed.
- The MLX lane no longer changes MLX's process-wide default device. It entered `mx.stream()`
  around every callback, and the terminal rebuild entered one on the consumer's thread; that
  context also sets and restores the default device, so when a host thread switched the default
  device to the CPU, a callback could end the switch early or leave every thread on the CPU. The
  lane now binds its stream once per thread.
- A finished progress bar's pace, and the time the phase summary logs for its phase, stop at its
  finish time. 0.1.0 kept counting while the other bars ran, so a finished bar's seconds per
  frame kept growing.
- Compiling a Core ML model for the Neural Engine no longer leaves a copy of the compiled model
  in the user's temporary directory, and SpyNet's Neural Engine loader no longer leaves one
  compiled model per pyramid level there on every cache miss.
- Two runs that start on the same empty Neural Engine cache no longer break each other. Every
  cache writer staged through one fixed name, so one of two SpyNet or BSVD runs started together
  failed, and compiling a model could fail with `ENOTEMPTY`. Each run now stages in a directory
  of its own and renames the result into place, and a run that finds the entry already published
  uses that copy. When the cache cannot be written, SpyNet loads each level straight from Core
  ML's output.

### Documentation

- The [RealPLKSR guide](docs/processors/realplksr.md) covers `deh264` restoration, and the
  family's weights README and attribution list the checkpoint's download, SHA-256, and license.
- The README describes the `.partial.mp4` output and its fragments and links this changelog.
- The [usage guide](docs/USAGE.md) documents how `--reader auto` treats MPEG program streams and
  the `deh264` restore profile, and names `--preprocess-order` as the way to reorder
  preprocessing.
- The [API](docs/API.md) compatibility policy points to this changelog for breaking changes.

### Development

- The `dev` extra requires torch 2.15.0, mypy 2.4.0, Ruff 0.17.1, NumPy 2.5.4, and pytest
  9.1.1, and Ruff and mypy target Python 3.15. torch 2.15.0 is the first release whose Python
  3.15 wheels were built against the final interpreter; the 2.14 builds crash on tensor
  operations outside the main thread. PyTorch publishes these wheels on its own index, so
  install the extra with `uv pip install -e '.[dev]' --torch-backend cpu`.
- The test suite gives Core ML a cache home of its own, `~/Library/Caches/KinoVSR/tests-home`,
  through `CFFIXED_USER_HOME`, so a full run no longer fills the user's own Core ML cache, which
  Core ML never prunes. The test home persists between runs and is cleared past 2 GiB, but only
  by a session that finds no other session using it.
- Tests that need an external BSVD, FBCNN, or RealPLKSR checkpoint skip when it is not
  installed. Their guards caught a `FileNotFoundError` that the pipeline runtime reports as a
  `PipelineRuntimeError`, so on a checkout without those weights they failed instead of
  skipping; each now checks for its checkpoint before it runs.
- Under FFmpeg 9, an MP4 edit list's tail trim drops real samples from the shorter final ALAC
  frame that AVAssetWriter writes, so the pipeline tests count output audio with AVFoundation,
  which wrote it.

## 0.1.0 - 2026-09-26

Initial public release: MLX-native video super-resolution and restoration for Apple Silicon on
macOS 27. KinoVSR reads video through AVFoundation, or through the optional PyAV reader for other
containers, and writes 10-bit HEVC that carries the source audio. Native stages provide
VideoToolbox and MetalFX upscaling, VideoToolbox frame-rate conversion, and timeline, crop,
exposure, and flicker repair. Learned MLX families cover denoising, deblocking, restoration, and
2x and 4x upscaling, and BSVD's denoiser also runs on the Neural Engine. One CLI composes them
into chains set by flags or TOML, `kinovsr weights` lists and verifies the weights each family's
manifest declares, `kinovsr metrics` scores results, and `kinovsr.api` is the supported host API.
