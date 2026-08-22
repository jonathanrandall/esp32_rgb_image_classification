# esp32_rgb_image_classification

Real-time image classification on an ESP32-S3 camera board, in the **pixel
domain**: the camera is read as raw RGB565, reduced to a 32x24 grid by
averaging 5x5 blocks of pixels, and classified by a small int8 CNN running
under [ESP-NN](https://github.com/espressif/esp-nn).

Five classes: `people`, `computer`, `doors`, `fruit`, `car`.

**81.0% int8 test accuracy on 2,304 input values per frame.** Everything here
— training, quantization, C export, bit-exactness verification, and the
firmware — is what produced the weights in
`esp32_cam/esp32_rgb_cnn/include/model_weights.h`.

> This is the pixel-domain half of a comparison. The compressed-domain half —
> classifying JPEG DCT coefficients directly, skipping the decode — lives in
> [esp32_dct_jpg_image_classification](https://github.com/jonathanrandall/esp32_dct_jpg_image_classification).

---

## Why average blocks instead of resizing

A 5x5 box mean is not an arbitrary choice of downsampler. It is the same
operation the JPEG pipeline already performs.

The orthonormal 2-D DCT-II divides by `sqrt(N)` per axis, so for an 8x8 block
the DC coefficient is exactly

```
DC = 8 x (mean of the block)
```

An 8x8 block mean and a JPEG DC plane are therefore the *same picture at the
same resolution*, differing only in scale. That makes block-mean RGB the
equal-resolution control for a DCT classifier: feed both models the same grid
and the only thing left varying is the representation, not the resolution.

5x5 is the operating point this project settled on. Measured, same
architecture, same data:

| reduction | input grid | values/frame | int8 test |
|---|---|---:|---:|
| 1x1 (full resolution) | 120x160 | 57,600 | 82.5% |
| 4x4 | 30x40 | 3,600 | 81.0% |
| **5x5** | **24x32** | **2,304** | **81.0%** |
| 8x8 | 15x20 | 900 | 78.0% |

4x4 scores the same as 5x5 while carrying 56% more data, so its extra values
buy nothing. Past 5x5 the curve breaks — 8x8 gives up three points.

A useful property falls out of this input: block means are centered to
`[-128, 127]`, which is *already exactly int8*. The input quantizer's scale is
1.0 and nothing needs calibrating.

---

## Results

From `results/accuracy_table.json`, produced by the run that generated the
shipped weights:

| split | float | QAT | int8 |
|---|---:|---:|---:|
| train | 90.7% | 96.1% | 96.1% |
| val | 79.9% | 81.0% | 80.8% |
| test | 78.2% | **81.0%** | **81.0%** |

int8-vs-QAT prediction agreement: **99.3%**. The int8 reference is bit-exact
against the C export — see *Verification* below.

QAT scoring above float is not a typo; quantization noise acts as a
regularizer on a model this small.

For on-device latency, read it from the board rather than from this file:
`GET /status` reports `capture_ms`, `convert_ms`, `infer_ms`, `jpeg_ms` and
`total_ms` per frame, plus a per-layer breakdown. (For scale: the 4x4 variant
measured 33 ms of inference on the same board; 5x5 is smaller.)

---

## Hardware

- **Freenove ESP32-S3-WROOM CAM** (OV2640), PSRAM required.
- Capture is `PIXFORMAT_RGB565` at `FRAMESIZE_QQVGA` (160x120), set **at
  `esp_camera_init()` time**. The "init at QVGA, then `set_framesize()` down"
  trick is JPEG-only — doing it with RGB565 sizes the framebuffer for the
  wrong format and you get a `Stack canary watchpoint triggered (cam_task)`
  panic. This is noted in `src/main.cpp` where it bites.

---

## Building the firmware

[PlatformIO](https://platformio.org/). The environment is
`freenove_esp32_s3_wroom`.

**You must create `esp32_cam/esp32_rgb_cnn/include/secrets.h` first** — it is
not in this repository, and the build will not proceed without it:

```c
#pragma once
#define WIFI_SSID     "your-network"
#define WIFI_PASSWORD "your-password"
```

Then:

```bash
cd esp32_cam/esp32_rgb_cnn
pio run -t upload
pio device monitor
```

The serial log prints the board's IP. Open it in a browser.

### Endpoints

| | |
|---|---|
| `http://<ip>/` | UI: live preview + running classification |
| `http://<ip>:81/stream` | MJPEG stream (separate port, separate httpd task) |
| `http://<ip>/status` | JSON: timing, per-layer breakdown, self-test results, signal diagnostics, buffer report |
| `http://<ip>/config` | JSON: build configuration |

`/status` is worth reading before anything else. It carries the boot self-test
verdict, a `byte_order` block that measures the RGB565 byte order both ways,
and a `buffers` report of where every activation buffer was allocated.

### ESP-NN

Enabled by two flags in `platformio.ini`:

```ini
build_flags =
    -DCONFIG_NN_OPTIMIZED=1
    -DCONFIG_IDF_TARGET_ESP32S3=1
```

`model_weights.h` carries both code paths behind those defines: with them,
`esp_nn_conv_s8` with OHWI weights; without them, the identical
`model_forward()` in portable C with OIHW weights. Only the conv kernel and
weight layout differ, which makes portable-vs-accelerated a clean same-weights
comparison on this board.

To build portable, comment out both `-D` lines **and** add `lib_ignore =
esp_nn`. The second step is not optional: PlatformIO's dependency finder
decides what to compile by text-scanning for `#include` and does not evaluate
preprocessor conditionals, so dropping the defines alone still drags
`lib/esp_nn/`'s sources into the build. Verify either way with
`xtensa-esp32s3-elf-nm firmware.elf | grep -c esp_nn`.

The one real constraint when writing ESP-NN code by hand: the conv **scratch
buffer pointer** must be 16-byte aligned. `heap_caps_malloc` guarantees only
8, and a misaligned scratch pointer makes the assembly kernel silently corrupt
its output. Use `heap_caps_aligned_alloc(16, ...)`. The generated
`model_esp_nn_init()` already does this, so using the header's own init gets
it right for free.

---

## Training pipeline

Full detail in [`python_code/README.md`](python_code/README.md), including a
[complete CLI reference](python_code/README.md#cli-reference) — every option in
every script, with its default. The short version:

```bash
cd python_code

# 1. fetch + curate the dataset (see data_curation/ for the people filter)
python get_everyday_openimages_data.py
python detect_people_yolo.py
python build_data.py

# 2. train: float -> QAT -> bit-exact int8
python train_rgb_cnn.py \
    --rgb-block-width 5 --rgb-block-height 5 \
    --classes people,computer,doors,fruit,car \
    --artifacts-name rgb_cnn_5x5

# 3. export C weights and verify them against the Python int8 reference
python export_rgb_cnn_c_weights.py rgb_cnn_5x5
python verify_rgb_cnn_c_export.py  rgb_cnn_5x5
```

`export_rgb_cnn_c_weights.py` writes `output/rgb_cnn_5x5/esp32/model_weights.h`;
`verify_rgb_cnn_c_export.py` writes the self-test vector headers
(`rgb_synth_vectors.h`, `rgb_real_images.h`) beside it in
`output/rgb_cnn_5x5/`. Copy all three into `esp32_cam/esp32_rgb_cnn/include/`
and rebuild.

### Hand curation and fine-tuning

Train the base model on all of `data/` untouched, then hand-curate a subset and
fine-tune on it. Batch pull, a single review page with a folder dropdown, and a
reject record that survives a `build_data.py` rebuild:

```bash
python python_code/curation_pull.py          # pull a batch to review
python python_code/make_review_gallery.py    # one page, folder dropdown
python python_code/curation_resolve.py       # accept/reject, record durably

python python_code/train_rgb_cnn.py \
    --data-dir data_hand_curated --fine-tune-from <base-run> \
    --rgb-block-width 5 --rgb-block-height 5 \
    --classes <same list and order as the base run> \
    --artifacts-name <base-run>_ft
```

Full walkthrough: [`finetune_curation.md`](finetune_curation.md).

Reproducibility caveat: training is seeded (`--seed`, default 1234) but
`torch.use_deterministic_algorithms` is not set, so cuDNN's backward kernels
give roughly 0.5–1.4 points of run-to-run variation. Differences smaller than
that between configurations are noise, not signal.

---

## Verification

The firmware's `rgb_model_forward_timed()` is a buffer-hoisted mirror of the
generated `model_forward()`. The generated function cannot be called on this
target under any circumstances — it declares ~619 KB of intermediates as stack
locals, more than the ESP32-S3 has internal SRAM in total. So there is no
on-device reference to diff against, and correctness has to be established
against the PC-side int8 reference instead, by shipping expected outputs in
headers.

Three tiers of vectors run at boot, in ascending order of what they actually
prove:

1. **LCG noise** (`rgb_synth_vectors.h`) — cheap smoke test, weak evidence.
   White noise has no spatial structure to survive four convolutions and a
   global average pool, so all seeds pool to nearly the same feature vector.
   Nearly blind to row-pitch mistakes, padding off-by-ones, and stride-2 phase
   errors.
2. **Structured patterns** — catch the spatial bugs tier 1 cannot.
3. **Real dataset images** (`rgb_real_images.h`) — the actual distribution.

Results land in `/status` under `self_test`, including `bit_exact_all` and
`max_logit_diff`. A `determinism_ok` flag catches non-reproducibility across
repeated runs of the same input.

---

## Repository layout

```
esp32_cam/esp32_rgb_cnn/     PlatformIO firmware
  src/main.cpp                 capture, RGB565->block mean, inference, HTTP, streaming
  include/model_weights.h      generated: int8 weights + model_forward()
  include/rgb_synth_vectors.h  generated: LCG + structured-pattern vectors
  include/rgb_real_images.h    generated: real test-split images + expected logits
  include/rgb_test_vectors.h   hand-written: the harness that runs all three tiers
  include/camera_pins.h        Freenove ESP32-S3-WROOM pinout
  lib/esp_nn/                  vendored from espressif/esp-nn (see VENDORED_FROM.md)
  network_info.json            the shipped model's manifest

python_code/                 training, export, verification
  dct_common/                  shared library (feature extraction, QAT, quantization)

data_curation/               how the raw pull was selected and filtered
finetune_curation.md         the hand-curation -> fine-tuning workflow
results/                     accuracy table, confusion matrix, manifest for the shipped run
```

Generated headers are committed deliberately: `model_weights.h` and the
self-test vectors are what make this repository buildable and checkable
without rerunning training.

---

## Attribution

`lib/esp_nn/` is vendored from
[espressif/esp-nn](https://github.com/espressif/esp-nn) at commit
`10b6c0fc884a3b05f94a752f91d00ebadfe5d8d0`, scoped to the ESP32-S3 int8 conv2d
path only. `lib/esp_nn/VENDORED_FROM.md` records exactly which files were
taken, which were deliberately left out, and why.

Dataset images come from
[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) via
[FiftyOne](https://docs.voxel51.com/) (plus a Places365 pass for `garden`).
Each is a padded crop around one annotated object, letterboxed to 160x120 and
re-encoded 4:2:2 to match the camera. People filtering uses YOLO11m over the saved pixels, on
top of Open Images' own annotations — the annotations alone are incomplete,
since they box salient objects rather than every person in every frame.

See `LICENSE` for this project's terms.
