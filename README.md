# esp32_rgb_image_classification

Real-time image classification on an ESP32-S3 camera board, in the **pixel
domain**: the camera is read as raw RGB565, reduced to a 32x24 grid by
averaging 5x5 blocks of pixels, and classified by a small int8 CNN running
under [ESP-NN](https://github.com/espressif/esp-nn).

Five classes: `people`, `computer`, `doors`, `fruit`, `car`.

**81.3% int8 test accuracy on 2,304 input values per frame**, 78.9% balanced. Everything here
— training, quantization, C export, bit-exactness verification, and the
firmware — is what produced the weights in
`esp32_cam/esp32_rgb_cnn/include/model_weights.h`.

### Both arms are here

This is one half of a comparison, and the other half is in this repository too.

| arm | reads | firmware | training script |
|---|---|---|---|
| **pixel domain** | RGB565 → 5×5 block means | `esp32_cam/esp32_rgb_cnn/` | `train_rgb_cnn.py` |
| **compressed domain** | JPEG DCT coefficients, no decode | `esp32_cam/esp32_classifier/` | `train_cnn.py` |

The compressed-domain arm never reconstructs pixels: the OV2640 encodes JPEG in
hardware, so the DCT coefficients already exist in the bitstream and the model
reads them directly. The pixel arm exists to measure what that is worth on
matched data and a matched architecture.

An 8×8 block mean *is* the DCT DC coefficient (the orthonormal DCT-II divides by
`sqrt(N)` per axis, so `DC = 8 × mean`), which is what makes block-mean RGB the
honest equal-resolution control rather than an arbitrary downsampler.

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

> This sweep was measured on an earlier build of the dataset and with the
> narrower `16,32,64` conv stack. It is kept because the *shape* of the curve —
> where the knee is — is what chose 5x5, and that is what it shows. Do not read
> its absolute values against the table below.

A useful property falls out of this input: block means are centered to
`[-128, 127]`, which is *already exactly int8*. The input quantizer's scale is
1.0 and nothing needs calibrating.

---

## Results

From `results/accuracy_table.json`, produced by the run that generated the
shipped weights:

| split | float | QAT | int8 |
|---|---:|---:|---:|
| train | 83.4% | 85.3% | 86.5% |
| val | 82.0% | 82.5% | 82.2% |
| test | 80.2% | 81.3% | **81.3%** |

Balanced (macro-averaged per-class recall), which weights every class equally:
**78.9%** on test.

int8-vs-QAT prediction agreement: **99.7%**. The int8 reference is bit-exact
against the C export — see *Verification* below.

QAT scoring above float is not a typo; quantization noise acts as a
regularizer on a model this small.

Measured on the board via `GET /status`, which reports per-frame timing and a
per-layer breakdown:

| stage | ms |
|---|---:|
| RGB565 -> block-mean convert | 3.5 |
| **inference** | **21.0** |
| JPEG encode (for the preview stream) | 18.8 |
| total | 43.5 |

19.3 fps end to end. Note the JPEG encode is nearly as expensive as the
inference: capturing RGB565 means the frame cannot double as the stream, so
previewing costs a software encode. That is a property of the pixel-domain
pipeline, not of the model.

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

### Streaming reliability

Two symptoms that looked unrelated — the preview stalling for a second or
more, and the board occasionally resetting — turned out to be one fault, and
the reset was the stall escalating.

A stalled Wi-Fi send blocks the stream loop, so frame buffers stop being
returned. The camera driver runs out and logs an overflow from `cam_task`.
That log call reaches newlib's `vprintf`, which allocates a mutex, and
overflows `cam_task`'s stack: **reporting the problem is what killed the
board.** Silencing the driver's tags before `esp_camera_init()` fixes it —

```c
esp_log_level_set("cam_hal",   ESP_LOG_NONE);
esp_log_level_set("s3 ll_cam", ESP_LOG_NONE);
esp_log_level_set("camera",    ESP_LOG_NONE);
```

— and also removes a stall *amplifier*, because each of those lines blocked
~2.6 ms on UART0 per dropped frame, preventing the DMA from being serviced and
causing more drops.

Two smaller fixes sit alongside it: `GET /claim` on port 80 lets a new viewer
evict a stale-but-open socket that owns the single stream slot (port 81 is
inside its handler and cannot hear anything, which is also why
`lru_purge_enable` does not help), and the page's watchdog now recognises
`last_frame_age_ms == -1` — "no frame since boot" is the *absence* of an age,
so an `age > 8000` test could never catch a post-reset stall.

What none of this fixes is the radio. Full write-up, including the decoded
backtrace and the link measurements:
[`stream_stall_issue.md`](stream_stall_issue.md).

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
    --conv-channels 32,32,64 --use-augmentation

# 3. export C weights and verify them against the Python int8 reference
python export_to_firmware.py --upload
```

`export_to_firmware.py` runs the exporter and the verifier in order, refuses to
install anything the verifier did not pass, backs up the headers it replaces,
and flashes. Doing it by hand is easy to get wrong: the exporter writes
`model_weights.h`, but the **verifier** writes the two self-test vector headers,
one directory up — copy only the first and the boot self-test compares a new
model against another model's expected logits.

The compressed-domain arm is the same shape, with its own exporter and
verifier:

```bash
python train_cnn.py \
    --capture-width 160 --capture-height 120 --chroma-subsampling 4:2:2 \
    --num-ac-coeffs 2 --num-chroma-ac-coeffs 0 \
    --classes people,computer,doors,fruit,car --use-augmentation

python export_cnn_c_weights.py cnn     # -> output/cnn/model_weights.h
python verify_cnn_c_export.py cnn      # expect 25/25 bit-exact
```

Then copy `output/cnn/{model_weights.h,test_vectors.h}` into
`esp32_cam/esp32_classifier/include/` and set `DCT_NUM_AC_COEFFS` in
`src/dct_features.h` to match — a `static_assert` in `main.cpp` fails the build
if they disagree.

### Capturing real frames from the board

Both models above are trained on Open Images photographs, and the camera
produces something meaningfully different — a whole room rather than a padded
crop around one object, indoor lighting, the OV2640's own exposure, and the
sensor's own JPEG quantization tables. A model can score well on the test split
and behave poorly on the board without either number being wrong.

The sharpest case: the dataset pipeline **drops person-containing images from
every class except `people`**, so the model has never seen a person and a
computer in one frame — exactly what a camera pointed at a desk shows. The test
split cannot reveal this, having been filtered the same way.

```bash
python capture_board_frames.py --label people --count 200 --interval 1.0 --show-prediction
```

Frames land in `board_captures/<label>/`, numbered continuously across runs.
`--show-prediction` prints what the board currently thinks each frame is, which
is the quickest way to find the live scenes the model gets wrong.

**Capture from `esp32_classifier` only** — it is the one firmware serving
`/frame`, and the only one whose JPEGs come from the OV2640's *hardware*
encoder. `esp32_rgb_cnn` captures RGB565 and software-encodes its preview with
`frame2jpg()`, so its frames carry the software encoder's quantization tables
instead of the sensor's.

That single capture set feeds **both** models: `train_cnn.py` reads the
coefficients straight out of the bitstream, `train_rgb_cnn.py` decodes the same
file to pixels. The reverse is impossible — hardware coefficients cannot be
recovered from a software re-encode.

Captured frames are already 160×120 at 4:2:2, so they need no resize or
re-encode, and must not be given one.

The board serves one viewer, so an open browser tab holds the stream slot and
every `/frame` request times out. The tool calls `/claim` first to take it.

> A cheaper idea was tried first and **failed**: `build_data.py
> --split-by-people` labels the co-occurrence (`computer_people` /
> `computer_no_people`) instead of dropping it, using the YOLO person
> confidences already recorded for every image. The resulting model scored
> precision 0.000 / recall 0.000 on both `_people` classes — it never predicted
> them once. At a 15×20 DC grid a person inside a desk scene simply is not
> separable. The flag is kept so the result can be reproduced, not because it
> helped.

---

## CLI reference — the two training scripts

Every option, with its real default taken from the argparse definitions.
Anything with a blank default is required only in the sense that the script
supplies its own; `None` means "computed at runtime", explained in the notes.

### `train_cnn.py` — compressed domain (JPEG DCT coefficients)

| option | type | default | what it does |
|---|---|---|---|
| `--capture-width` | int | `160` | Build resolution of `data/`. Must be a multiple of 16. |
| `--capture-height` | int | `120` | Multiple of 16 under 4:2:0, multiple of 8 under 4:2:2. |
| `--chroma-subsampling` | `4:2:0` \| `4:2:2` | `4:2:2` | Matches what the OV2640 actually emits. `4:2:0` is the project's original assumption, kept only so older 96×96 configs still reproduce. |
| `--dataset-source` | str | `everyday_openimages160x120` | Source directory (relative to the project root) that `data/` is built from. |
| `--num-ac-coeffs` | int | `3` | Luma AC coefficients kept per 8×8 block **on top of** DC. `0` = DC-only. See the warning below before raising it. |
| `--num-chroma-ac-coeffs` | int | `None` → matches `--num-ac-coeffs` | Chroma AC count, independent of luma. **Set this to `0` on OV2640 hardware** (see below). |
| `--classes` | csv | `None` → all classes | Subset of class names, e.g. `people,computer,doors,fruit,car`. |
| `--epochs` | int | `60` | Float-training epochs. Early stopping patience is 12. |
| `--qat-epochs` | int | `20` | Quantization-aware fine-tuning epochs after float training. |
| `--dropout` | float | `0.3` | Dropout before the classifier head. |
| `--no-chroma` | flag | off | Drop Cb/Cr fusion entirely (luma-only ablation). |
| `--lum-channels` | int | `16` | `lum_conv` output channels. |
| `--stride2-channels` | int | `32` | `stride2_conv` output channels. |
| `--post-concat-channels` | int | `64` | `post_concat_conv` output channels. |
| `--extra-conv-channels` | csv | `32` | Extra conv stages after the concat, e.g. `64,64`. Empty string for none. |
| `--coeff-scan-order` | `zigzag` \| `axis_first` | `zigzag` | Which positions in the 8×8 block the kept coefficients come from: JPEG zig-zag, or pure horizontal/vertical frequencies first. |
| `--use-augmentation` | flag | off | **Offline** augmentation: each train image is decoded, augmented, and re-encoded to a real JPEG before coefficients are extracted — necessary because this arm reads coefficients from the bitstream rather than computing them. Val/test never augmented. |
| `--augment-copies` | int | `1` | Augmented variants per train image when `--use-augmentation` is on. |
| `--seed` | int | `1234` | Seeds Python/NumPy/Torch. Note `torch.use_deterministic_algorithms` is **not** set, so cuDNN still gives ~0.5–1.4 points of run-to-run variation. |

Output always goes to `output/cnn/` — this script has no `--artifacts-name`.

> **Coefficient budget: more is not better on real hardware.** Higher AC counts
> score *better* on the test split and *worse* on the camera, because the extra
> planes are ~93% zeros on this sensor. Chroma AC is effectively dead on the
> OV2640: coefficient 1 is nonzero in 7.2% of camera blocks against 66.5% in
> training, and coefficient 3 in none at all. Keep `--num-chroma-ac-coeffs 0`
> for anything you intend to deploy. Full measurements in
> [`python_code/README.md`](python_code/README.md).

### `train_rgb_cnn.py` — pixel domain (RGB block means)

| option | type | default | what it does |
|---|---|---|---|
| `--capture-width` | int | `160` | Build resolution of `data/`, multiple of 16. |
| `--capture-height` | int | `120` | Build resolution of `data/`. |
| `--chroma-subsampling` | `4:2:0` \| `4:2:2` | `4:2:2` | Only affects the `data/` build/reuse check — irrelevant to this model's own RGB decode. |
| `--dataset-source` | str | `everyday_openimages160x120` | As `train_cnn.py`. |
| `--rgb-block-width` | int | `5` | Average this many pixels horizontally into one input value. `5` gives the 32×24 grid this project ships. Use `8` for the exact equal-resolution DCT control (an 8×8 block mean *is* the DC coefficient, giving 20×15); `1` is full resolution. |
| `--rgb-block-height` | int | `5` | As above, vertically. |
| `--downsample-factor` | int | `1` | Resize decoded pixels down by this factor *before* block averaging. `1` = full capture resolution. |
| `--classes` | csv | `None` → all classes | Subset of class names. |
| `--conv-channels` | csv | `16,32,64` | Main conv stack. First stage is stride 1; every later stage is stride 2. The shipped model uses `32,32,64`. |
| `--extra-conv-channels` | csv | `32` | Extra stride-1 stages after the main stack. Empty string for none. |
| `--dropout` | float | `0.3` | Dropout before the classifier head. |
| `--epochs` | int | `60` | Float-training epochs. |
| `--qat-epochs` | int | `20` | QAT epochs. |
| `--artifacts-name` | str | `rgb_cnn` | Subdirectory of `output/` to write to. Change it to avoid overwriting a trained model. |
| `--data-dir` | str | `data` | Dataset to train on. Point at `data_hand_curated` for fine-tuning. **Any value other than `data` skips `ensure_dataset()`**, so a hand-built directory is never silently regenerated. |
| `--test-data-dir` | str | `None` → `--data-dir` if it has `test/`, else `data` | Where the test split comes from. Fine-tuning sets deliberately have no `test/`, so base and fine-tuned models are scored on the same untouched split. |
| `--fine-tune-from` | str | `None` | `output/` subdirectory to initialise from, e.g. `rgb_cnn`. Loads its `float_model.pt` and **refuses to run** if the manifest's architecture, block reduction, or class list/order disagree with this run. |
| `--fine-tune-lr` | float | `1e-4` | Learning rate used with `--fine-tune-from`, ~10× below the from-scratch rate. QAT afterwards still uses its own rate. |
| `--use-augmentation` | flag | off | **Live** augmentation, applied fresh to decoded pixels every epoch — no JPEG re-encode needed here, unlike the DCT arm, so the augmentation varies per epoch rather than being fixed at extraction time. Val/test never augmented. |
| `--seed` | int | `1234` | As `train_cnn.py`, same cuDNN caveat. |

Shared between both: `--epochs`, `--qat-epochs`, `--dropout`, `--seed`,
`--classes`, `--capture-*`, `--chroma-subsampling`, `--dataset-source` and
`--extra-conv-channels` mean the same thing in each. The differences that
matter are that only `train_rgb_cnn.py` has fine-tuning and `--artifacts-name`,
and only `train_cnn.py` has the coefficient options.

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
  speed_up.md                  what made inference fast, and where the headroom still is
  network_info.json            the shipped model's manifest

esp32_cam/esp32_classifier/  PlatformIO firmware -- the compressed-domain arm
  src/main.cpp                 capture, DCT extraction, inference, HTTP, streaming
  src/dct_features.{h,cpp}     the JPEG parser: pulls DCT coefficients from the
                               bitstream without ever reconstructing pixels.
                               DCT_NUM_AC_COEFFS here must match the model --
                               a static_assert in main.cpp enforces it.
  include/model_weights.h      generated: int8 weights + model_forward()
  include/test_vectors.h       generated: 25 vectors + expected logits
  include/real_test_jpegs.h    generated: 3 real camera JPEGs + expected planes,
                               a decoder check independent of the model
  scratchpad/                  gen_real_test_header.cpp regenerates the above
  lib/esp_nn/                  vendored from espressif/esp-nn

python_code/                 training, export, verification -- both arms
  train_rgb_cnn.py             pixel domain
  train_cnn.py                 compressed domain
  export_to_firmware.py        export -> verify -> back up -> install -> flash (RGB)
  export_cnn_c_weights.py      \  the DCT arm's export/verify pair, run by hand
  verify_cnn_c_export.py       /
  dct_common/                  shared library (feature extraction, QAT, quantization)

data_curation/               how the raw pull was selected and filtered
finetune_curation.md         the hand-curation -> fine-tuning workflow
stream_stall_issue.md        why the stream stalls, and why that caused resets
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
