# End to end: the compressed-domain (DCT) model

Dataset → training → C export → firmware → board, for the arm that
classifies **JPEG DCT coefficients directly**, without ever reconstructing
pixels. Firmware: `esp32_cam/esp32_classifier/`.

> **Read the README files too.** This is the command sequence; the READMEs
> carry the reasoning, the full option tables and the failure modes.
> [`README.md`](README.md) covers the architecture, the three filtering
> passes, using your own classes, and streaming reliability;
> [`python_code/README.md`](python_code/README.md) has the complete CLI
> reference with every option and its default, plus the coefficient-budget
> measurements you should read before changing `--num-ac-coeffs`.

---

## 0. Prerequisites

Python 3.10+, and [PlatformIO](https://platformio.org/) for the firmware.
A GPU for training; verification runs anywhere.

```bash
pip install torch numpy tqdm fiftyone ultralytics jpeglib pillow
```

Nothing outside this repository is required. The dataset is downloaded in
step 1; the YOLO weights are fetched by `ultralytics` in step 2.

Hardware: **Freenove ESP32-S3-WROOM CAM** (OV2640), PSRAM required.

---

## 1. Build the dataset

```bash
cd python_code
python get_everyday_openimages_data.py   # pull from Open Images V7 via FiftyOne
python detect_people_yolo.py             # YOLO11m person pass over the saved pixels
python build_data.py                     # merge + filter -> ../data/{train,val,test}
```

To use **your own classes**, edit `CLASS_MAP` in
`get_everyday_openimages_data.py` and `CLASS_MERGE_MAP` / `CLASS_DROP` in
`build_data.py` before running. `build_data.py` writes the resolved class
list to `dct_common/class_names.json`, which every training script reads.
See the README's "Using your own classes".

`build_data.py` is the **only** thing that can produce `data/`. It merges
source classes into the final taxonomy (`laptop + keyboard + monitor →
computer`), and those merged names do not exist in the raw download.

---

## 2. Train

```bash
python train_cnn.py \
    --capture-width 160 --capture-height 120 --chroma-subsampling 4:2:2 \
    --num-ac-coeffs 3 --num-chroma-ac-coeffs 0 \
    --extra-conv-channels 32 \
    --classes people,computer,doors,fruit,car,garden \
    --use-augmentation
```

Float → QAT → bit-exact int8 reference, then artifacts to `output/cnn/`.

Three things to know:

- **`train_cnn.py` has no `--artifacts-name`.** Every run overwrites
  `output/cnn/`. Copy the directory first if you want to keep a run.
- **`--num-chroma-ac-coeffs 0` is deliberate.** On the OV2640 chroma AC is
  effectively dead — coefficient 1 is nonzero in 7.2% of camera blocks
  against 66.5% in training. Raising it improves the test split and hurts
  the camera. Same caution applies to `--num-ac-coeffs`; see
  `python_code/README.md`, "Choosing the coefficient budget".
- The run is seeded but `torch.use_deterministic_algorithms` is not set, so
  expect 0.5–1.4 points of run-to-run variation. Smaller differences are
  noise.

---

## 3. Export and verify

**`train_cnn.py` does not write the C headers.** After training,
`output/cnn/` holds a fresh manifest next to stale `.h` files. Copying them
at this point ships the *previous* model with the new paperwork — run both
steps:

```bash
python export_cnn_c_weights.py cnn     # -> output/cnn/model_weights.h (+ output/cnn/esp32/)
python verify_cnn_c_export.py cnn      # -> output/cnn/test_vectors.h ; expect 25/25 bit-exact
```

Do not install anything if the verifier does not print
`VERIFICATION PASSED`.

---

## 4. Install into the firmware

```bash
cp output/cnn/model_weights.h        ../esp32_cam/esp32_classifier/include/
cp output/cnn/test_vectors.h         ../esp32_cam/esp32_classifier/include/
cp output/cnn/esp32/network_info.json ../esp32_cam/esp32_classifier/include/
```

Then make the hand-written coefficient count match the model, in
`esp32_cam/esp32_classifier/src/dct_features.h`:

```c
#define DCT_NUM_AC_COEFFS 3   // must equal --num-ac-coeffs from step 2
```

A `static_assert` in `main.cpp` fails the build if these disagree, so a
mismatch is caught at compile time rather than becoming silent garbage on
the board.

### If you changed `--num-ac-coeffs`

`include/real_test_jpegs.h` holds three real camera JPEGs with the DCT
planes the decoder should produce for them. It is baked at one coefficient
count, and `main.cpp` gates the decoder self-test on that number. Regenerate
it and update the gate, or that test silently reports `0/0`:

```bash
cd ../esp32_cam/esp32_classifier
g++ -O2 -I src -I scratchpad -o /tmp/gen \
    scratchpad/gen_real_test_header.cpp src/dct_features.cpp
/tmp/gen > include/real_test_jpegs.h
```

Then set the count in `main.cpp`'s `REAL_TEST_JPEGS_MATCH` to match.

Sanity check on a regeneration: only the AC planes should change. The DC and
chroma planes must come out **byte-identical**, since only the coefficient
count moved.

---

## 5. Wi-Fi credentials

Not in this repository. Create
`esp32_cam/esp32_classifier/include/secrets.h`:

```c
#pragma once
#define WIFI_SSID     "your-network"
#define WIFI_PASSWORD "your-password"
```

---

## 6. Build and flash

```bash
cd esp32_cam/esp32_classifier
pio run -t upload
```

---

## 7. Check it on the board

Open `http://esp32cam_dct.local/` (or the IP from the serial log), then read
`http://esp32cam_dct.local/status`:

| field | what it should say |
|---|---|
| `model_forward_self_test` | `class_match` equal to `num_vectors` (25/25). `bit_exact` is often lower — ESP-NN's accelerated conv is not bit-exact against the portable reference, and a `max_abs_logit_diff` of 1 is normal |
| `decoder_self_test` | `3/3`. `0/0` means the header above does not match `DCT_NUM_AC_COEFFS` and the test was skipped |
| `reset_reason` | `poweron`. Read this before suspecting power supplies — it distinguishes `panic` from `brownout` |
| `t_infer_ms` | ~19 ms for the shipped configuration |

Endpoints: `/` (UI), `:81/stream` (MJPEG), `:81/frame` (one frame + status),
`/status`, `/claim`, `/sendlog`, `/control`.

---

## 8. Optional: capture real frames and retrain

The dataset is Open Images photographs; the camera sees a room, its own
exposure, and its own JPEG quantization tables. Closing that gap measurably
improved on-camera behaviour here.

```bash
cd python_code
python capture_board_frames.py --label people --count 200 --interval 1.0 --show-prediction
```

Frames land in `board_captures/<label>/` at 160×120 4:2:2 — already the
training format, so no resize and no re-encode (re-encoding would replace
the sensor's quantization tables, which is the thing worth having).

**Capture from this firmware, not the RGB one.** `esp32_classifier` is the
only one serving `/frame`, and the only one whose JPEGs come from the
OV2640's hardware encoder. One capture set feeds both arms: `train_cnn.py`
reads the coefficients from the bitstream, `train_rgb_cnn.py` decodes the
same file to pixels. The reverse is impossible.

To fold them in, copy into `data/train/<class>/` **with a distinct prefix** —
capture filenames collide with Open Images ones:

```bash
for f in ../board_captures/people/*.jpg; do
    cp "$f" "../data/train/people/board_$(basename "$f")"
done
```

Copy, never move: `board_captures/` is the only copy and is not
reproducible, while `data/` is deleted and rebuilt by `build_data.py`.
Then retrain from step 2.

`--show-prediction` prints what the board currently thinks each frame is,
which is the fastest way to find the live scenes the model gets wrong.
