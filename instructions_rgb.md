# End to end: the pixel-domain (RGB) model

Dataset → training → C export → firmware → board, for the arm that reads
the camera as **raw RGB565** and reduces it to a 32×24 grid by averaging
5×5 pixel blocks. Firmware: `esp32_cam/esp32_rgb_cnn/`.

> **Read the README files too.** This is the command sequence; the READMEs
> carry the reasoning, the full option tables and the failure modes.
> [`README.md`](README.md) covers why block means rather than resizing, the
> three filtering passes, using your own classes, the verification tiers and
> streaming reliability; [`python_code/README.md`](python_code/README.md)
> has the complete CLI reference with every option and its default.

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

Identical to the DCT arm — the two share one dataset.

```bash
cd python_code
python get_everyday_openimages_data.py   # pull from Open Images V7 via FiftyOne
python detect_people_yolo.py             # YOLO11m person pass over the saved pixels
python build_data.py                     # merge + filter -> ../data/{train,val,test}
```

For your own classes, edit `CLASS_MAP` in
`get_everyday_openimages_data.py` and `CLASS_MERGE_MAP` / `CLASS_DROP` in
`build_data.py` first. See the README's "Using your own classes".

---

## 2. Train

```bash
python train_rgb_cnn.py \
    --classes people,computer,doors,fruit,car \
    --conv-channels 32,32,64 \
    --use-augmentation
```

`--rgb-block-width` and `--rgb-block-height` default to **5**, so the 5×5
reduction is what you get without passing anything. Artifacts go to
`output/rgb_cnn/` (change with `--artifacts-name`, which this script has and
`train_cnn.py` does not).

Two notes:

- **Block means are already int8.** Centred to `[-128, 127]`, so the input
  quantizer's scale is 1.0 and nothing needs calibrating.
- **`--rgb-block-width 8 --rgb-block-height 8` is a meaningful special
  case**, not just a smaller input. An 8×8 block mean *is* the DCT DC
  coefficient (`DC = 8 × mean` under the orthonormal DCT-II), so 8×8 makes
  this the exact equal-resolution control for the DCT arm. Use it for the
  comparison; use 5×5 for deployment.

---

## 3. Export, verify, install, build — one command

```bash
python export_to_firmware.py rgb_cnn --build
```

That runs all four steps in order and **refuses to install anything the
verifier did not pass**. It backs up the headers it replaces, cross-checks
that the installed files agree with each other, and prints per-class
precision alongside predicted-vs-actual share with an over-prediction
warning.

Add `--upload` to flash in the same step, or `--dry-run` to see what would
change without touching anything.

Doing it by hand is easy to get wrong, which is why the script exists: the
*exporter* writes `model_weights.h`, but the **verifier** writes the two
self-test vector headers. Copy only the first and the boot self-test
compares your new model against a different model's expected logits. For
reference, the four files it installs are:

| from `output/rgb_cnn/` | to `esp32_cam/esp32_rgb_cnn/` |
|---|---|
| `esp32/model_weights.h` | `include/model_weights.h` |
| `rgb_synth_vectors.h` | `include/rgb_synth_vectors.h` |
| `rgb_real_images.h` | `include/rgb_real_images.h` |
| `esp32/network_info.json` | `network_info.json` |

`include/rgb_test_vectors.h` is **hand-written** — the harness that runs the
three tiers. It is not generated and must not be overwritten.

---

## 4. Wi-Fi credentials

Not in this repository. Create
`esp32_cam/esp32_rgb_cnn/include/secrets.h`:

```c
#pragma once
#define WIFI_SSID     "your-network"
#define WIFI_PASSWORD "your-password"
```

---

## 5. Build and flash

```bash
cd esp32_cam/esp32_rgb_cnn
pio run -t upload
```

(`export_to_firmware.py --upload` does this for you.)

---

## 6. Check it on the board

Open `http://esp32cam.local/` (note: the DCT firmware answers to
`esp32cam_dct.local` — different name, so both can be on the network at
once), then read `/status`:

| field | what it should say |
|---|---|
| `self_test` | `bit_exact_all` true, `max_logit_diff` 0, `determinism_ok` true |
| `byte_order` | measures RGB565 byte order both ways — on the Freenove board it is **not** byte-swapped |
| `buffers` | where each activation buffer was allocated |
| `reset_reason` | `poweron` |

Endpoints: `/` (UI), `:81/stream` (MJPEG), `/status`, `/config`, `/claim`.

Expected timing: RGB565 → block-mean convert ~3.5 ms, inference ~21 ms,
JPEG encode ~18.8 ms, total ~43.5 ms (19.3 fps). The JPEG encode is nearly
as expensive as the inference because capturing RGB565 means the frame
cannot double as the preview stream — a property of the pixel-domain
pipeline, not of the model.

---

## 7. Optional: real board frames

Capture from the **DCT firmware**, not this one. `esp32_rgb_cnn` has no
`/frame` endpoint, and it software-encodes its preview from RGB565 with
`frame2jpg()` — those JPEGs carry the software encoder's quantization
tables rather than the sensor's, making them worse for RGB training and
wrong for DCT training.

Captures from `esp32_classifier` serve both arms: `train_rgb_cnn.py`
decodes them to pixels. See [`instructions_dct.md`](instructions_dct.md)
step 8 for the procedure, including the filename-collision trap.

---

## 8. Optional: hand curation and fine-tuning

Independent of the automatic filters, and entirely optional — the reject
lists are not shipped, and a missing list is treated as "nothing rejected
yet", so the pipeline runs normally without it.

```bash
python curation_pull.py          # pull a batch to review
python make_review_gallery.py    # one page, folder dropdown
python curation_resolve.py       # accept/reject, recorded durably

python train_rgb_cnn.py \
    --data-dir data_hand_curated --fine-tune-from rgb_cnn \
    --classes <same list and order as the base run> \
    --artifacts-name rgb_cnn_ft
```

Full walkthrough: [`finetune_curation.md`](finetune_curation.md).

`--fine-tune-from` refuses to run if the base run's architecture, block
reduction, or class list and order disagree with this one. The fine-tuning
set deliberately has no `test/` split: base and fine-tuned models must be
scored on the same untouched split for the comparison to mean anything.
