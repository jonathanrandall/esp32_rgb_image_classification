# esp32_rgb_image_classification

Real-time image classification on an ESP32-S3 camera board, in the **pixel
domain**: the camera is read as raw RGB565, reduced to a 32x24 grid by
averaging 5x5 blocks of pixels, and classified by a small int8 CNN running
under [ESP-NN](https://github.com/espressif/esp-nn).

Five classes: `computer`, `fruit`, `people`, `doors`, `car`.

**81.6% int8 test accuracy on 2,304 input values per frame**, 79.1% balanced. Everything here
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

## Hardware

- I used the **Freenove ESP32-S3 CAM** board. **This is not recommended.** At
  the time I bought this board, it shipped with the OV2640. It now ships with
  the **GC0308** camera, which does not capture JPEG on hardware. The GC0308
  will work with the pixel-domain arm but not the compressed-domain arm.
- PSRAM required.
- Capture is `PIXFORMAT_RGB565` at `FRAMESIZE_QQVGA` (160x120), set **at
  `esp_camera_init()` time**. The "init at QVGA, then `set_framesize()` down"
  trick is JPEG-only — doing it with RGB565 sizes the framebuffer for the
  wrong format and you get a `Stack canary watchpoint triggered (cam_task)`
  panic. This is noted in `src/main.cpp` where it bites.

### Check your camera module: the DCT arm needs an OV2640

Boards sold as "ESP32-S3 CAM" do not all carry the same sensor, and the two
arms of this project do not have the same requirement:

| | needs | why |
|---|---|---|
| **DCT arm** (`esp32_classifier`) | **OV2640 only** | it classifies the sensor's **hardware JPEG** DCT coefficients. A sensor with no JPEG encoder produces no coefficients to read |
| **RGB arm** (`esp32_rgb_cnn`) | any sensor doing RGB565 at 160x120 | it reads raw pixels |

So **buy or check for an OV2640** if the compressed-domain arm is what you
came for. Substituting another sensor is not a matter of settings — there is
nothing to decode.

A **GC0308** is the module most likely to turn up in its place (a VGA sensor,
no JPEG encoder). With one fitted, the DCT firmware stops during `setup()`
like this:

```
E (794) camera: JPEG format is not supported on this sensor
Camera init failed with error 0x106
Camera init failed, halting.
```

`0x106` is `ESP_ERR_NOT_SUPPORTED`. Note where that halt sits: **before**
`connect_wifi()`, so the board never joins the network and never answers
`/status`. The symptom you actually see is "the board will not connect to
Wi-Fi", which sends you looking in entirely the wrong place — the camera is
working fine and the credentials are fine.

Two things make this harder to diagnose than it should be, so check them
first:

- `init_camera()` calls `esp_log_level_set("cam_hal"/"camera", ESP_LOG_NONE)`
  deliberately (the driver logging from `cam_task` overflows its stack — see
  `stream_stall_issue.md`), which also silences the one line that names the
  problem. Raise those to `ESP_LOG_VERBOSE` temporarily to see it.
- On this board `Serial` reaches USB only because `esp32_classifier`'s
  `platformio.ini` sets `-DARDUINO_USB_CDC_ON_BOOT=1`. `esp32_rgb_cnn` does
  not, so the same failure there is completely silent on `/dev/ttyACM0`.

To identify the fitted sensor, print it after a successful `esp_camera_init()`
(use `PIXFORMAT_RGB565`, which non-JPEG sensors do support):

```c
sensor_t *s = esp_camera_sensor_get();
camera_sensor_info_t *si = esp_camera_sensor_get_info(&s->id);
Serial.printf("PID=0x%04x %s supports_jpeg=%d\n", s->id.PID, si->name, si->support_jpeg);
// OV2640 -> PID=0x0026 ... supports_jpeg=1
// GC0308 -> PID=0x009b ... supports_jpeg=0
```

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
| train | 83.6% | 85.1% | 86.1% |
| val | 81.2% | 82.2% | 82.7% |
| test | 81.0% | 81.7% | **81.6%** |

Balanced (macro-averaged per-class recall), which weights every class equally:
**79.1%** on test.

Per class, on the test split — precision alongside predicted share, because
balanced accuracy is macro *recall* and is blind to a class that over-fires:

| class | recall | precision | predicted share | actual share |
|---|---:|---:|---:|---:|
| computer | 91.7% | 83.1% | 41.6% | 37.7% |
| fruit | 70.0% | 83.1% | 13.6% | 16.2% |
| people | 64.4% | 73.4% | 14.8% | 16.9% |
| doors | 86.5% | 85.9% | 16.4% | 16.3% |
| car | 83.0% | 79.5% | 13.5% | 12.9% |

The compressed-domain arm on the same five classes and the same data:
**78.8%** int8, 77.5% balanced.

int8-vs-QAT prediction agreement: **99.5%**. The int8 reference is bit-exact
against the C export — see *Verification* below.

QAT scoring above float is not a typo; quantization noise acts as a
regularizer on a model this small.

Measured on the board via `GET /status`, which reports per-frame timing and a
per-layer breakdown. **Both block sizes are listed, because the choice of
block is also a choice of speed** — it sets the model's input resolution:

| stage | 5x5 blocks (shipped) | 8x8 blocks |
|---|---:|---:|
| RGB565 capture | 0-3 | 0-3 |
| RGB565 -> block-mean convert | 3.4 | 3.5 |
| **inference** | **30.7** | **21.0** |
| JPEG encode (for the preview stream) | 18.2 | 18.8 |
| total | 52.3 | 43.5 |
| **end to end** | **16.9 fps** | **19.3 fps** |

The inference difference is input size, not architecture — the two runs use
the same layers. 5x5 blocks give a 32x24 grid (768 positions), 8x8 give 20x15
(300), so 2.6x the positions costs 1.5x the inference time. The shipped model
uses 5x5 because the accuracy is worth the 2.4 fps; 8x8 is the configuration
to use for the comparison against the DCT arm, where it is the exact
equal-resolution control (an 8x8 block mean *is* the DCT DC coefficient).

Note that the JPEG encode is nearly as expensive as the inference at 8x8, and
still a third of the frame at 5x5: capturing RGB565 means the frame cannot
double as the stream, so previewing costs a software encode. That is a
property of the pixel-domain pipeline, not of the model — and it is exactly
the cost the compressed-domain arm does not pay.

### The same measurement on the compressed-domain arm

Read from `GET /status` on `esp32_classifier`, same board, same five classes,
while a client is streaming:

| stage | ms |
|---|---:|
| capture | 11.5 |
| DCT parse | 1.3 |
| **inference** | **19.9** |
| total | 32.6 |
| **end to end** | **27.4 fps** (25.8-27.8, occasionally 28) |

**There is no JPEG encode line, and that is the entire point.** The camera
already produced a JPEG, the model reads its coefficients directly, and the
same bytes go out as the preview stream. The pixel arm has to capture RGB565
*and* then encode a JPEG purely so that something can be displayed — 18.2 ms
of its 52.3 ms frame doing work the compressed-domain arm never does.

Side by side, at the shipped settings:

| | RGB (5x5) | DCT |
|---|---:|---:|
| inference | 30.7 ms | 19.9 ms |
| end to end | 52.3 ms | 32.6 ms |
| **frame rate** | **16.9 fps** | **27.4 fps** |
| int8 test accuracy | **81.6%** | 78.8% |

That is the trade this repository exists to measure: the pixel arm is about
three points more accurate, the compressed arm about 1.6x the throughput. At the
equal-resolution setting (`--rgb-block-width 8`, where a block mean *is* the
DC coefficient) the pixel arm runs 19.3 fps against the same 27.4.

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
    --classes computer,fruit,people,doors,car \
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
    --classes computer,fruit,people,doors,car --use-augmentation

python export_cnn_c_weights.py cnn     # -> output/cnn/model_weights.h
python verify_cnn_c_export.py cnn      # expect 25/25 bit-exact
```

Then copy `output/cnn/{model_weights.h,test_vectors.h}` into
`esp32_cam/esp32_classifier/include/` and set `DCT_NUM_AC_COEFFS` in
`src/dct_features.h` to match — a `static_assert` in `main.cpp` fails the build
if they disagree.

## Capturing frames from the board (`capture_board_frames.py`)

Both models are trained on Open Images photographs, and the camera produces
something meaningfully different — a whole room rather than a padded crop
around one object, indoor lighting, the OV2640's own exposure, and the sensor's
own JPEG quantization tables. A model can score well on the test split and
behave poorly on the board without either number being wrong. This tool closes
that gap by collecting training data through the camera itself.

The sharpest case: the dataset pipeline **drops person-containing images from
every class except `people`**, so the model has never seen a person and a
computer in one frame — exactly what a camera pointed at a desk shows. The test
split cannot reveal this, having been filtered the same way.

### Running it

Flash `esp32_cam/esp32_classifier` and note the board's address (it answers to
`esp32cam_dct.local`; `--host` takes an IP if mDNS is not working). Then, from
`python_code/`:

```bash
# headless — capture 200 frames blind, ~1s apart
python capture_board_frames.py --label people --count 200 --interval 1.0 --show-prediction

# interactive — a live preview window; SPACE saves the frame on screen
python capture_board_frames.py --interactive --label people

# a board that mDNS cannot find
python capture_board_frames.py --host 192.168.1.125 --label car --count 50
```

| option | default | what it does |
|---|---|---|
| `--label` | *(required)* | class name; the output subdirectory and filename prefix |
| `--count` | `100` | frames to capture (headless only) |
| `--interval` | `0.5` | seconds between captures, and the auto-capture period in the window |
| `--host` | `esp32cam_dct.local` | board hostname or IP |
| `--port` | `81` | stream port, where `/frame` and `/stream` live |
| `--out-dir` | `../board_captures` | output root |
| `--timeout` | `10.0` | per-request timeout, seconds |
| `--show-prediction` | off | print the board's own top class per frame (headless) |
| `--interactive` | off | open the preview window instead of capturing blind |
| `--labels` | *(from the board)* | extra classes for the dropdown, comma-separated |
| `--scale` | `4` | preview magnification — 160×120 shown at 640×480 |
| `--status-interval` | `0.4` | seconds between `/status` polls, the source of the live prediction |

Frames land in `board_captures/<label>/<label>_NNNNNN.jpg`, numbered
continuously across runs so several sessions — different rooms, different light
— accumulate into one class.

### The interactive window

```bash
python capture_board_frames.py --interactive --label people
```

Shows the live stream, the board's current prediction, and a per-class tally.

| key | what it does |
|---|---|
| `SPACE` | save the frame on screen |
| `A` | toggle auto-capture every `--interval` seconds |
| `U` | undo — delete the frame just saved |
| `1`–`9` | switch class, same as picking it from the dropdown |
| `C` | re-claim the stream slot (a browser tab took it) |
| `Q` / `Esc` | quit |

The class is a **dropdown** that can be changed mid-session, and it is
editable: type a name that is not in the list, press Enter, and it is added —
which is what you want when collecting data for a class the model does not have
yet. Leave `--labels` off and the list fills itself from the class names the
*flashed model* reports in `/status`, so it always matches what is running.

The prediction turns **orange when it disagrees** with the class being saved
as. Frames the model gets wrong are the highest-value data in this pipeline,
and that makes them capturable on sight rather than findable afterwards by
reading log output.

While the dropdown has keyboard focus it is a text field, so `SPACE`, `A`, `U`
and `Q` type characters instead of firing; `Esc`, or picking an entry, hands
focus back.

It needs tkinter (`sudo apt install python3-tk` on Linux; bundled with CPython
elsewhere) and Pillow. Both are imported only when `--interactive` is passed,
so the headless path still runs on a stdlib-only Python.

### Folding captures into the dataset

```bash
for f in ../board_captures/people/*.jpg; do
    cp "$f" "../data/train/people/board_$(basename "$f")"
done
```

Then retrain normally. Two things to get right:

- **Use a distinct prefix.** Capture filenames collide with Open Images ones —
  73 of the first 93 did here.
- **Copy, never move.** `board_captures/` is the only copy and is not
  reproducible, while `data/` is deleted and rebuilt by `build_data.py`.

Vary the scene while capturing. Near-duplicate frames of one static setup
inflate the count without adding information.

### Why it works the way it does

**Capture from `esp32_classifier` only** — it is the one firmware serving
`/frame`, and the only one whose JPEGs come from the OV2640's *hardware*
encoder. `esp32_rgb_cnn` captures RGB565 and software-encodes its preview with
`frame2jpg()`, so its frames carry the software encoder's quantization tables
instead of the sensor's. Requesting `/frame` from it exits with that
explanation rather than a bare 404.

That single capture set feeds **both** models: `train_cnn.py` reads the
coefficients straight out of the bitstream, `train_rgb_cnn.py` decodes the same
file to pixels. The reverse is impossible — hardware coefficients cannot be
recovered from a software re-encode.

Captured frames are already 160×120 at 4:2:2, so they need no resize or
re-encode, and **must not be given one**. This is why the preview window is a
native tkinter window rather than an HTML page like the curation galleries:
saving a frame from a browser canvas re-encodes it, replacing the OV2640's
quantization tables with the browser's and destroying the one property that
makes a board capture worth more than an Open Images photograph.
`Image.open(...).save(...)` is the same trap in Python, so PIL here draws the
preview and never touches what is written — the bytes go from the socket
straight to the file. Both modes are identical in that respect; they differ
only in how they reach the board (headless polls the bounded `/frame`,
interactive holds one `/stream` connection).

The board serves **one** viewer, so an open browser tab — or a stale socket
from a closed one — holds the stream slot and every request times out, which
looks exactly like a dead board. Both modes call `/claim` on port 80 first, and
again after any failure, to take it back.

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
| `--classes` | csv | `None` → all classes | Subset of class names, e.g. `computer,fruit,people,doors,car`. |
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

### Filtering: three independent passes

Before training, the dataset build can drop images three different ways. They
are **separate flags and none implies another** — worth knowing, because only
the third involves anything not in this repository.

| pass | flag to disable | what it drops | needs anything not in this repo? |
|---|---|---|---|
| **intersections** | `--no-filter` | images whose Open Images annotations list another target class — a person annotated in a `car` photo, a chair in a `table` photo | **no** — the metadata is written by `get_everyday_openimages_data.py` |
| **YOLO people** | `--no-yolo-filter` | images where a YOLO11m pass detects a person at ≥0.25 confidence, in every class except `people`. Catches people Open Images never annotated | **no** — written by `detect_people_yolo.py`; `ultralytics` fetches the weights |
| **hand curation** | `--no-curation-filter` | filenames listed in `meta/{split}_curation_rejects.json` — images judged bad by eye | **no**, but the *list* is not shipped; see below |

The first two are fully automatic and reproduce from a clean clone. Run them
and train, and nothing else is required.

**The hand-curation reject lists are deliberately not in this repository.**
They are one person's judgements about ~193 specific Open Images files, and
they are meaningless without the image set those filenames refer to — which
is also not shipped, because it is far too large. Their absence is handled,
not an error: `load_curation_rejects()` treats a missing file as *"nothing
hand-rejected yet"*, so `build_data.py` runs normally with that pass a no-op.
`curation_resolve.py` **creates** the file the first time you reject anything,
so the workflow below works from an empty start and accumulates your own
judgements rather than inheriting someone else's.

One honest consequence: a dataset built from a clean clone contains ~112 more
images (across the five deployed classes) than the one that produced the
shipped weights, so retraining reproduces them only to within the run-to-run
noise floor, not exactly. That difference is smaller than the noise floor
itself — see the reproducibility caveat below.

### Using your own classes

Nothing here is tied to the five classes this project ships. To pick your own:

1. Edit `CLASS_MAP` in `get_everyday_openimages_data.py` — the Open Images
   V7 class names to pull, and how many per class (`TRAIN_PER_CLASS`,
   default 1000).
2. Edit `CLASS_MERGE_MAP` and `CLASS_DROP` in `build_data.py` if you want
   several source classes combined into one label (this project merges
   `laptop + keyboard + monitor → computer`) or dropped entirely. Both are
   edit-the-script config rather than CLI flags, deliberately — they are
   dataset definitions, not per-run options.
3. Run the pipeline. `build_data.py` writes the resolved class list to
   `dct_common/class_names.json`, which every training script reads, so the
   taxonomy stays in sync automatically.
4. Pass the subset you want to train on with `--classes`.

The full path, with no dependency on anything outside this repository beyond
pip packages (`fiftyone`, `ultralytics`, `torch`, `jpeglib`, `numpy`, `tqdm`):

```bash
cd python_code
python get_everyday_openimages_data.py   # pull from Open Images V7 via FiftyOne
python detect_people_yolo.py             # YOLO11m person pass over the saved pixels
python build_data.py                     # merge + filter + build data/{train,val,test}
python train_rgb_cnn.py --classes <your,classes> --use-augmentation
```

### Hand curation and fine-tuning

Optional, and independent of the two automatic filters above. Train the base
model on all of `data/` untouched, then hand-curate a subset and fine-tune on
it. Batch pull, a single review page with a folder dropdown, and a reject
record that survives a `build_data.py` rebuild:

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
  capture_board_frames.py      collect training frames through the camera itself
  capture_interactive.py         the --interactive preview window
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
