# python_code — training the 5x5 block-mean RGB CNN

Everything that produces `esp32_cam/esp32_rgb_cnn/include/model_weights.h`:
dataset assembly, three-stage training, C export, and bit-exactness
verification against the exported C.

## Requirements

```
torch  torchinfo  numpy  pillow  jpeglib  fiftyone  ultralytics
```

`fiftyone` is only needed to download Open Images; `ultralytics` only for the
people filter. Training and export need neither. A CUDA GPU is optional —
`dct_common/device.py` picks whatever is available.

---

## 1. Dataset

```bash
python get_everyday_openimages_data.py
```

Downloads from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
via FiftyOne (plus a Places365 pass for `garden`, which has no bounding boxes).

Each saved image is a **crop around one annotated object**, not a whole photo:
the box is padded by 15%, grown to 4:3 by pulling in real neighbouring content,
then letterboxed to **160x120** — the OV2640's native `FRAMESIZE_QQVGA`. Never
a plain aspect-distorting resize. One crop per source photo — the largest
detection whose box covers between 5% and 90% of the frame.

Writes `everyday_openimages160x120/{train,val}/<class>/` at JPEG quality 95,
and a 96x96 copy alongside it. The 4:2:2 re-encode that matches the camera
happens later, in `build_data.py`.

It also records, per saved image, which *other* target classes were annotated
in the source photo, into `meta/train_intersections.json`. Nothing is dropped
at download time — filtering is a separate, inspectable step.

```bash
python detect_people_yolo.py
```

Open Images' annotations are incomplete: they box salient objects, not every
person in every photo. So this runs **YOLO11m** over the actual saved pixels
and records the maximum person-detection confidence per image, into
`meta/train_yolo_people.json`. Roughly 45% of the images eventually removed
are caught *only* by this pass, not by the annotations.

Both signals matter. Train on unfiltered data and the network learns that
people are part of what a door looks like.

```bash
python build_data.py
```

Merges fine-grained source classes into the deployed taxonomy
(`laptop` + `keyboard` + `monitor` -> `computer`, `table` + `chair` + `couch`
-> `furniture`, `bowls` + `plates` + `cups` -> `crockery`; the merge map and the
drop list are edit-the-script config near the top of `build_data.py`, not CLI
flags), drops images
caught by either people filter, carves out a test split, and writes
`data/{train,val,test}/<class>/*.jpg`. It also rewrites
`dct_common/class_names.json` to match what it built — pass
`--no-write-class-names` to leave that alone.

Useful flags: `--no-filter` (merge only, keep everything),
`--no-yolo-filter` (annotations only), `--test-fraction` (default 0.15),
`--seed` (default 1234).

`filter_intersections.py` runs the filtering step standalone into a separate
destination, which is the easy way to eyeball what a filter setting actually
removes before committing to it.

See [`../data_curation/`](../data_curation/) for the selection tooling.

---

## 2. Training

```bash
python train_rgb_cnn.py \
    --rgb-block-width 5 --rgb-block-height 5 \
    --classes people,computer,doors,fruit,car,furniture,garden \
    --use-augmentation
```

That writes to `output/rgb_cnn` (the default `--artifacts-name`). Use a
distinct name per configuration — `rgb_5x5_7cls`, `rgb_5x5_11cls` — if you
train several: two runs sharing a name silently overwrite each other's weights,
which is easy to do while iterating on the class list.

Three stages, in one run:

1. **Float.** Plain training, best-val checkpointing.
2. **QAT.** Fake-quantization inserted, initialized from the float weights by
   attribute name, trained at a lower learning rate.
3. **Bit-exact int8.** A pure-integer NumPy reference built from the QAT
   model, then scored against it. Agreement below ~99% means something in the
   quantization is wrong.

The reduction is the interesting knob. Block averaging and Lanczos are two
different reductions and the script refuses to combine them. Block averaging is
the one this project uses, because an 8x8 block mean is exactly the JPEG DC
coefficient divided by 8, which makes it the equal-resolution control for a DCT
classifier rather than just another downsampler.

Full flag reference with defaults: [CLI reference](#cli-reference) below.

Writes to `output/<artifacts-name>/`: `float_model.pt`, `qat_model.pt`,
`quantized_model.npz`, `model_manifest.json`, `accuracy_table.json`,
`confusion_matrix.json`.

### What the shipped weights were built with

`model_manifest.json` now records the full invocation, so this is recovered
from the artifacts rather than remembered:

```
--rgb-block-width 5 --rgb-block-height 5 --classes people,computer,doors,fruit,car,furniture,garden --use-augmentation
```

| | |
|---|---|
| classes | people, computer, doors, fruit, car, furniture, garden |
| reduction | 5x5 block mean -> 32x24 |
| architecture | conv 16,32,64 + extra 32 |
| epochs | 60 float, 20 QAT |
| augmentation | on |
| seed | 1234 |
| int8 test | 73.2% |

`--classes` sets the model's class index order, and that order is baked into
`MODEL_CLASS_NAMES` in the exported header — reorder the flag and the
firmware's labels silently stop matching its logits. `--fine-tune-from`
refuses to proceed across a mismatch for exactly this reason.

**On reproducibility.** `--seed` seeds Python, NumPy and Torch, but
`torch.use_deterministic_algorithms` is *not* set, so cuDNN's nondeterministic
backward kernels give roughly 0.5–1.4 points of run-to-run variation on
identical settings. Treat differences smaller than that between configurations
as noise. This was confirmed by retraining a published configuration and
getting 81.9% where 81.1% was recorded.

---

## 3. Export and verify

```bash
python export_to_firmware.py                 # the `rgb_cnn` run
python export_to_firmware.py --upload        # ...and flash it
python export_to_firmware.py rgb_5x5_ft --dry-run
```

That wrapper is the recommended path — see its entry in the
[CLI reference](#cli-reference). The two underlying scripts can still be run
directly:

```bash
python export_rgb_cnn_c_weights.py rgb_cnn
python verify_rgb_cnn_c_export.py  rgb_cnn
```

Both take the `output/` subdirectory name as their single argument and default
to `rgb_cnn` if omitted.

The exporter writes `output/<run>/esp32/`:

- `model_weights.h` — int8 weights, quantized multipliers, and a complete
  `model_forward()`. Carries **both** conv paths behind
  `#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)`:
  ESP-NN with OHWI weights, or portable C with OIHW. Same weights, same
  surrounding code — only the kernel and layout differ.
- `network_info.json` — the manifest the firmware and this README quote from.

The verifier writes the self-test vector headers the firmware compiles in —
`rgb_synth_vectors.h` (LCG + structured patterns) and `rgb_real_images.h` (two
real test-split images) — plus a standalone `verify_rgb_export.c` harness and
its compiled binary, into `output/<run>/`.

It re-extracts features from the real test JPEGs — using
`extract_rgb_blocks` when the manifest says `reduction: "block_mean"` — runs
them through both the Python int8 reference and the compiled C, and diffs.
This is the check that catches layout mistakes; the QAT-vs-int8 agreement
number from training does not, because both sides of that comparison are
Python.

Note the two scripts write to **different directories**. `export_to_firmware.py`
exists mainly because of that asymmetry — copying only `model_weights.h` leaves
the boot self-test comparing a new model against another model's expected
logits, which fails on device with no hint as to the cause.

---

## CLI reference

Every option in every script here, with its default. Extracted from the
`add_argument` calls themselves, not transcribed by hand.

### `train_rgb_cnn.py`

**Reduction** — how the 160x120 capture becomes the model's input grid.

| option | type | default | meaning |
|---|---|---|---|
| `--rgb-block-width` | int | `8` | average this many pixels horizontally into one input value |
| `--rgb-block-height` | int | `8` | as above, vertically |
| `--downsample-factor` | int | `1` | Lanczos resize to `(capture_w/f) x (capture_h/f)` instead; `1` = full resolution |

Two traps here. First, block averaging is **on by default** — the default 8x8
already produces a 20x15 grid, so a bare `python train_rgb_cnn.py` is not
training on full-resolution pixels. For full resolution you must pass
`--rgb-block-width 1 --rgb-block-height 1`. Second, the two reductions are
mutually exclusive and the check is `width > 1 or height > 1`, so
`--downsample-factor` is rejected outright unless both block dimensions are
explicitly set to 1.

**Dataset** — these describe how `data/` was built, i.e. the source JPEGs. They
are not model parameters.

| option | type | default | meaning |
|---|---|---|---|
| `--capture-width` | int | `160` | `data/`'s build resolution, multiple of 16 |
| `--capture-height` | int | `120` | `data/`'s build resolution |
| `--chroma-subsampling` | str | `4:2:2` | `4:2:0` or `4:2:2`; only affects `ensure_dataset()`'s build/reuse check — irrelevant to this model's own RGB decode |
| `--dataset-source` | str | `everyday_openimages160x120` | source directory to build `data/` from, relative to the project root |
| `--classes` | str | `None` (all) | comma-separated subset |

`--classes` deserves emphasis: **the order you write it in becomes the model's
class index order**, and that order is baked into `MODEL_CLASS_NAMES` in the
exported header. Reorder the flag and the firmware's labels silently stop
matching its logits while everything still builds and runs.

**Architecture**

| option | type | default | meaning |
|---|---|---|---|
| `--conv-channels` | str | `16,32,64` | main conv stack widths; first stage stride 1, every later stage stride 2 |
| `--extra-conv-channels` | str | `32` | extra stride-1 stages appended after the main stack; `""` for none |
| `--dropout` | float | `0.3` | before the final `Linear` |

**Training**

| option | type | default | meaning |
|---|---|---|---|
| `--epochs` | int | `60` | float stage |
| `--qat-epochs` | int | `20` | QAT stage |
| `--use-augmentation` | flag | **off** | live train-split augmentation, re-rolled every epoch |
| `--seed` | int | `1234` | seeds Python, NumPy and Torch |
| `--artifacts-name` | str | `rgb_cnn` | subdirectory of `output/` to write to |

**Data source and fine-tuning**

| option | type | default | meaning |
|---|---|---|---|
| `--data-dir` | str | `data` | dataset to train on. Any value but `data` **skips `ensure_dataset()`** |
| `--test-data-dir` | str | auto | test split source; falls back to `data/` when `--data-dir` has no `test/` |
| `--fine-tune-from` | str | `None` | `output/` subdirectory to initialize from instead of random weights |
| `--fine-tune-lr` | float | `1e-4` | learning rate used with `--fine-tune-from` |

`--data-dir` skipping `ensure_dataset()` for non-`data` values is deliberate,
not an oversight: a hand-curated directory would otherwise be seen as a
class/size mismatch and silently rebuilt from `--dataset-source`, throwing the
curation away.

`--fine-tune-from` checks the source run's manifest against this run's
architecture, reduction, and class list **including its order**, and refuses to
proceed on a mismatch. Order matters because the final `Linear`'s rows are
positional — a reordered class list loads cleanly, trains happily, and exports
a model whose labels do not match its logits.

See [`../finetune_curation.md`](../finetune_curation.md) for the full workflow.

`--use-augmentation` is `store_true`, so omitting it means augmentation is off;
val and test are never augmented either way. Note this flag means something
different here than in a DCT-domain trainer: this one applies flip / crop /
rotate / brightness / contrast / blur directly to decoded pixels and varies
them every epoch, because there is no JPEG round trip to preserve.

`--artifacts-name` is the one to remember. It defaults to `rgb_cnn`, so
repeated bare runs **overwrite each other**.

### `export_rgb_cnn_c_weights.py` and `verify_rgb_cnn_c_export.py`

Neither uses `argparse`. Each takes one optional positional argument:

| position | default | meaning |
|---|---|---|
| `1` | `rgb_cnn` | the `output/` subdirectory to read |

Anything beyond the first argument is ignored. The default is a real hazard —
running either script bare does not mean "the run I just did", it means
`output/rgb_cnn/`, which may be an unrelated older model. Always pass the name.

### `build_data.py`

| option | type | default | meaning |
|---|---|---|---|
| `--source` | str | `everyday_openimages160x120` | source directory, relative to the project root |
| `--capture-width` | int | `160` | multiple of 16 |
| `--capture-height` | int | `120` | multiple of 16 under 4:2:0, of 8 under 4:2:2 |
| `--chroma-subsampling` | str | `4:2:2` | `4:2:0` or `4:2:2`; the default matches the real OV2640 |
| `--no-filter` | flag | off | merge classes but keep every image — skips `meta/`-based intersection filtering |
| `--no-yolo-filter` | flag | off | keep the Open-Images-intersections filter, drop the YOLO11m person filter |
| `--no-curation-filter` | flag | off | ignore the hand-curation reject list |
| `--no-write-class-names` | flag | off | build `data/` without rewriting `dct_common/class_names.json` |
| `--test-fraction` | float | `0.15` | fraction of each class's train images carved into `data/test/` |
| `--seed` | int | `1234` | split seed |

The class merge map and the drop list are **edit-the-script config near the top
of `build_data.py`**, not CLI flags. `--no-write-class-names` is what you want
for a preview run, since the default rewrites `class_names.json` in place.

### `detect_people_yolo.py`

| option | type | default | meaning |
|---|---|---|---|
| `--source` | str | `everyday_openimages160x120` | source directory, relative to the project root |
| `--batch-size` | int | `64` | YOLO11m inference batch |

### `filter_intersections.py`

| option | type | default | meaning |
|---|---|---|---|
| `--source` | str | `everyday_openimages160x120` | source directory, relative to the project root |
| `--dest` | str | `<source>_filtered` | destination directory |
| `--no-yolo-filter` | flag | off | keep only the Open-Images-intersections filter |

**`export_to_firmware.py`** — the whole deploy in one command: export, verify,
back up, install, optionally build. Prefer it over running the two scripts by
hand, since they write to *different* directories (the exporter produces
`model_weights.h`, the **verifier** produces the two self-test vector headers)
and copying only the first leaves the device self-test comparing a new model
against another model's expected logits.

| option | type | default | meaning |
|---|---|---|---|
| `run` | positional | `rgb_cnn` | `output/` subdirectory to deploy |
| `--firmware` | str | `esp32_cam/esp32_rgb_cnn` | PlatformIO project root |
| `--skip-export` | flag | off | reuse existing generated headers |
| `--skip-verify` | flag | off | skip the bit-exactness check (don't) |
| `--force` | flag | off | install even if the firmware already matches |
| `--build` | flag | off | run `pio run` afterwards |
| `--upload` | flag | off | build and flash |
| `--dry-run` | flag | off | report only |

It refuses to install anything the verifier did not pass, checks the four
files agree with each other, and prints a per-class precision and
predicted-share table — warning when a class is predicted far more often than
it occurs, which is the over-prediction that balanced accuracy cannot see.

### Curation scripts

The hand-curation loop — full walkthrough in
[`../finetune_curation.md`](../finetune_curation.md).

**`curation_pull.py`** — seeds `data_tmp/` from `data/{train,val}` on first
run, then moves a review batch into `curation_review/`.

| option | type | default | meaning |
|---|---|---|---|
| `--data-dir` | str | `data` | untouched dataset to seed the working pool from |
| `--classes` | str | all | comma-separated subset to pull |
| `--train-count` | int | `100` | train images per class per batch |
| `--val-count` | int | `20` | val images per class per batch |
| `--seed` | int | unseeded | reproducible sample |

**`make_review_gallery.py`** — one HTML page for the whole batch with a folder
dropdown, written to `curation_review/_review.html`. No options; takes an
optional positional path (default `curation_review`). Re-run after every pull,
since a `file://` page cannot enumerate directories.

**`curation_resolve.py`** — accepted images to `data_hand_curated/`, rejected
to `data_rejected/`, rejected filenames into the source `meta/`.

| option | type | default | meaning |
|---|---|---|---|
| `--source` | str | `everyday_openimages160x120` | dataset whose `meta/` holds the durable record |
| `--dry-run` | flag | off | report only, change nothing |

**`apply_curation_rejects.py`** — folds gallery exports into the same metadata
directly, for reviewing outside the loop.

| option | type | default | meaning |
|---|---|---|---|
| `lists` | positional | — | `rejected_*.txt` file(s), or a directory of them |
| `--source` | str | `everyday_openimages160x120` | dataset to resolve filenames against |
| `--split` | str | from filename | force the source split |
| `--remove` | flag | off | un-reject instead of reject |
| `--dry-run` | flag | off | report only |

**`make_gallery.py`** — the original one-directory contact sheet. Takes image
directories as positional arguments, no flags.

### `get_everyday_openimages_data.py`

No CLI options at all — everything is module-level constants near the top of
the file. The ones worth knowing before a download:

| constant | default | meaning |
|---|---|---|
| `RESOLUTIONS` | `[160x120, 96x96]` | output sizes written |
| `TRAIN_PER_CLASS` | `1000` | target saved images per class, train |
| `VAL_PER_CLASS` | `100` | target saved images per class, val |
| `DOWNLOAD_MULTIPLIER` | `4.0` | over-fetch factor, since many candidates fail the area filters |
| `MIN_OBJECT_AREA` / `MAX_OBJECT_AREA` | `0.05` / `0.90` | reject boxes too small or too frame-filling |
| `CROP_PADDING` | `0.15` | padding around the box |
| `JPEG_QUALITY` | `95` | re-encode quality |
| `RANDOM_SEED` | `42` | selection seed |
| `VAL_FRACTION` | `0.10` | `garden` only — Places365 has no separate val split here |
| `CLASS_MAP` | — | Open Images label -> class name |

---

## `dct_common/`

Shared library, imported by everything above.

| module | |
|---|---|
| `config.py` | `Config` dataclass + `CLASS_NAMES`, read from `class_names.json` |
| `dataset.py` | `ensure_dataset()` — builds/validates `data/` at the requested capture size |
| `rgb_features.py` | `extract_rgb_pixels()`, `extract_rgb_blocks()` — the block mean lives here |
| `features.py`, `zigzag.py` | DCT coefficient extraction (used by `splits.py`; not on the RGB path) |
| `splits.py` | `build_rgb_block_split()` and friends — assembles train/val/test arrays |
| `augmentation.py` | `augment_image()`, behind `--use-augmentation` |
| `models/rgb_cnn.py` | `RgbCnnClassifier`, its QAT twin, and `Int8RgbCnnReference` |
| `qat.py` | fake-quantization modules |
| `quantization.py` | `quantize_multipliers()`, `ragged_object_array()` |
| `device.py` | device selection, `benchmark_cpu_inference()` |
| `metrics.py` | accuracy tables, confusion matrices, classification reports |
| `seeding.py` | `set_seed()` |

One trap worth knowing: build ragged per-layer weight arrays with
`ragged_object_array()` from `quantization.py`, never
`np.array(..., dtype=object)` directly. NumPy silently builds a rectangular
array instead of a ragged one whenever the per-layer shapes happen to share a
leading dimension, and the failure surfaces far from its cause.

`extract_rgb_blocks()` rounds half-up with `np.rint` after averaging in
floating point. The firmware's `rgb565_to_planar_rgb888()` expands to 8 bits
*first*, then averages, then rounds the same way — the two have to agree or
the deployed model sees inputs the trained model never saw.
