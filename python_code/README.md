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
via FiftyOne, resizes each image (never crops) to **160x120** — the OV2640's
native `FRAMESIZE_QQVGA` — and re-encodes as JPEG with **4:2:2** chroma
subsampling to match what the camera actually emits. Writes
`everyday_openimages160x120/{train,val}/<class>/`.

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
(`laptop` + `keyboard` + `monitor` -> `computer`; the merge map is
edit-the-script config near the top of `build_data.py`, not a CLI flag), drops images
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
    --classes people,computer,doors,fruit,car \
    --artifacts-name rgb_cnn_5x5
```

Three stages, in one run:

1. **Float.** Plain training, best-val checkpointing.
2. **QAT.** Fake-quantization inserted, initialized from the float weights by
   attribute name, trained at a lower learning rate.
3. **Bit-exact int8.** A pure-integer NumPy reference built from the QAT
   model, then scored against it. Agreement below ~99% means something in the
   quantization is wrong.

The reduction is the interesting knob:

| flag | meaning |
|---|---|
| `--rgb-block-width` / `--rgb-block-height` | average `w x h` pixel blocks down to one input value (default 8) |
| `--downsample-factor` | Lanczos resize instead of block averaging — mutually exclusive with the two above |
| `--classes` | comma-separated subset; **order here sets the model's class index order** |
| `--artifacts-name` | output directory under `output/` (default `rgb_cnn`) |

Block averaging and Lanczos are two different reductions and the script
refuses to combine them. Block averaging is the one this project uses, because
an 8x8 block mean is exactly the JPEG DC coefficient divided by 8, which makes
it the equal-resolution control for a DCT classifier rather than just another
downsampler.

Architecture flags: `--conv-channels` (default `16,32,64`; first stage stride
1, every later stage stride 2), `--extra-conv-channels` (default `32`;
same-resolution stages appended, empty string for none), `--dropout`,
`--epochs` (60), `--qat-epochs` (20), `--use-augmentation`, `--seed` (1234).

Writes to `output/<artifacts-name>/`: `float_model.pt`, `qat_model.pt`,
`quantized_model.npz`, `model_manifest.json`, `accuracy_table.json`,
`confusion_matrix.json`.

**On reproducibility.** `--seed` seeds Python, NumPy and Torch, but
`torch.use_deterministic_algorithms` is *not* set, so cuDNN's nondeterministic
backward kernels give roughly 0.5–1.4 points of run-to-run variation on
identical settings. Treat differences smaller than that between configurations
as noise. This was confirmed by retraining a published configuration and
getting 81.9% where 81.1% was recorded.

---

## 3. Export and verify

```bash
python export_rgb_cnn_c_weights.py rgb_cnn_5x5
python verify_rgb_cnn_c_export.py  rgb_cnn_5x5
```

Both take the `output/` subdirectory name as their single argument and default
to `rgb_cnn` if omitted.

The exporter writes `output/rgb_cnn_5x5/esp32/`:

- `model_weights.h` — int8 weights, quantized multipliers, and a complete
  `model_forward()`. Carries **both** conv paths behind
  `#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)`:
  ESP-NN with OHWI weights, or portable C with OIHW. Same weights, same
  surrounding code — only the kernel and layout differ.
- `network_info.json` — the manifest the firmware and this README quote from.

The verifier writes the self-test vector headers the firmware compiles in —
`rgb_synth_vectors.h` (LCG + structured patterns) and `rgb_real_images.h` (two
real test-split images) — plus a standalone `verify_rgb_export.c` harness and
its compiled binary, into `output/rgb_cnn_5x5/`.

It re-extracts features from the real test JPEGs — using
`extract_rgb_blocks` when the manifest says `reduction: "block_mean"` — runs
them through both the Python int8 reference and the compiled C, and diffs.
This is the check that catches layout mistakes; the QAT-vs-int8 agreement
number from training does not, because both sides of that comparison are
Python.

To deploy, copy `output/rgb_cnn_5x5/esp32/model_weights.h` plus
`output/rgb_cnn_5x5/rgb_synth_vectors.h` and `rgb_real_images.h` into
`esp32_cam/esp32_rgb_cnn/include/` and rebuild.

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
