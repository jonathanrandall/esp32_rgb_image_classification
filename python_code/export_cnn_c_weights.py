#!/usr/bin/env python3
"""Export a trained CNN's (plain or AC/chroma-projected) int8 weights to a
self-contained C header for ESP32-S3 firmware.

Handles both CNN variants trained by this package:
- `train_cnn.py` / `DctCnnClassifier` (output/cnn): AC-luma and chroma
  planes are avg-pooled and concatenated *raw* (no learned transform)
  before post_concat_conv.
- `train_ac_cnn.py` / `DctCnnAcClassifier` (output/cnn_ac): AC-luma and
  chroma planes each go through a learned 1x1 conv projection
  (ac_proj/chroma_proj) before concatenation. Detected automatically by
  the presence of `ac_proj_weight`/`chroma_proj_weight` in the model's
  quantized_model.npz -- no separate flag needed, the two variants are
  structurally distinguishable from their saved artifacts alone.

Reads output/<cnn|cnn_ac>/quantized_model.npz + model_manifest.json
(produced by train_cnn.py / train_ac_cnn.py) and emits
output/<cnn|cnn_ac>/model_weights.h: int8 weight / int32 bias /
(multiplier, shift) tables per conv/dense layer, using the exact same
Q31-mantissa fixed-point convention as export_c_weights.py (the MLP
exporter) -- gemmlowp/TFLite/ESP-NN style: int32 accumulate, then
`(acc * mult + rounding) >> (31 - shift)`, never floating-point division
at inference time -- plus three fixed-point input-quantization tables
(DC/AC/chroma, since each raw DCT-coefficient plane has its own
calibrated scale) and `model_forward()` in plain C. Its conv2d_int8()
compiles two ways from the same source: a portable reference (no
ESP-IDF/ESP-NN dependency, what verify_cnn_c_export.py checks), or --
when the firmware build defines CONFIG_NN_OPTIMIZED and
CONFIG_IDF_TARGET_ESP32S3 -- an ESP-NN-accelerated implementation
(measured 30-38x faster per conv layer on real hardware, not bit-exact
against the reference; see full_esp32_integration.md for the full design
trail and esp32_cam/esp32_classifier/esp32_nn_instructions.md for how
this was prototyped/validated before being wired in here).

Usage:
    python export_cnn_c_weights.py            # both output/cnn and output/cnn_ac, whichever exist
    python export_cnn_c_weights.py cnn         # just one
    python export_cnn_c_weights.py cnn_ac
"""

import json
import sys
from pathlib import Path

import numpy as np

from dct_common.quantization import quantize_multipliers

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def format_1d(name: str, ctype: str, values, per_line: int = 12) -> str:
    values = [str(int(v)) for v in values]
    lines = [", ".join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
    body = ",\n    ".join(lines)
    return f"static const {ctype} {name}[{len(values)}] = {{\n    {body}\n}};"


def format_2d(name: str, ctype: str, rows, per_line: int = 20) -> str:
    n_rows, n_cols = rows.shape
    row_blocks = []
    for r in range(n_rows):
        values = [str(int(v)) for v in rows[r]]
        lines = [", ".join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
        row_blocks.append("{\n        " + ",\n        ".join(lines) + "\n    }")
    body = ",\n    ".join(row_blocks)
    return f"static const {ctype} {name}[{n_rows}][{n_cols}] = {{\n    {body}\n}};"


def format_flat_conv_weight(name: str, weights4d: np.ndarray) -> str:
    """Conv weights [out_ch, in_ch, kh, kw] -> a flat int8 C array, in
    whichever layout the active conv2d_int8() implementation needs. Same
    array name either way, so model_forward()'s call sites never change --
    only one of the two orderings is ever compiled into a given build:

    - portable reference (default): row-major (out_ch, in_ch, kh, kw) --
      OIHW -- ((oc*in_ch + ic)*kh + ky)*kw + kx, matching the naive
      conv2d_int8()'s hand-written indexing.
    - ESP-NN path (CONFIG_NN_OPTIMIZED + CONFIG_IDF_TARGET_ESP32S3): OHWI
      (out_ch, kh, kw, in_ch), matching esp_nn_conv_s8_esp32s3's required
      filter layout -- confirmed by reading esp_nn_conv_ansi.c's actual
      indexing (not assumed), see full_esp32_integration.md.

    No padding: an earlier version of this function zero-padded the
    input-channel axis to satisfy what looked like a kh*in_ch-multiple-
    of-16 alignment requirement. That diagnosis was wrong -- the real
    constraint was the scratch buffer's pointer alignment (fixed once, in
    model_esp_nn_init()), not per-layer channel counts -- see
    full_esp32_integration.md's "Update" history. Removed once the real
    cause was confirmed on hardware and every layer, padded or not,
    worked correctly through esp_nn with a properly aligned buffer.
    """
    oihw = format_1d(name, "int8_t", weights4d.reshape(-1))
    ohwi = format_1d(name, "int8_t", weights4d.transpose(0, 2, 3, 1).reshape(-1))
    return (
        "#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)\n"
        f"{ohwi}\n"
        "#else\n"
        f"{oihw}\n"
        "#endif"
    )


def render_scratch_query_block(shape: dict) -> str:
    """One `{ ... }`-scoped block computing esp_nn_get_conv_scratch_size()
    for a single conv layer shape and folding it into `max_scratch`. Plain
    string concatenation (not an f-string) throughout -- this is C code
    dense with literal braces, not worth fighting Python's f-string
    brace-escaping over."""
    w, h = shape["W"], shape["H"]
    kw, kh = shape["kw"], shape["kh"]
    stride, padding = shape["stride"], shape["padding"]
    in_ch, out_ch = shape["in_ch"], shape["out_ch"]
    w_out = (w + 2 * padding - kw) // stride + 1
    h_out = (h + 2 * padding - kh) // stride + 1
    lines = [
        "    {",
        "        data_dims_t in_d = { " + str(w) + ", " + str(h) + ", " + str(in_ch) + ", 0 };",
        "        data_dims_t filt_d = { " + str(kw) + ", " + str(kh) + ", " + str(in_ch) + ", 0 };",
        "        data_dims_t out_d = { " + str(w_out) + ", " + str(h_out) + ", " + str(out_ch) + ", 0 };",
        "        conv_params_t cp = { 0, 0, {" + str(stride) + ", " + str(stride) + "}, {"
        + str(padding) + ", " + str(padding) + "}, {1, 1}, {0, 127} };",
        "        int sz = esp_nn_get_conv_scratch_size(&in_d, &filt_d, &out_d, &cp);",
        "        if (sz > max_scratch) max_scratch = sz;",
        "    }",
    ]
    return "\n".join(lines)


ESP_NN_CONV2D_TEMPLATE = """\
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)

#include "esp_nn.h"
#include "esp_heap_caps.h"

/* CHW<->NHWC repack scratch around each esp_nn_conv_s8 call -- sized to the
 * largest single layer's input or output in this model (__REPACK_MAX__
 * bytes), static (not stack -- this project has a documented stack-
 * overflow history) and not malloc'd per-frame. Not reentrant: model_forward()
 * must not be called concurrently from more than one context. */
#define MODEL_CONV_REPACK_MAX __REPACK_MAX__
static int8_t model_repack_in[MODEL_CONV_REPACK_MAX];
static int8_t model_repack_out[MODEL_CONV_REPACK_MAX];

static int model_esp_nn_scratch_ready = 0;

/* One-time (lazy, first call) setup: query esp_nn_conv_s8_esp32s3's own
 * internal scratch requirement for every conv layer shape in this model
 * (varies by shape, not a fixed size -- do not hardcode this), take the
 * max, and hand it a single PSRAM-allocated buffer reused across every
 * call. This is separate from the small CHW/NHWC repack buffers above --
 * ESP-NN's internal scratch measured 40-80KB in this project's prototype,
 * PSRAM is required, not optional.
 *
 * The buffer must be 16-BYTE ALIGNED, not just PSRAM -- confirmed on real
 * hardware: the esp32s3 assembly kernel silently corrupts output when the
 * scratch pointer isn't (mod16==0: bit-exact; mod16==8: most elements
 * wrong). heap_caps_malloc only guarantees 8-byte alignment, not enough --
 * heap_caps_aligned_alloc is required. (An earlier diagnosis blamed this on
 * kh*in_ch not being a multiple of 16 instead, and this file used to carry
 * a per-layer runtime fallback plus input-channel zero-padding to work
 * around it -- both removed once the real cause (this pointer's alignment)
 * was confirmed on hardware and every layer, including the ones the old
 * theory said were unsafe, worked correctly through esp_nn once this
 * buffer was properly aligned. See full_esp32_integration.md's "Update"
 * history for the full story if you're wondering where that code went.) */
static void model_esp_nn_init(void) {
    if (model_esp_nn_scratch_ready) return;
    int max_scratch = 0;
__SCRATCH_INIT__
    void *scratch = heap_caps_aligned_alloc(16, max_scratch, MALLOC_CAP_SPIRAM);
    esp_nn_set_conv_scratch_buf(scratch);
    model_esp_nn_scratch_ready = 1;
}

static void conv2d_int8(
    const int8_t *input, int in_ch, int H, int W,
    const int8_t *weight, const int32_t *bias, const int32_t *mult, const int32_t *shift,
    int out_ch, int kh, int kw, int stride, int padding, int out_min, int out_max,
    int8_t *output)
{
    model_esp_nn_init();
    int h_out = (H + 2 * padding - kh) / stride + 1;
    int w_out = (W + 2 * padding - kw) / stride + 1;

    /* input is CHW (matches this file's other generated code); esp_nn wants
     * NHWC -- repack once per call into scratch. Measured ~5.5% overhead
     * total (both directions) vs. the accelerated conv itself -- see
     * full_esp32_integration.md. */
    for (int c = 0; c < in_ch; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                model_repack_in[(y * W + x) * in_ch + c] = input[(c * H + y) * W + x];

    data_dims_t input_dims = { W, H, in_ch, 0 };
    data_dims_t filter_dims = { kw, kh, in_ch, 0 };
    data_dims_t output_dims = { w_out, h_out, out_ch, 0 };
    conv_params_t conv_params = {
        .in_offset = 0, .out_offset = 0,
        .stride = { stride, stride }, .padding = { padding, padding }, .dilation = { 1, 1 },
        .activation = { out_min, out_max },
    };
    /* weight is already OHWI (out_ch, kh, kw, in_ch) -- see
     * format_flat_conv_weight() in export_cnn_c_weights.py. shift/mult cast
     * away const: esp_nn_conv_s8's ansi and esp32s3 implementations only
     * ever read them (confirmed directly in esp_nn_conv_ansi.c) --
     * quant_data_t's fields just aren't declared const upstream. Not a data
     * copy, no correctness risk. */
    quant_data_t quant_data = { .shift = (int32_t *)shift, .mult = (int32_t *)mult };

    esp_nn_conv_s8(&input_dims, model_repack_in, &filter_dims, weight, bias,
                   &output_dims, model_repack_out, &conv_params, &quant_data);

    /* NHWC scratch output -> CHW, back to this file's normal layout. */
    for (int c = 0; c < out_ch; c++)
        for (int y = 0; y < h_out; y++)
            for (int x = 0; x < w_out; x++)
                output[(c * h_out + y) * w_out + x] = model_repack_out[(y * w_out + x) * out_ch + c];
}

#else

/* Portable bit-exact reference (see verify_cnn_c_export.py) -- no ESP-IDF/
 * ESP-NN dependency, compiles and runs anywhere, for correctness testing.
 * Generic int8 conv2d: int32 accumulation (zero-padded), then per-output-
 * channel Q31-mantissa requantize, then clip. input/weight/output are flat
 * arrays: input [in_ch][H][W], weight [out_ch][in_ch][kh][kw] (both
 * row-major), output [out_ch][h_out][w_out] where
 * h_out=(H+2*padding-kh)/stride+1 (same for w_out). */
static void conv2d_int8(
    const int8_t *input, int in_ch, int H, int W,
    const int8_t *weight, const int32_t *bias, const int32_t *mult, const int32_t *shift,
    int out_ch, int kh, int kw, int stride, int padding, int out_min, int out_max,
    int8_t *output)
{
    int h_out = (H + 2 * padding - kh) / stride + 1;
    int w_out = (W + 2 * padding - kw) / stride + 1;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oy = 0; oy < h_out; oy++) {
            for (int ox = 0; ox < w_out; ox++) {
                int32_t acc = bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = oy * stride + ky - padding;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ox * stride + kx - padding;
                            if (ix < 0 || ix >= W) continue;
                            int8_t in_val = input[(ic * H + iy) * W + ix];
                            int8_t w_val = weight[((oc * in_ch + ic) * kh + ky) * kw + kx];
                            acc += (int32_t)in_val * (int32_t)w_val;
                        }
                    }
                }
                int64_t product = (int64_t)acc * (int64_t)mult[oc];
                int32_t q = model_round_shift(product, 31 - shift[oc]);
                output[(oc * h_out + oy) * w_out + ox] = (int8_t)model_clip(q, out_min, out_max);
            }
        }
    }
}

#endif /* CONFIG_NN_OPTIMIZED && CONFIG_IDF_TARGET_ESP32S3 */"""


def render_esp_nn_conv2d(conv_layer_shapes: list, repack_max: int) -> str:
    """Guarded conv2d_int8(): ESP-NN-accelerated (esp_nn_conv_s8_esp32s3)
    when CONFIG_NN_OPTIMIZED + CONFIG_IDF_TARGET_ESP32S3 are both defined
    (set via platformio.ini build_flags in the firmware project), else the
    portable reference. Same name/signature either way, so every call site
    in model_forward() is unchanged regardless of which is active.

    The ESP-NN side of this (data layout, struct fields, scratch-buffer
    handling, the shift/mult const cast) was determined from a working,
    on-hardware-measured prototype, not guessed -- see
    full_esp32_integration.md for the full trail. This function cannot be
    verified by verify_cnn_c_export.py (no Xtensa toolchain/hardware on
    this machine) -- only the portable #else branch is PC-testable; the
    #if branch needs on-device re-verification against test_vectors.h
    after every export, same as the original prototype.
    """
    scratch_init = "\n".join(render_scratch_query_block(s) for s in conv_layer_shapes)
    return (
        ESP_NN_CONV2D_TEMPLATE
        .replace("__REPACK_MAX__", str(repack_max))
        .replace("__SCRATCH_INIT__", scratch_init)
    )


def emit_conv_layer(lines: list, prefix: str, weight4d: np.ndarray, bias, mult, shift) -> None:
    lines.append(format_flat_conv_weight(f"{prefix}_WEIGHT", weight4d))
    lines.append(format_1d(f"{prefix}_BIAS", "int32_t", bias))
    lines.append(format_1d(f"{prefix}_MULT", "int32_t", mult))
    lines.append(format_1d(f"{prefix}_SHIFT", "int32_t", shift))
    lines.append("")


def main() -> None:
    requested = sys.argv[1:] or ["cnn", "cnn_ac"]
    for model_dir_name in requested:
        out_dir = OUTPUT_ROOT / model_dir_name
        manifest_path = out_dir / "model_manifest.json"
        if not manifest_path.exists():
            print(f"Skipping {model_dir_name}: no {manifest_path} (train it first)")
            continue
        export_one(model_dir_name, out_dir)


def export_one(model_dir_name: str, out_dir: Path) -> None:
    manifest = json.load(open(out_dir / "model_manifest.json"))
    npz = np.load(out_dir / "quantized_model.npz", allow_pickle=True)

    is_ac_variant = "ac_proj_weight" in npz.files
    num_ac_coeffs = manifest["num_ac_coeffs"]
    use_chroma = manifest["use_chroma"]
    num_coeffs = manifest["num_coeffs"]
    # Chroma AC count is decoupled from luma's (Config.num_chroma_ac_coeffs,
    # see dct_common/config.py) -- falls back to num_coeffs/num_ac_coeffs for
    # manifests written before that field existed, matching Config's own
    # __post_init__ default (None -> match num_ac_coeffs).
    num_chroma_ac_coeffs = manifest.get("num_chroma_ac_coeffs", num_ac_coeffs)
    num_chroma_coeffs = manifest.get("num_chroma_coeffs", num_coeffs)
    y_rows, y_cols = manifest["y_rows"], manifest["y_cols"]
    c_rows, c_cols = manifest["c_rows"], manifest["c_cols"]
    # stride2_conv's actual output spatial size (kernel=3, stride=2, padding=1,
    # matching Int8CnnReference/Int8CnnAcReference's hardcoded stride2_conv
    # call) -- NOT necessarily equal to c_rows/c_cols. That coincidence only
    # held for the original 4:2:0/even-y_rows configs; under 4:2:2 (c_rows can
    # equal y_rows, no vertical chroma downsampling) or an odd y_rows/y_cols
    # (e.g. 160x120's y_rows=15), they diverge. This is the real "working
    # resolution" from stride2_conv onward through post_concat_conv/extra_convs
    # (which all preserve H/W) -- the Python/NumPy reference gets this for
    # free via h.shape[-2:] on the actual tensor; the C exporter has to
    # compute it explicitly since it generates static array sizes.
    stride2_rows = (y_rows + 2 * 1 - 3) // 2 + 1
    stride2_cols = (y_cols + 2 * 1 - 3) // 2 + 1
    lum_channels = manifest["lum_channels"]
    stride2_channels = manifest["stride2_channels"]
    post_concat_channels = manifest["post_concat_channels"]
    extra_conv_channels = manifest["extra_conv_channels"]
    num_classes = manifest["num_classes"]

    if is_ac_variant:
        ac_branch_channels = manifest["ac_proj_channels"] if num_ac_coeffs > 0 else 0
        chroma_branch_channels = manifest["chroma_proj_channels"] if use_chroma else 0
    else:
        ac_branch_channels = num_ac_coeffs
        chroma_branch_channels = (2 * num_chroma_coeffs) if use_chroma else 0
    post_concat_in = stride2_channels + ac_branch_channels + chroma_branch_channels

    final_channels = extra_conv_channels[-1] if extra_conv_channels else post_concat_channels

    # Per-conv-layer shapes, used only to size/init the ESP-NN scratch and
    # CHW<->NHWC repack buffers (guarded -- the portable path doesn't need
    # this at all). Mirrors model_forward()'s own conv2d_int8() call
    # sequence exactly, built from the same manifest-derived values used
    # to generate that body below, so it can't drift out of sync with it.
    #
    # Every layer goes through esp_nn unconditionally now -- an earlier
    # version of this list also tracked pad_to_16/weight_in_ch per layer
    # to work around what looked like a per-layer channel-alignment
    # requirement. That turned out to be a wrong diagnosis (the real
    # constraint was model_esp_nn_init()'s scratch buffer pointer
    # alignment, fixed once, see its comment above); removed once
    # confirmed on hardware that every layer works correctly through
    # esp_nn with no per-layer special-casing at all.
    conv_layer_shapes = [
        dict(in_ch=1, H=y_rows, W=y_cols, out_ch=lum_channels, kh=3, kw=3, stride=1, padding=1),
        dict(in_ch=lum_channels, H=y_rows, W=y_cols, out_ch=stride2_channels, kh=3, kw=3, stride=2, padding=1),
    ]
    if is_ac_variant and num_ac_coeffs > 0:
        conv_layer_shapes.append(dict(in_ch=num_ac_coeffs, H=stride2_rows, W=stride2_cols,
                                       out_ch=manifest["ac_proj_channels"], kh=1, kw=1, stride=1, padding=0))
    if is_ac_variant and use_chroma:
        conv_layer_shapes.append(dict(in_ch=2 * num_chroma_coeffs, H=stride2_rows, W=stride2_cols,
                                       out_ch=manifest["chroma_proj_channels"], kh=1, kw=1, stride=1, padding=0))
    conv_layer_shapes.append(dict(in_ch=post_concat_in, H=stride2_rows, W=stride2_cols,
                                   out_ch=post_concat_channels, kh=3, kw=3, stride=1, padding=1))
    prev_ch = post_concat_channels
    for ch in extra_conv_channels:
        conv_layer_shapes.append(dict(in_ch=prev_ch, H=stride2_rows, W=stride2_cols,
                                       out_ch=ch, kh=3, kw=3, stride=1, padding=1))
        prev_ch = ch

    repack_max = 0
    for s in conv_layer_shapes:
        h_out = (s["H"] + 2 * s["padding"] - s["kh"]) // s["stride"] + 1
        w_out = (s["W"] + 2 * s["padding"] - s["kw"]) // s["stride"] + 1
        repack_max = max(repack_max, s["in_ch"] * s["H"] * s["W"], s["out_ch"] * h_out * w_out)

    dc_scale = npz["dc_scale"]  # shape [1]
    dc_mult, dc_shift = quantize_multipliers(1.0 / dc_scale)
    if num_ac_coeffs > 0:
        ac_scales = npz["ac_scales"]
        ac_mult, ac_shift = quantize_multipliers(1.0 / ac_scales)
    if use_chroma:
        chroma_scales = npz["chroma_scales"]
        chroma_mult, chroma_shift = quantize_multipliers(1.0 / chroma_scales)

    guard = f"DCT_CNN_{model_dir_name.upper()}_MODEL_WEIGHTS_H"
    lines = []
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("/*")
    lines.append(f" * Auto-generated by export_cnn_c_weights.py from")
    lines.append(f" * python_code/output/{model_dir_name}/quantized_model.npz + model_manifest.json.")
    lines.append(" * DO NOT EDIT BY HAND -- change the model/training config and")
    lines.append(f" * re-run {'train_ac_cnn.py' if is_ac_variant else 'train_cnn.py'} && "
                  f"python export_cnn_c_weights.py {model_dir_name} instead.")
    lines.append(" *")
    lines.append(f" * Architecture: {'AC/chroma 1x1-projected CNN (DctCnnAcClassifier)' if is_ac_variant else 'plain CNN (DctCnnClassifier)'}")
    lines.append(f" * Source run accuracy (test split): float={manifest['float_test_accuracy']:.4f}"
                  f"  qat={manifest['qat_test_accuracy']:.4f}"
                  f"  int8={manifest['int8_reference_test_accuracy']:.4f}"
                  f"  agreement={manifest['int8_vs_qat_agreement']:.4f}")
    lines.append(f" * Capture: {manifest['capture_width']}x{manifest['capture_height']}"
                  f"  Y grid={y_rows}x{y_cols}  Cb=Cr grid={c_rows}x{c_cols}"
                  f"  num_coeffs={num_coeffs} (DC + {num_ac_coeffs} AC)  use_chroma={use_chroma}"
                  + (f"  chroma_num_coeffs={num_chroma_coeffs} (DC + {num_chroma_ac_coeffs} AC)" if use_chroma else ""))
    lines.append(" *")
    lines.append(" * Input planes (must match dct_common/features.py's extract_y_dct_planes /")
    lines.append(" * extract_cbcr_dct_planes layout -- DEQUANTIZED by each image's own JPEG")
    lines.append(" * quant table, [coeff][block_row][block_col], never flattened): a DC plane")
    lines.append(" * (Y zig-zag position 0), num_ac_coeffs Y AC planes (zig-zag positions")
    lines.append(" * 1..num_ac_coeffs), and if use_chroma, 2*chroma_num_coeffs chroma planes")
    lines.append(" * (Cb's DC+AC positions, then Cr's -- chroma_num_coeffs is independent of")
    lines.append(" * luma's num_coeffs, see Config.num_chroma_ac_coeffs).")
    lines.append(" *")
    lines.append(" * Inference recipe (see model_forward() below for the portable C")
    lines.append(" * implementation): quantize each raw plane to int8 with its own")
    lines.append(" * per-(DC|AC-position|chroma-position) scale (model_quantize_*, same")
    lines.append(" * fixed-point convention as export_c_weights.py's INPUT_QUANT_MULT/SHIFT)")
    lines.append(" * -> lum_conv (3x3) -> stride2_conv (3x3, stride 2) -> "
                  + ("AC/chroma 1x1 projections, then " if is_ac_variant else "")
                  + "concat with stride2_conv's output -> post_concat_conv (3x3) -> "
                    "extra conv stage(s) -> global average pool -> dense output layer -> logits.")
    lines.append(" * Every conv layer: int32 accumulate (bias + sum input*weight, zero-padded),")
    lines.append(" * then per-output-channel Q31-mantissa requantize, then clip to [0,127]")
    lines.append(" * (ReLU'd) except the final dense layer, which clips to [-128,127] (raw logits).")
    lines.append(" */")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append(f"#define MODEL_CAPTURE_WIDTH {manifest['capture_width']}")
    lines.append(f"#define MODEL_CAPTURE_HEIGHT {manifest['capture_height']}")
    lines.append(f"#define MODEL_Y_ROWS {y_rows}")
    lines.append(f"#define MODEL_Y_COLS {y_cols}")
    lines.append(f"#define MODEL_C_ROWS {c_rows}")
    lines.append(f"#define MODEL_C_COLS {c_cols}")
    lines.append(f"#define MODEL_STRIDE2_ROWS {stride2_rows}")
    lines.append(f"#define MODEL_STRIDE2_COLS {stride2_cols}")
    lines.append(f"#define MODEL_NUM_COEFFS {num_coeffs}")
    lines.append(f"#define MODEL_NUM_AC_COEFFS {num_ac_coeffs}")
    lines.append(f"#define MODEL_USE_CHROMA {1 if use_chroma else 0}")
    # Always defined (not gated on use_chroma) so model_forward()'s signature
    # -- see render_c_reference -- never changes shape across a --no-chroma
    # toggle: callers can always pass a chroma_planes argument, it's just
    # unused internally when MODEL_USE_CHROMA is 0.
    lines.append(f"#define MODEL_NUM_CHROMA_COEFFS {num_chroma_coeffs}")
    lines.append(f"#define MODEL_NUM_CHROMA_AC_COEFFS {num_chroma_ac_coeffs}")
    lines.append(f"#define MODEL_NUM_CLASSES {num_classes}")
    lines.append(f"#define MODEL_LUM_CHANNELS {lum_channels}")
    lines.append(f"#define MODEL_STRIDE2_CHANNELS {stride2_channels}")
    if is_ac_variant and num_ac_coeffs > 0:
        lines.append(f"#define MODEL_AC_PROJ_CHANNELS {manifest['ac_proj_channels']}")
    if is_ac_variant and use_chroma:
        lines.append(f"#define MODEL_CHROMA_PROJ_CHANNELS {manifest['chroma_proj_channels']}")
    lines.append(f"#define MODEL_POST_CONCAT_IN {post_concat_in}")
    lines.append(f"#define MODEL_POST_CONCAT_CHANNELS {post_concat_channels}")
    for i, ch in enumerate(extra_conv_channels[:-1]):
        lines.append(f"#define MODEL_EXTRA_CONV_{i}_CHANNELS {ch}")
    lines.append(f"#define MODEL_FINAL_CHANNELS {final_channels}")
    lines.append("")
    class_name_literals = ", ".join(f'"{c}"' for c in manifest["class_names"])
    lines.append(f"static const char *const MODEL_CLASS_NAMES[MODEL_NUM_CLASSES] = {{ {class_name_literals} }};")
    lines.append("")

    lines.append("/* ---- Input quantization: raw dequantized DCT coefficient -> int8 ---- */")
    lines.append(format_1d("DC_QUANT_MULT", "int32_t", dc_mult))
    lines.append(format_1d("DC_QUANT_SHIFT", "int32_t", dc_shift))
    if num_ac_coeffs > 0:
        lines.append(format_1d("AC_QUANT_MULT", "int32_t", ac_mult))
        lines.append(format_1d("AC_QUANT_SHIFT", "int32_t", ac_shift))
    if use_chroma:
        lines.append(format_1d("CHROMA_QUANT_MULT", "int32_t", chroma_mult))
        lines.append(format_1d("CHROMA_QUANT_SHIFT", "int32_t", chroma_shift))
    lines.append("")

    lines.append("/* ---- lum_conv (3x3, stride 1) ---- */")
    emit_conv_layer(lines, "LUM", npz["lum_weight"], npz["lum_bias"], npz["lum_mult"], npz["lum_shift"])
    lines.append("/* ---- stride2_conv (3x3, stride 2) ---- */")
    emit_conv_layer(lines, "STRIDE2", npz["stride2_weight"], npz["stride2_bias"], npz["stride2_mult"], npz["stride2_shift"])

    if is_ac_variant and num_ac_coeffs > 0:
        lines.append("/* ---- ac_proj (1x1) ---- */")
        emit_conv_layer(lines, "AC_PROJ", npz["ac_proj_weight"], npz["ac_proj_bias"], npz["ac_proj_mult"], npz["ac_proj_shift"])
    if is_ac_variant and use_chroma:
        lines.append("/* ---- chroma_proj (1x1) ---- */")
        emit_conv_layer(lines, "CHROMA_PROJ", npz["chroma_proj_weight"], npz["chroma_proj_bias"], npz["chroma_proj_mult"], npz["chroma_proj_shift"])

    lines.append("/* ---- post_concat_conv (3x3, stride 1) ---- */")
    emit_conv_layer(lines, "POST_CONCAT", npz["post_concat_weight"], npz["post_concat_bias"], npz["post_concat_mult"], npz["post_concat_shift"])

    extra_weights = list(npz["extra_weights"]) if len(extra_conv_channels) else []
    extra_biases = list(npz["extra_biases"]) if len(extra_conv_channels) else []
    extra_mults = list(npz["extra_mults"]) if len(extra_conv_channels) else []
    extra_shifts = list(npz["extra_shifts"]) if len(extra_conv_channels) else []
    for i in range(len(extra_conv_channels)):
        lines.append(f"/* ---- extra_convs[{i}] (3x3, stride 1) ---- */")
        emit_conv_layer(lines, f"EXTRA_CONV_{i}", extra_weights[i], extra_biases[i], extra_mults[i], extra_shifts[i])

    lines.append("/* ---- output (dense) ---- */")
    lines.append(format_2d("OUTPUT_WEIGHT", "int8_t", npz["output_weight"]))
    lines.append(format_1d("OUTPUT_BIAS", "int32_t", npz["output_bias"]))
    lines.append(format_1d("OUTPUT_MULT", "int32_t", npz["output_mult"]))
    lines.append(format_1d("OUTPUT_SHIFT", "int32_t", npz["output_shift"]))
    lines.append("")

    lines.append(render_c_reference(is_ac_variant, num_ac_coeffs, use_chroma, num_chroma_coeffs,
                                     len(extra_conv_channels), guard, conv_layer_shapes, repack_max))

    out_path = out_dir / "model_weights.h"
    header_text = "\n".join(lines) + "\n"
    out_path.write_text(header_text)
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)  "
          f"variant={'cnn_ac' if is_ac_variant else 'cnn'} num_ac_coeffs={num_ac_coeffs} "
          f"use_chroma={use_chroma} num_classes={num_classes}")

    esp32_dir = out_dir / "esp32"
    esp32_dir.mkdir(exist_ok=True)
    (esp32_dir / "model_weights.h").write_text(header_text)
    network_info = dict(manifest, architecture="cnn_ac" if is_ac_variant else "cnn", weights_header="model_weights.h")
    (esp32_dir / "network_info.json").write_text(json.dumps(network_info, indent=2) + "\n")
    print(f"Wrote {esp32_dir}/ (model_weights.h + network_info.json -- self-contained deployment bundle)")


def render_c_reference(is_ac_variant: bool, num_ac_coeffs: int, use_chroma: bool, num_chroma_coeffs: int,
                        num_extra: int, guard: str, conv_layer_shapes: list, repack_max: int) -> str:
    parts = []
    parts.append("""\
/* ---------------------------------------------------------------------
 * model_forward() and its helpers below are the same either way this
 * header is built -- what changes is conv2d_int8() itself:
 *
 *   - default: a portable, no-ESP-IDF/ESP-NN-dependency reference
 *     implementation. Compiles and runs anywhere, for correctness testing
 *     (see verify_cnn_c_export.py) -- this is the ONLY path that tool can
 *     check, no Xtensa toolchain/hardware on the machine that runs it.
 *   - CONFIG_NN_OPTIMIZED + CONFIG_IDF_TARGET_ESP32S3 both defined (set
 *     via platformio.ini build_flags in the firmware project): an
 *     ESP-NN-accelerated implementation (esp_nn_conv_s8_esp32s3, SIMD
 *     assembly) -- measured 30-38x faster per layer on real hardware, see
 *     full_esp32_integration.md for the full trail (data layout, struct
 *     fields, why it isn't bit-exact, how much it actually costs). This
 *     path needs on-device re-verification against test_vectors.h after
 *     every export -- it cannot be checked from here.
 * ------------------------------------------------------------------- */

static inline int32_t model_round_shift(int64_t product, int32_t total_shift) {
    if (total_shift > 0) {
        int64_t rounding = (int64_t)1 << (total_shift - 1);
        return (int32_t)((product + rounding) >> total_shift);
    }
    return (int32_t)(product << (-total_shift));
}

static inline int32_t model_clip(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int8_t model_quantize_dc(int32_t coefficient) {
    int64_t product = (int64_t)coefficient * (int64_t)DC_QUANT_MULT[0];
    int32_t q = model_round_shift(product, 31 - DC_QUANT_SHIFT[0]);
    return (int8_t)model_clip(q, -127, 127);
}""")
    if num_ac_coeffs > 0:
        parts.append("""
static inline int8_t model_quantize_ac(int32_t coefficient, int position) {
    int64_t product = (int64_t)coefficient * (int64_t)AC_QUANT_MULT[position];
    int32_t q = model_round_shift(product, 31 - AC_QUANT_SHIFT[position]);
    return (int8_t)model_clip(q, -127, 127);
}""")
    if use_chroma:
        parts.append("""
static inline int8_t model_quantize_chroma(int32_t coefficient, int position) {
    int64_t product = (int64_t)coefficient * (int64_t)CHROMA_QUANT_MULT[position];
    int32_t q = model_round_shift(product, 31 - CHROMA_QUANT_SHIFT[position]);
    return (int8_t)model_clip(q, -127, 127);
}""")

    parts.append("\n" + render_esp_nn_conv2d(conv_layer_shapes, repack_max))

    parts.append("""
/* Integer average pooling from [C][H][W] to [C][oh][ow] using the same
 * per-output-index window rule as torch's adaptive_avg_pool2d: output
 * index i pools input range [floor(i*H/oh), ceil((i+1)*H/oh)). Degenerates
 * to non-overlapping equal-size bins when H is a multiple of oh; at odd
 * grid sizes (e.g. 160x120 under 4:2:2, y_rows=15) adjacent windows
 * genuinely overlap by a row/column, matching what the float/QAT model's
 * adaptive_avg_pool2d actually computed -- round-to-nearest, half away
 * from zero (matches dct_common/quantization.py's avg_pool_int_exact,
 * safe for the negative AC/chroma coefficient values this pools). */
static void avg_pool_int8(const int8_t *input, int C, int H, int W, int oh, int ow, int8_t *output) {
    for (int c = 0; c < C; c++) {
        for (int oy = 0; oy < oh; oy++) {
            int y0 = (oy * H) / oh;
            int y1 = ((oy + 1) * H + oh - 1) / oh;
            for (int ox = 0; ox < ow; ox++) {
                int x0 = (ox * W) / ow;
                int x1 = ((ox + 1) * W + ow - 1) / ow;
                int32_t sum = 0;
                for (int ky = y0; ky < y1; ky++) {
                    for (int kx = x0; kx < x1; kx++) {
                        sum += input[(c * H + ky) * W + kx];
                    }
                }
                int32_t denom = (y1 - y0) * (x1 - x0);
                int32_t abs_sum = sum < 0 ? -sum : sum;
                int32_t rounded = (abs_sum + denom / 2) / denom;
                output[(c * oh + oy) * ow + ox] = (int8_t)(sum < 0 ? -rounded : rounded);
            }
        }
    }
}

static int model_argmax(const int8_t *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}""")

    # ---- model_forward ----
    # ac_planes/chroma_planes are ALWAYS part of the signature, even when
    # num_ac_coeffs==0 / use_chroma==False (in which case they're accepted
    # but unused -- see the (void) casts below). This keeps model_forward()'s
    # signature stable across a --num-ac-coeffs 0 or --no-chroma retrain, so
    # firmware call sites don't need editing just because a config toggled;
    # only a real architecture change (e.g. cnn vs cnn_ac) still does.
    sig_lines = ["/* Full forward pass: raw dequantized DCT coefficient planes -> int8 logits.",
                 " * ac_planes/chroma_planes are always present in this signature -- see",
                 " * MODEL_NUM_AC_COEFFS/MODEL_NUM_CHROMA_COEFFS (both may be 0, meaning the",
                 " * corresponding argument is accepted but ignored). */",
                 "static void model_forward("]
    sig_lines.append("    const int32_t dc_plane[MODEL_Y_ROWS][MODEL_Y_COLS],")
    sig_lines.append("    const int32_t ac_planes[MODEL_NUM_AC_COEFFS][MODEL_Y_ROWS][MODEL_Y_COLS],")
    sig_lines.append("    const int32_t chroma_planes[2 * MODEL_NUM_CHROMA_COEFFS][MODEL_C_ROWS][MODEL_C_COLS],")
    sig_lines.append("    int8_t logits[MODEL_NUM_CLASSES])")
    sig_lines.append("{")
    body = list(sig_lines)

    if num_ac_coeffs == 0:
        body.append("    (void)ac_planes;  /* MODEL_NUM_AC_COEFFS == 0 for this model -- unused */")
    if not use_chroma:
        body.append("    (void)chroma_planes;  /* MODEL_USE_CHROMA == 0 for this model -- unused */")

    body.append("    int8_t q_dc[MODEL_Y_ROWS * MODEL_Y_COLS];")
    body.append("    for (int y = 0; y < MODEL_Y_ROWS; y++)")
    body.append("        for (int x = 0; x < MODEL_Y_COLS; x++)")
    body.append("            q_dc[y * MODEL_Y_COLS + x] = model_quantize_dc(dc_plane[y][x]);")
    body.append("")
    body.append("    int8_t lum_out[MODEL_LUM_CHANNELS * MODEL_Y_ROWS * MODEL_Y_COLS];")
    body.append("    conv2d_int8(q_dc, 1, MODEL_Y_ROWS, MODEL_Y_COLS,")
    body.append("                LUM_WEIGHT, LUM_BIAS, LUM_MULT, LUM_SHIFT,")
    body.append("                MODEL_LUM_CHANNELS, 3, 3, 1, 1, 0, 127, lum_out);")
    body.append("")
    body.append("    /* stride2_conv's actual output size -- MODEL_STRIDE2_ROWS/COLS, NOT")
    body.append("     * necessarily MODEL_C_ROWS/COLS (those only coincide for the original")
    body.append("     * 4:2:0/even-y_rows configs; see this exporter's stride2_rows/cols comment). */")
    body.append("    int8_t stride2_out[MODEL_STRIDE2_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
    body.append("    conv2d_int8(lum_out, MODEL_LUM_CHANNELS, MODEL_Y_ROWS, MODEL_Y_COLS,")
    body.append("                STRIDE2_WEIGHT, STRIDE2_BIAS, STRIDE2_MULT, STRIDE2_SHIFT,")
    body.append("                MODEL_STRIDE2_CHANNELS, 3, 3, 2, 1, 0, 127, stride2_out);")
    body.append("")
    body.append("    int8_t concat_buf[MODEL_POST_CONCAT_IN * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
    body.append("    memcpy(concat_buf, stride2_out, sizeof(stride2_out));")
    body.append("    int concat_offset = MODEL_STRIDE2_CHANNELS;")

    if num_ac_coeffs > 0:
        body.append("")
        body.append("    int8_t q_ac[MODEL_NUM_AC_COEFFS * MODEL_Y_ROWS * MODEL_Y_COLS];")
        body.append("    for (int k = 0; k < MODEL_NUM_AC_COEFFS; k++)")
        body.append("        for (int y = 0; y < MODEL_Y_ROWS; y++)")
        body.append("            for (int x = 0; x < MODEL_Y_COLS; x++)")
        body.append("                q_ac[(k * MODEL_Y_ROWS + y) * MODEL_Y_COLS + x] = model_quantize_ac(ac_planes[k][y][x], k);")
        body.append("    /* pooled DOWN to stride2_conv's output size, to match for concatenation --")
        body.append("     * not MODEL_C_ROWS/COLS, which describe chroma's own native grid, unrelated. */")
        body.append("    int8_t ac_pooled[MODEL_NUM_AC_COEFFS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
        body.append("    avg_pool_int8(q_ac, MODEL_NUM_AC_COEFFS, MODEL_Y_ROWS, MODEL_Y_COLS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS, ac_pooled);")
        if is_ac_variant:
            body.append("    int8_t ac_proj_out[MODEL_AC_PROJ_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
            body.append("    conv2d_int8(ac_pooled, MODEL_NUM_AC_COEFFS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS,")
            body.append("                AC_PROJ_WEIGHT, AC_PROJ_BIAS, AC_PROJ_MULT, AC_PROJ_SHIFT,")
            body.append("                MODEL_AC_PROJ_CHANNELS, 1, 1, 1, 0, 0, 127, ac_proj_out);")
            body.append("    memcpy(concat_buf + concat_offset * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS, ac_proj_out, sizeof(ac_proj_out));")
            body.append("    concat_offset += MODEL_AC_PROJ_CHANNELS;")
        else:
            body.append("    memcpy(concat_buf + concat_offset * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS, ac_pooled, sizeof(ac_pooled));")
            body.append("    concat_offset += MODEL_NUM_AC_COEFFS;")

    if use_chroma:
        body.append("")
        body.append("    int8_t q_chroma[2 * MODEL_NUM_CHROMA_COEFFS * MODEL_C_ROWS * MODEL_C_COLS];")
        body.append("    for (int k = 0; k < 2 * MODEL_NUM_CHROMA_COEFFS; k++)")
        body.append("        for (int y = 0; y < MODEL_C_ROWS; y++)")
        body.append("            for (int x = 0; x < MODEL_C_COLS; x++)")
        body.append("                q_chroma[(k * MODEL_C_ROWS + y) * MODEL_C_COLS + x] = model_quantize_chroma(chroma_planes[k][y][x], k);")
        body.append("    /* chroma's native C_ROWS x C_COLS grid, pooled DOWN to stride2_conv's")
        body.append("     * output size -- an identity only when C_ROWS/COLS happen to already equal")
        body.append("     * STRIDE2_ROWS/COLS (the original 4:2:0/even-y_rows case); a real resize")
        body.append("     * otherwise (e.g. 4:2:2, where chroma keeps luma's row count). */")
        body.append("    int8_t chroma_pooled[2 * MODEL_NUM_CHROMA_COEFFS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
        body.append("    avg_pool_int8(q_chroma, 2 * MODEL_NUM_CHROMA_COEFFS, MODEL_C_ROWS, MODEL_C_COLS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS, chroma_pooled);")
        if is_ac_variant:
            body.append("    int8_t chroma_proj_out[MODEL_CHROMA_PROJ_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
            body.append("    conv2d_int8(chroma_pooled, 2 * MODEL_NUM_CHROMA_COEFFS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS,")
            body.append("                CHROMA_PROJ_WEIGHT, CHROMA_PROJ_BIAS, CHROMA_PROJ_MULT, CHROMA_PROJ_SHIFT,")
            body.append("                MODEL_CHROMA_PROJ_CHANNELS, 1, 1, 1, 0, 0, 127, chroma_proj_out);")
            body.append("    memcpy(concat_buf + concat_offset * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS, chroma_proj_out, sizeof(chroma_proj_out));")
            body.append("    concat_offset += MODEL_CHROMA_PROJ_CHANNELS;")
        else:
            body.append("    memcpy(concat_buf + concat_offset * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS, chroma_pooled, sizeof(chroma_pooled));")
            body.append("    concat_offset += 2 * MODEL_NUM_CHROMA_COEFFS;")

    body.append("")
    body.append("    int8_t post_concat_out[MODEL_POST_CONCAT_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
    body.append("    conv2d_int8(concat_buf, MODEL_POST_CONCAT_IN, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS,")
    body.append("                POST_CONCAT_WEIGHT, POST_CONCAT_BIAS, POST_CONCAT_MULT, POST_CONCAT_SHIFT,")
    body.append("                MODEL_POST_CONCAT_CHANNELS, 3, 3, 1, 1, 0, 127, post_concat_out);")

    prev_buf = "post_concat_out"
    prev_channels = "MODEL_POST_CONCAT_CHANNELS"
    for i in range(num_extra):
        out_buf = f"extra_conv_{i}_out"
        out_channels_macro = "MODEL_FINAL_CHANNELS" if i == num_extra - 1 else f"MODEL_EXTRA_CONV_{i}_CHANNELS"
        body.append("")
        body.append(f"    int8_t {out_buf}[{out_channels_macro} * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];")
        body.append(f"    conv2d_int8({prev_buf}, {prev_channels}, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS,")
        body.append(f"                EXTRA_CONV_{i}_WEIGHT, EXTRA_CONV_{i}_BIAS, EXTRA_CONV_{i}_MULT, EXTRA_CONV_{i}_SHIFT,")
        body.append(f"                {out_channels_macro}, 3, 3, 1, 1, 0, 127, {out_buf});")
        prev_buf = out_buf
        prev_channels = out_channels_macro

    body.append("")
    body.append("    int8_t pooled[MODEL_FINAL_CHANNELS];")
    body.append(f"    avg_pool_int8({prev_buf}, MODEL_FINAL_CHANNELS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS, 1, 1, pooled);")
    body.append("")
    body.append("    for (int j = 0; j < MODEL_NUM_CLASSES; j++) {")
    body.append("        int32_t acc = OUTPUT_BIAS[j];")
    body.append("        for (int i = 0; i < MODEL_FINAL_CHANNELS; i++) {")
    body.append("            acc += (int32_t)pooled[i] * (int32_t)OUTPUT_WEIGHT[j][i];")
    body.append("        }")
    body.append("        int64_t product = (int64_t)acc * (int64_t)OUTPUT_MULT[j];")
    body.append("        int32_t q = model_round_shift(product, 31 - OUTPUT_SHIFT[j]);")
    body.append("        logits[j] = (int8_t)model_clip(q, -128, 127); /* raw logits, no activation */")
    body.append("    }")
    body.append("}")
    parts.append("\n".join(body))
    parts.append(f"\n#endif /* {guard} */")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
