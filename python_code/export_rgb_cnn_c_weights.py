#!/usr/bin/env python3
"""Export train_rgb_cnn.py's int8 weights to a self-contained C header.

Reads output/rgb_cnn/quantized_model.npz + model_manifest.json and emits
output/rgb_cnn/model_weights.h: int8 weight / int32 bias / (multiplier,
shift) tables per conv/dense layer, same Q31-mantissa fixed-point
convention as export_cnn_c_weights.py (gemmlowp/TFLite/ESP-NN style: int32
accumulate, then `(acc * mult + rounding) >> (31 - shift)`, never
floating-point division), plus a portable `model_forward()` in plain C.

Simpler than export_cnn_c_weights.py in two real ways, not just smaller:
  - Input quantization is a FIXED, exact scale (pixel - 128, see
    dct_common/rgb_features.py) -- no percentile-calibrated multiply-shift
    tables needed for the input, unlike every DCT exporter.
  - The conv trunk is one uniform, variable-length stack (see
    dct_common/models/rgb_cnn.py's module docstring for why: no AC/chroma
    fusion branch to special-case), so model_forward() is a single loop
    over layers instead of the DCT exporter's several distinctly-shaped
    stages.

Emits BOTH a portable reference and an ESP-NN-accelerated path, selected
by the same `#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)`
as export_cnn_c_weights.py. Acceleration is required, not optional: the
whole point of the RGB firmware is a DCT-vs-RGB latency comparison, and
ESP-NN is worth 14-19x on the DCT side (esp32_nn_testing.md: 306 ms
portable -> 21.9 ms accelerated), so an unaccelerated RGB number compared
against an accelerated DCT number would measure the kernels, not the
architectures. Both sides must hold acceleration equal.

No per-layer alignment special-casing, deliberately. The "esp32s3 kernel
corrupts output when kh*in_channels isn't a multiple of 16" rule is a
disproven diagnosis (still quoted in some stale firmware comments): the
real constraint is that ESP-NN's conv scratch POINTER must be 16-byte
aligned, which model_esp_nn_init() handles once via
heap_caps_aligned_alloc(16, ...). Every layer goes through esp_nn
unconditionally -- including conv0 at in_channels=3, which the old theory
called unsafe. The DCT model's lum_conv runs accelerated at
in_channels=1 (kh*in_ch=3), which settles it. See export_cnn_c_weights.py's
three separate notes on the removal of the old fallback/zero-padding.

BUFFERS ARE PSRAM-ALLOCATED, not stack or .bss -- the one real structural
difference from export_cnn_c_weights.py, forced by this model's much
larger activations. The DCT model's conv stack runs on a 15x20 block grid
(a few KB per layer, fine as stack locals); this model's conv0 output
alone is 16*120*160 = 307,200 bytes, and the whole activation set is
~619 KB against the S3's ~512 KB of internal RAM. So model_init()
heap-allocates every activation buffer, preferring internal SRAM and
falling back to PSRAM, hottest buffer first.

ACTIVATIONS ARE NHWC THROUGHOUT, which is the layout esp_nn requires, so
the accelerated path hands it the model's own buffers with no transpose.
An earlier version kept activations in CHW like the DCT exporter and
repacked into scratch on both sides of every conv call; on hardware that
was 611 ms of a 1084 ms inference -- 57% of the total -- spent computing
nothing. The DCT exporter still repacks and is correct to: its activations
are a few KB, where the transpose is negligible. A layout mismatch is
cheap on small activations and dominant on large ones.

NOT reused from export_cnn_c_weights.py: this model has no AC/chroma
planes and no input-quantization tables, so much of that file's structure
doesn't apply. format_1d/format_2d (pure array-formatting, no
C-generation logic) and render_scratch_query_block (pure shape-arithmetic
emitting one esp_nn_get_conv_scratch_size block, model-agnostic) ARE
imported directly rather than re-typed.

Usage:
    python export_rgb_cnn_c_weights.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from dct_common.models.rgb_cnn import _conv_stage_strides
from export_cnn_c_weights import format_1d, format_2d, render_scratch_query_block

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def format_flat_conv_weight(name: str, weights4d: np.ndarray) -> str:
    """Emits BOTH weight layouts under the ESP-NN #if, exactly as
    export_cnn_c_weights.py does -- only one is ever compiled into a build:

    - portable reference (default): OIHW (out_ch, in_ch, kh, kw), matching
      the naive conv2d_int8()'s hand-written indexing.
    - ESP-NN path: OHWI (out_ch, kh, kw, in_ch), the layout
      esp_nn_conv_s8_esp32s3 requires.

    No input-channel zero-padding: see the module docstring on why the
    kh*in_ch alignment rule doesn't exist."""
    oihw = format_1d(name, "int8_t", weights4d.reshape(-1))
    ohwi = format_1d(name, "int8_t", weights4d.transpose(0, 2, 3, 1).reshape(-1))
    return (
        "#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)\n"
        f"{ohwi}\n"
        "#else\n"
        f"{oihw}\n"
        "#endif"
    )


ESP_NN_CONV2D_TEMPLATE = """\
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)

#include "esp_nn.h"
#include "esp_heap_caps.h"

/* NO REPACK BUFFERS. Activations are NHWC end to end -- the layout esp_nn
 * wants -- so the model's own buffers are handed to esp_nn_conv_s8
 * directly. An earlier version of this exporter kept activations in CHW
 * and transposed into scratch on both sides of every conv call; measured
 * on hardware that cost 611 ms of the 1084 ms inference, 57% of the total,
 * to compute nothing. Deleting it took inference to ~466 ms and freed
 * ~600 KB of PSRAM scratch.
 *
 * Note this is specific to models with LARGE activations. The DCT
 * exporter (export_cnn_c_weights.py) still repacks and is right to: its
 * block-grid activations are a few KB, so the transpose is negligible
 * there. A layout mismatch is cheap on small activations and dominant on
 * large ones -- conv0's output here is 307,200 bytes, and the strided
 * writes fall outside cache on every line. */
static int model_esp_nn_scratch_ready = 0;

/* One-time setup: query esp_nn_conv_s8_esp32s3's own internal scratch
 * requirement for every conv layer shape in this model (it varies by shape
 * -- do not hardcode it), take the max, and hand it a single PSRAM buffer
 * reused across every call.
 *
 * The buffer must be 16-BYTE ALIGNED, not merely in PSRAM. This is the real
 * (and only) alignment constraint: with a misaligned scratch pointer the
 * esp32s3 assembly kernel silently corrupts output (mod16==0 bit-exact,
 * mod16==8 most elements wrong). heap_caps_malloc guarantees only 8 bytes,
 * so heap_caps_aligned_alloc is required. An older diagnosis blamed this on
 * kh*in_channels not being a multiple of 16 and added per-layer fallbacks
 * and channel zero-padding; that was wrong and those workarounds were
 * removed after hardware confirmation. Every layer here -- including conv0
 * at in_channels=3 -- goes through esp_nn unconditionally. */
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

    data_dims_t input_dims = { W, H, in_ch, 0 };
    data_dims_t filter_dims = { kw, kh, in_ch, 0 };
    data_dims_t output_dims = { w_out, h_out, out_ch, 0 };
    conv_params_t conv_params = {
        .in_offset = 0, .out_offset = 0,
        .stride = { stride, stride }, .padding = { padding, padding }, .dilation = { 1, 1 },
        .activation = { out_min, out_max },
    };
    /* weight is already OHWI -- see format_flat_conv_weight() in
     * export_rgb_cnn_c_weights.py. shift/mult cast away const: esp_nn_conv_s8
     * only ever reads them, quant_data_t's fields just aren't declared const
     * upstream. Not a copy, no correctness risk. */
    quant_data_t quant_data = { .shift = (int32_t *)shift, .mult = (int32_t *)mult };

    /* Model buffers passed straight through -- both are already NHWC. */
    esp_nn_conv_s8(&input_dims, input, &filter_dims, weight, bias,
                   &output_dims, output, &conv_params, &quant_data);
}

#else
"""


def render_esp_nn_conv2d(layer_shapes: list) -> str:
    scratch_init = "\n".join(render_scratch_query_block(s) for s in layer_shapes)
    return ESP_NN_CONV2D_TEMPLATE.replace("__SCRATCH_INIT__", scratch_init)


def emit_conv_layer(lines: list, prefix: str, weight4d: np.ndarray, bias, mult, shift) -> None:
    lines.append(format_flat_conv_weight(f"{prefix}_WEIGHT", weight4d))
    lines.append(format_1d(f"{prefix}_BIAS", "int32_t", bias))
    lines.append(format_1d(f"{prefix}_MULT", "int32_t", mult))
    lines.append(format_1d(f"{prefix}_SHIFT", "int32_t", shift))
    lines.append("")


def main() -> None:
    # Optional directory name, matching export_cnn_c_weights.py. Without it this
    # only ever exported output/rgb_cnn, so a model trained into a named
    # directory (train_rgb_cnn.py --artifacts-name) could not be exported at all
    # and the argument was silently ignored.
    model_dir_name = sys.argv[1] if len(sys.argv) > 1 else "rgb_cnn"
    out_dir = OUTPUT_ROOT / model_dir_name
    manifest_path = out_dir / "model_manifest.json"
    if not manifest_path.exists():
        print(f"No {manifest_path} -- run train_rgb_cnn.py first.")
        return

    manifest = json.load(open(manifest_path))
    npz = np.load(out_dir / "quantized_model.npz", allow_pickle=True)

    rgb_width, rgb_height = manifest["rgb_width"], manifest["rgb_height"]
    conv_channels = manifest["conv_channels"]
    extra_conv_channels = manifest["extra_conv_channels"]
    num_classes = manifest["num_classes"]
    final_channels = (extra_conv_channels[-1] if extra_conv_channels else conv_channels[-1])

    # Per-layer shapes: main stack (first stage stride 1, rest stride 2,
    # see _conv_stage_strides -- reused directly, not re-derived) then
    # extra stages (always stride 1). Spatial size shrinks only on the
    # main stack's stride-2 layers; extra stages preserve whatever size
    # the main stack landed on.
    strides = _conv_stage_strides(tuple(conv_channels)) + [1] * len(extra_conv_channels)
    channels = list(conv_channels) + list(extra_conv_channels)
    in_channels = [3] + channels[:-1]

    # kh/kw/padding are carried explicitly (always 3/3/1 for this model --
    # it has no 1x1 convs) because render_scratch_query_block, imported
    # from export_cnn_c_weights.py, reads them off this dict.
    layer_shapes = []
    h, w = rgb_height, rgb_width
    for in_ch, out_ch, stride in zip(in_channels, channels, strides):
        h_out = (h + 2 * 1 - 3) // stride + 1
        w_out = (w + 2 * 1 - 3) // stride + 1
        layer_shapes.append(dict(in_ch=in_ch, out_ch=out_ch, H=h, W=w, H_out=h_out, W_out=w_out,
                                 stride=stride, kh=3, kw=3, padding=1))
        h, w = h_out, w_out
    final_h, final_w = h, w


    conv_weights = list(npz["conv_weights"])
    conv_biases = list(npz["conv_biases"])
    conv_mults = list(npz["conv_mults"])
    conv_shifts = list(npz["conv_shifts"])
    assert len(conv_weights) == len(layer_shapes), (len(conv_weights), len(layer_shapes))

    guard = "RGB_CNN_MODEL_WEIGHTS_H"
    lines = []
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("/*")
    lines.append(" * Auto-generated by export_rgb_cnn_c_weights.py from")
    lines.append(" * python_code/output/rgb_cnn/quantized_model.npz + model_manifest.json.")
    lines.append(" * DO NOT EDIT BY HAND -- change the config and re-run")
    lines.append(" * train_rgb_cnn.py && python export_rgb_cnn_c_weights.py instead.")
    lines.append(" *")
    lines.append(" * PIXEL-DOMAIN model, NOT part of this repo's DCT-coefficient firmware")
    lines.append(" * story -- needs a full on-device JPEG-to-RGB decode the DCT models never")
    lines.append(" * do. Prepared for a separate project (see project memory); this header")
    lines.append(" * alone does not include that decode step.")
    lines.append(" *")
    lines.append(f" * Source run accuracy (test split): float={manifest['float_test_accuracy']:.4f}"
                  f"  qat={manifest['qat_test_accuracy']:.4f}"
                  f"  int8={manifest['int8_reference_test_accuracy']:.4f}"
                  f"  agreement={manifest['int8_vs_qat_agreement']:.4f}")
    lines.append(f" * PC CPU latency (float model, NOT an ESP32 estimate): {manifest['cpu_inference_ms']:.3f} ms/image")
    lines.append(f" * Capture: {manifest['capture_width']}x{manifest['capture_height']}"
                  f"  downsample_factor={manifest['downsample_factor']}  model input={rgb_width}x{rgb_height}")
    lines.append(f" * conv_channels={conv_channels}  extra_conv_channels={extra_conv_channels}")
    lines.append(" *")
    lines.append(" * Input: raw decoded RGB pixels, PLANAR [3][H][W] (R plane, G plane, B")
    lines.append(" * plane -- NOT interleaved RGB565/RGB888 the way a camera driver typically")
    lines.append(" * hands them back), uint8 0..255, already resized to MODEL_RGB_WIDTH x")
    lines.append(" * MODEL_RGB_HEIGHT if downsample_factor > 1 (see dct_common/rgb_features.py)")
    lines.append(" * for the exact resize -- LANCZOS on the PC side; matching that exactly")
    lines.append(" * on-device is a real open question this header does not answer). The")
    lines.append(" * planar-vs-interleaved and any resize/format conversion from whatever the")
    lines.append(" * camera driver actually outputs is the calling project's responsibility.")
    lines.append(" *")
    lines.append(" * Input quantization is a FIXED, exact scale -- q = pixel - 128 (int8 range")
    lines.append(" * [-128,127]), no calibrated multiply-shift table needed (unlike every DCT")
    lines.append(" * exporter): centered pixel values are already exactly int8-representable,")
    lines.append(" * see dct_common/models/rgb_cnn.py:Int8RgbCnnReference's docstring.")
    lines.append(" *")
    lines.append(" * TWO conv paths, same results either way: an ESP-NN-accelerated one when")
    lines.append(" * CONFIG_NN_OPTIMIZED && CONFIG_IDF_TARGET_ESP32S3 are both defined, else a")
    lines.append(" * portable reference that needs no ESP-IDF. Weight tables below are emitted")
    lines.append(" * in OHWI for the former and OIHW for the latter, under the same #if, so")
    lines.append(" * only one layout is ever compiled in. Every conv layer: int32 accumulate")
    lines.append(" * (bias + sum input*weight, zero-padded), per-output-channel Q31-mantissa")
    lines.append(" * requantize, clip to [0,127] (ReLU'd) except the final dense layer")
    lines.append(" * ([-128,127], raw logits).")
    lines.append(" *")
    lines.append(" * CALL model_init() ONCE AT STARTUP AND CHECK ITS RETURN. Activations are")
    lines.append(f" * ~{(3 * rgb_height * rgb_width + sum(s['out_ch'] * s['H_out'] * s['W_out'] for s in layer_shapes)) // 1024} KB")
    lines.append(" * and are heap-allocated (PSRAM on ESP32), not stack or .bss -- they do not")
    lines.append(" * fit in the S3's internal RAM. model_forward() will zero the logits and")
    lines.append(" * return early if that allocation failed.")
    lines.append(" */")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define MODEL_RGB_WIDTH {rgb_width}")
    lines.append(f"#define MODEL_RGB_HEIGHT {rgb_height}")
    lines.append(f"#define MODEL_NUM_CONV_LAYERS {len(layer_shapes)}")
    lines.append(f"#define MODEL_FINAL_CHANNELS {final_channels}")
    lines.append(f"#define MODEL_FINAL_H {final_h}")
    lines.append(f"#define MODEL_FINAL_W {final_w}")
    lines.append(f"#define MODEL_NUM_CLASSES {num_classes}")
    lines.append("")
    class_name_literals = ", ".join(f'"{c}"' for c in manifest["class_names"])
    lines.append(f"static const char *const MODEL_CLASS_NAMES[MODEL_NUM_CLASSES] = {{ {class_name_literals} }};")
    lines.append("")

    for i, (shape, w4, b, m, s) in enumerate(zip(layer_shapes, conv_weights, conv_biases, conv_mults, conv_shifts)):
        lines.append(f"/* ---- conv[{i}]: in_ch={shape['in_ch']} out_ch={shape['out_ch']} "
                      f"{shape['H']}x{shape['W']} -> {shape['H_out']}x{shape['W_out']} (stride={shape['stride']}) ---- */")
        emit_conv_layer(lines, f"CONV{i}", w4, b, m, s)

    lines.append("/* ---- output (dense) ---- */")
    lines.append(format_2d("OUTPUT_WEIGHT", "int8_t", npz["output_weight"]))
    lines.append(format_1d("OUTPUT_BIAS", "int32_t", npz["output_bias"]))
    lines.append(format_1d("OUTPUT_MULT", "int32_t", npz["output_mult"]))
    lines.append(format_1d("OUTPUT_SHIFT", "int32_t", npz["output_shift"]))
    lines.append("")

    lines.append(render_c_reference(guard, layer_shapes))

    out_path = out_dir / "model_weights.h"
    header_text = "\n".join(lines) + "\n"
    out_path.write_text(header_text)
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)  "
          f"{len(layer_shapes)} conv layers, input {rgb_width}x{rgb_height}, num_classes={num_classes}")

    esp32_dir = out_dir / "esp32"
    esp32_dir.mkdir(exist_ok=True)
    (esp32_dir / "model_weights.h").write_text(header_text)
    network_info = dict(manifest, architecture="rgb_cnn", weights_header="model_weights.h")
    (esp32_dir / "network_info.json").write_text(json.dumps(network_info, indent=2) + "\n")
    print(f"Wrote {esp32_dir}/ (model_weights.h + network_info.json -- self-contained deployment bundle)")


def render_c_reference(guard: str, layer_shapes: list) -> str:
    parts = []
    parts.append("""\
/* Requantize/clip helpers -- common to both the ESP-NN and portable paths
 * (the accelerated conv requantizes internally, but the final dense layer
 * below uses these in every build). */

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
""")

    parts.append(render_esp_nn_conv2d(layer_shapes))

    parts.append("""\

/* Portable bit-exact int8 reference -- no ESP-IDF/ESP-NN dependency,
 * compiles and runs anywhere (this is the path the PC verification
 * checks). Same signature as the accelerated version above so
 * model_forward()'s call sites are identical in both builds. int32
 * accumulation (zero-padded), per-output-channel Q31-mantissa requantize,
 * clip.
 *
 * ACTIVATIONS ARE NHWC: input is [H][W][in_ch], output [h_out][w_out][out_ch]
 * -- channel is the fastest-varying axis, matching what esp_nn requires so
 * the accelerated path needs no transpose. Weights stay OIHW here
 * ([out_ch][in_ch][kh][kw]); the ESP-NN build gets an OHWI copy of the same
 * values instead, selected by the #if in the weight tables above. */
static void conv2d_int8(
    const int8_t *input, int in_ch, int H, int W,
    const int8_t *weight, const int32_t *bias, const int32_t *mult, const int32_t *shift,
    int out_ch, int kh, int kw, int stride, int padding, int out_min, int out_max,
    int8_t *output)
{
    int h_out = (H + 2 * padding - kh) / stride + 1;
    int w_out = (W + 2 * padding - kw) / stride + 1;
    for (int oy = 0; oy < h_out; oy++) {
        for (int ox = 0; ox < w_out; ox++) {
            for (int oc = 0; oc < out_ch; oc++) {
                int32_t acc = bias[oc];
                for (int ky = 0; ky < kh; ky++) {
                    int iy = oy * stride + ky - padding;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < kw; kx++) {
                        int ix = ox * stride + kx - padding;
                        if (ix < 0 || ix >= W) continue;
                        for (int ic = 0; ic < in_ch; ic++) {
                            int8_t in_val = input[(iy * W + ix) * in_ch + ic];
                            int8_t w_val = weight[((oc * in_ch + ic) * kh + ky) * kw + kx];
                            acc += (int32_t)in_val * (int32_t)w_val;
                        }
                    }
                }
                int64_t product = (int64_t)acc * (int64_t)mult[oc];
                int32_t q = model_round_shift(product, 31 - shift[oc]);
                output[(oy * w_out + ox) * out_ch + oc] = (int8_t)model_clip(q, out_min, out_max);
            }
        }
    }
}

#endif /* ESP-NN vs portable conv2d_int8 */

/* Global average pool over an NHWC activation [H][W][C] -> [C] (exact
 * integer average, round to nearest, half away from zero -- matches
 * dct_common/quantization.py:avg_pool_int_exact). Accumulators are held
 * per channel so the input is walked once in memory order rather than C
 * times with a stride, which matters on a PSRAM-resident buffer. */
static void global_avg_pool_int8(const int8_t *input, int C, int H, int W, int8_t *output) {
    int32_t denom = H * W;
    int32_t sums[MODEL_FINAL_CHANNELS];
    for (int c = 0; c < C; c++) sums[c] = 0;
    for (int i = 0; i < H * W; i++)
        for (int c = 0; c < C; c++)
            sums[c] += input[i * C + c];
    for (int c = 0; c < C; c++) {
        int32_t sum = sums[c];
        int32_t abs_sum = sum < 0 ? -sum : sum;
        int32_t rounded = (abs_sum + denom / 2) / denom;
        output[c] = (int8_t)(sum < 0 ? -rounded : rounded);
    }
}

static int model_argmax(const int8_t *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}""")

    q_in_bytes = 3 * layer_shapes[0]["H"] * layer_shapes[0]["W"]
    act_bytes = [s["out_ch"] * s["H_out"] * s["W_out"] for s in layer_shapes]
    total_kb = (q_in_bytes + sum(act_bytes)) // 1024

    body = []
    body.append("")
    body.append("/* ---- activation buffers ------------------------------------------------")
    body.append(" *")
    body.append(f" * ~{total_kb} KB of activations for this model -- far past what the ESP32-S3's")
    body.append(" * ~512 KB of internal RAM can hold as stack locals or .bss, so these are")
    body.append(" * allocated once at init and reused. On ESP32 they go to PSRAM; off-target")
    body.append(" * (PC verification builds) plain malloc, so the portable path still")
    body.append(" * compiles and runs anywhere with no ESP-IDF dependency.")
    body.append(" *")
    body.append(" * PSRAM is markedly slower than internal SRAM. That penalty is real and is")
    body.append(" * part of what an RGB-vs-DCT latency comparison is measuring -- it is")
    body.append(" * deliberately not engineered around here.")
    body.append(" */")
    # Allocation order: descending read-traffic-per-byte, so the hottest
    # buffers get first claim on scarce internal SRAM. Traffic per byte for a
    # buffer is set by whichever layer CONSUMES it: each element is re-read
    # once per output channel per kernel tap, thinned by stride^2. The final
    # conv output is only read once, by the pool, so it ranks last. Buffers
    # too large to ever fit internal (conv0's output at 307 KB) fall back to
    # PSRAM on their own via the reserve check -- no special-casing needed.
    buffers = [dict(name="model_q_in", bytes=q_in_bytes, consumer=layer_shapes[0])]
    for i, nbytes in enumerate(act_bytes):
        consumer = layer_shapes[i + 1] if i + 1 < len(layer_shapes) else None
        buffers.append(dict(name=f"model_conv{i}_out", bytes=nbytes, consumer=consumer))
    for b in buffers:
        c = b["consumer"]
        b["traffic"] = (c["out_ch"] * c["kh"] * c["kw"] / (c["stride"] ** 2)) if c else 1.0
    alloc_order = sorted(buffers, key=lambda b: -b["traffic"])

    body.append("/* Placement policy: prefer internal SRAM, fall back to PSRAM, keeping")
    body.append(" * MODEL_INTERNAL_RESERVE_BYTES of internal free for WiFi/lwIP and the rest")
    body.append(" * of the firmware. Override that reserve by #define-ing it before including")
    body.append(" * this header. Buffers are requested hottest-first (read traffic per byte,")
    body.append(" * computed at export time from the consuming layer's shape) so the scarce")
    body.append(" * internal budget goes where it buys the most.")
    body.append(" *")
    body.append(" * Placement is RECORDED, not just performed -- model_buffer_report() lets")
    body.append(" * the firmware print where each buffer actually landed. For a latency")
    body.append(" * comparison this matters as much as the timing: an inference time is not")
    body.append(" * interpretable without knowing which activations were in PSRAM. */")
    body.append("#ifndef MODEL_INTERNAL_RESERVE_BYTES")
    body.append("#define MODEL_INTERNAL_RESERVE_BYTES (80 * 1024)")
    body.append("#endif")
    body.append("")
    body.append("#if defined(ESP_PLATFORM)")
    body.append('#include "esp_heap_caps.h"')
    body.append("#else")
    body.append("#include <stdlib.h>")
    body.append("#endif")
    body.append("")
    body.append("typedef struct {")
    body.append("    const char *name;")
    body.append("    int bytes;")
    body.append("    int internal;   /* 1 = internal SRAM, 0 = PSRAM (or host malloc) */")
    body.append("} model_buffer_info_t;")
    body.append("")
    n_buffers = len(buffers)
    body.append(f"#define MODEL_NUM_BUFFERS {n_buffers}")
    body.append("")
    body.append("static model_buffer_info_t model_buffer_log[MODEL_NUM_BUFFERS];")
    body.append("static int model_buffer_count = 0;")
    body.append("")
    body.append("/* Allocate one buffer, preferring internal SRAM. Records where it landed. */")
    body.append("static void *model_alloc_buffer(const char *name, int nbytes) {")
    body.append("    void *p = 0;")
    body.append("    int internal = 0;")
    body.append("#if defined(ESP_PLATFORM)")
    body.append("    /* 16-byte aligned deliberately. With NHWC activations these buffers are")
    body.append("     * passed straight into esp_nn_conv_s8, i.e. into the esp32s3 assembly")
    body.append("     * kernel, and this library's known failure mode is SILENT output")
    body.append("     * corruption on a misaligned pointer rather than a crash (see")
    body.append("     * model_esp_nn_init below). heap_caps_malloc guarantees only 8 bytes;")
    body.append("     * large blocks often happen to come back 16-aligned, which would make")
    body.append("     * this work by luck and fail on an allocation-pattern change. Cheap to")
    body.append("     * just guarantee it. */")
    body.append("    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);")
    body.append("    if ((size_t)nbytes + (size_t)MODEL_INTERNAL_RESERVE_BYTES <= free_internal) {")
    body.append("        p = heap_caps_aligned_alloc(16, nbytes, MALLOC_CAP_INTERNAL);")
    body.append("        internal = (p != 0);")
    body.append("    }")
    body.append("    if (!p) p = heap_caps_aligned_alloc(16, nbytes, MALLOC_CAP_SPIRAM);")
    body.append("#else")
    body.append("    p = malloc(nbytes);")
    body.append("#endif")
    body.append("    if (model_buffer_count < MODEL_NUM_BUFFERS) {")
    body.append("        model_buffer_log[model_buffer_count].name = name;")
    body.append("        model_buffer_log[model_buffer_count].bytes = nbytes;")
    body.append("        model_buffer_log[model_buffer_count].internal = internal;")
    body.append("        model_buffer_count++;")
    body.append("    }")
    body.append("    return p;")
    body.append("}")
    body.append("")
    body.append("/* Where each buffer landed. Valid after model_init(); *count is set to the")
    body.append(" * number of entries. */")
    body.append("static const model_buffer_info_t *model_buffer_report(int *count) {")
    body.append("    if (count) *count = model_buffer_count;")
    body.append("    return model_buffer_log;")
    body.append("}")
    body.append("")
    for b in buffers:
        body.append(f"static int8_t *{b['name']} = 0;  /* {b['bytes']} bytes */")
    body.append("static int model_buffers_ready = 0;")
    body.append("")
    body.append("/* Allocate every buffer model_forward() needs. Returns 1 on success, 0 if")
    body.append(" * any allocation failed. Safe to call more than once (no-op after the")
    body.append(" * first success). Call this once at startup and CHECK THE RETURN -- if it")
    body.append(" * fails there is not enough memory and model_forward() cannot run. */")
    body.append("static int model_init(void) {")
    body.append("    if (model_buffers_ready) return 1;")
    for b in alloc_order:
        body.append(f'    {b["name"]} = (int8_t *)model_alloc_buffer("{b["name"][6:]}", {b["bytes"]});'
                    f'  /* traffic/byte ~{b["traffic"]:.0f} */')
    body.append("    if (" + " || ".join(f"!{b['name']}" for b in buffers) + ") return 0;")
    body.append("    model_buffers_ready = 1;")
    body.append("    return 1;")
    body.append("}")
    body.append("")
    stage_names = ["quantize"] + [f"conv{i}" for i in range(len(layer_shapes))] + ["pool", "dense"]
    body.append("/* ---- per-stage profiling hooks -----------------------------------------")
    body.append(" *")
    body.append(" * No-ops unless the firmware defines them BEFORE including this header, so")
    body.append(" * they cost nothing in a normal build. They exist because the whole point")
    body.append(" * of this model on hardware is a per-layer latency breakdown: the")
    body.append(" * hypothesis under test is that conv0 costs far more than its ~11% share of")
    body.append(" * total MACs, and a single aggregate inference time cannot answer that.")
    body.append(" *")
    body.append(" * Example:")
    body.append(" *   static uint32_t t0, stage_us[MODEL_NUM_STAGES];")
    body.append(" *   #define MODEL_PROFILE_BEGIN(s) (t0 = micros())")
    body.append(" *   #define MODEL_PROFILE_END(s)   (stage_us[s] += micros() - t0)")
    body.append(" *   #include \"model_weights.h\"")
    body.append(" */")
    body.append("enum {")
    for i, name in enumerate(stage_names):
        body.append(f"    MODEL_STAGE_{name.upper()} = {i},")
    body.append(f"    MODEL_NUM_STAGES = {len(stage_names)}")
    body.append("};")
    body.append("")
    literals = ", ".join(f'"{n}"' for n in stage_names)
    body.append(f"static const char *const MODEL_STAGE_NAMES[MODEL_NUM_STAGES] = {{ {literals} }};")
    body.append("")
    body.append("#ifndef MODEL_PROFILE_BEGIN")
    body.append("#define MODEL_PROFILE_BEGIN(stage) ((void)0)")
    body.append("#endif")
    body.append("#ifndef MODEL_PROFILE_END")
    body.append("#define MODEL_PROFILE_END(stage) ((void)0)")
    body.append("#endif")
    body.append("")
    body.append("/* Full forward pass: planar uint8 RGB pixels -> int8 logits. Caller owns")
    body.append(" * decode/resize/planar-conversion -- see this header's top comment.")
    body.append(" * Calls model_init() lazily; if allocation fails the logits are zeroed and")
    body.append(" * the call returns without computing anything, so call model_init()")
    body.append(" * explicitly at boot and check its return rather than relying on this.")
    body.append(" *")
    body.append(" * Every stage is bracketed by MODEL_PROFILE_BEGIN/END. The conv primitives")
    body.append(" * and CONV<i>_* weight tables are also individually callable if you want a")
    body.append(" * separately instrumented path rather than these hooks. */")
    body.append("static void model_forward(")
    body.append("    const uint8_t rgb[3][MODEL_RGB_HEIGHT][MODEL_RGB_WIDTH],")
    body.append("    int8_t logits[MODEL_NUM_CLASSES])")
    body.append("{")
    body.append("    if (!model_init()) {")
    body.append("        for (int j = 0; j < MODEL_NUM_CLASSES; j++) logits[j] = 0;")
    body.append("        return;")
    body.append("    }")
    body.append("    /* Caller's rgb[] is planar [3][H][W]; the model wants NHWC, so this")
    body.append("     * loop centers and transposes in one pass. It is the ONLY layout change")
    body.append("     * in the whole forward pass -- everything downstream stays NHWC. */")
    body.append("    MODEL_PROFILE_BEGIN(MODEL_STAGE_QUANTIZE);")
    body.append("    for (int y = 0; y < MODEL_RGB_HEIGHT; y++)")
    body.append("        for (int x = 0; x < MODEL_RGB_WIDTH; x++)")
    body.append("            for (int c = 0; c < 3; c++)")
    body.append("                model_q_in[(y * MODEL_RGB_WIDTH + x) * 3 + c] = (int8_t)((int32_t)rgb[c][y][x] - 128);")
    body.append("    MODEL_PROFILE_END(MODEL_STAGE_QUANTIZE);")
    body.append("")

    prev_buf = "model_q_in"
    for i, shape in enumerate(layer_shapes):
        out_buf = f"model_conv{i}_out"
        body.append(f"    MODEL_PROFILE_BEGIN(MODEL_STAGE_CONV{i});")
        body.append(f"    conv2d_int8({prev_buf}, {shape['in_ch']}, {shape['H']}, {shape['W']},")
        body.append(f"                CONV{i}_WEIGHT, CONV{i}_BIAS, CONV{i}_MULT, CONV{i}_SHIFT,")
        body.append(f"                {shape['out_ch']}, {shape['kh']}, {shape['kw']}, {shape['stride']}, "
                    f"{shape['padding']}, 0, 127, {out_buf});")
        body.append(f"    MODEL_PROFILE_END(MODEL_STAGE_CONV{i});")
        body.append("")
        prev_buf = out_buf

    body.append("    int8_t pooled[MODEL_FINAL_CHANNELS];")
    body.append("    MODEL_PROFILE_BEGIN(MODEL_STAGE_POOL);")
    body.append(f"    global_avg_pool_int8({prev_buf}, MODEL_FINAL_CHANNELS, MODEL_FINAL_H, MODEL_FINAL_W, pooled);")
    body.append("    MODEL_PROFILE_END(MODEL_STAGE_POOL);")
    body.append("")
    body.append("    MODEL_PROFILE_BEGIN(MODEL_STAGE_DENSE);")
    body.append("    for (int j = 0; j < MODEL_NUM_CLASSES; j++) {")
    body.append("        int32_t acc = OUTPUT_BIAS[j];")
    body.append("        for (int i = 0; i < MODEL_FINAL_CHANNELS; i++) {")
    body.append("            acc += (int32_t)pooled[i] * (int32_t)OUTPUT_WEIGHT[j][i];")
    body.append("        }")
    body.append("        int64_t product = (int64_t)acc * (int64_t)OUTPUT_MULT[j];")
    body.append("        int32_t q = model_round_shift(product, 31 - OUTPUT_SHIFT[j]);")
    body.append("        logits[j] = (int8_t)model_clip(q, -128, 127); /* raw logits, no activation */")
    body.append("    }")
    body.append("    MODEL_PROFILE_END(MODEL_STAGE_DENSE);")
    body.append("}")
    parts.append("\n".join(body))
    parts.append(f"\n#endif /* {guard} */")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
