#!/usr/bin/env python3
"""Correctness-test the RGB CNN's exported C header against the Python int8
reference -- no ESP32 hardware needed. The pixel-domain counterpart of
verify_cnn_c_export.py, and it follows the same discipline: two independent
implementations of the same fixed-point arithmetic (Python, built straight
off the saved quantized_model.npz arrays, and C, compiled with a plain
compiler and run as a subprocess) must agree bit-for-bit before the exported
header is trusted enough to build firmware around.

Usage:
    python export_rgb_cnn_c_weights.py      # if not already run
    python verify_rgb_cnn_c_export.py

Writes into output/rgb_cnn/:
    rgb_synth_vectors.h   LCG + structured-pattern vectors and expected logits
    rgb_real_images.h     two real test-split images and their expected logits
    verify_rgb_export.c   the generated host harness
    verify_rgb_export     the compiled binary

Copy the two headers into esp32_cam/esp32_rgb_cnn/include/ alongside the
exported model_weights.h; the firmware re-runs the same vectors through the
ESP-NN kernels at every boot.

THREE TIERS, in ascending order of what they prove:

 1. LCG noise -- cheap smoke test, weak evidence. White noise has no spatial
    structure that survives four convolutions and a global average pool, so
    all seeds pool to nearly the same feature vector. A row-pitch mistake, a
    padding off-by-one, a stride-phase error or a channel-indexing slip would
    all still average to roughly the right pooled value. Passing these alone
    means very little.
 2. Structured patterns -- the PRIMARY geometric check. Horizontal ramp,
    vertical ramp and an 8x8 checkerboard, with per-channel asymmetry, so the
    geometry errors above cannot average themselves out. The checkerboard
    saturates the int8 rail; a transposed or misaligned implementation misses
    by tens of LSB rather than by rounding.
 3. Real photographs -- end-to-end sanity on images the PC-side model
    classifies correctly, so a mismatch means the port is wrong rather than
    the model being wrong.

WHY THE REFERENCE IS BUILT FROM THE NPZ, not from a fresh
Int8RgbCnnReference: constructing one recalibrates output_scale off the QAT
model, which shifts every logit and produces mismatches that look like port
bugs but aren't. This cost a day during the original bring-up.

NOTE ON SCOPE: this checks the PORTABLE path in the generated header, since
there is no Xtensa toolchain on the PC side. The accelerated (ESP-NN) path is
checked by the firmware's own boot self-test, which reports class agreement
and a maximum divergence of one unit in the last place -- the accelerator
requantizes with two rounding steps where this reference uses one.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dct_common.quantization import (
    avg_pool_int_exact,
    conv2d_int8_reference,
    dense_int8_reference_batched,
)
from dct_common.rgb_features import extract_rgb_pixels, extract_rgb_blocks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Directory name may be given on the command line, matching
# export_rgb_cnn_c_weights.py. Without this only output/rgb_cnn could be
# verified, so a model trained with --artifacts-name had no way to regenerate
# its test vectors -- and stale vectors are sized for the OLD model, which
# overflows the firmware's input buffer at boot.
_name = sys.argv[1] if len(sys.argv) > 1 else "rgb_cnn"
OUT_DIR = Path(__file__).resolve().parent / "output" / _name

SEEDS = [1, 2, 3, 12345]
PATTERN_NAMES = ["HRAMP", "VRAMP", "CHECKER"]


# --------------------------------------------------------------- reference
def load_reference(out_dir):
    z = np.load(out_dir / "quantized_model.npz", allow_pickle=True)
    convs = list(zip(z["conv_weights"], z["conv_biases"], z["conv_mults"],
                     z["conv_shifts"], z["conv_strides"]))
    out = (z["output_weight"], z["output_bias"], z["output_mult"], z["output_shift"])

    def forward(q_rgb):
        """Mirrors Int8RgbCnnReference._run_conv_stack + forward exactly."""
        h = q_rgb
        for w, b, mult, shift, stride in convs:
            h = conv2d_int8_reference(h, w, b, mult, shift, stride=int(stride),
                                      padding=1, output_min=0, output_max=127)
        pooled = avg_pool_int_exact(h, (1, 1))
        flat = pooled.reshape(pooled.shape[0], pooled.shape[1])
        return dense_int8_reference_batched(flat, *out, output_min=-128, output_max=127)

    return forward


# ------------------------------------------------------------- test inputs
def synth_fill(seed, h, w):
    """Numerical Recipes ranqd1 LCG, top byte. Must stay bit-identical to
    synth_fill() in the firmware's rgb_test_vectors.h -- the top byte is taken
    because the low bits of a power-of-two-modulus LCG have short periods."""
    n = 3 * h * w
    out = np.empty(n, dtype=np.uint8)
    s = seed
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (s >> 24) & 0xFF
    return out.reshape(3, h, w)


def pattern_fill(idx, h, w):
    """Must stay bit-identical to pattern_fill() in rgb_test_vectors.h.
    Note the integer division in CHECKER."""
    out = np.empty((3, h, w), dtype=np.uint8)
    for c in range(3):
        for y in range(h):
            for x in range(w):
                if idx == 0:
                    v = (x * 2 + c * 40) & 0xFF
                elif idx == 1:
                    v = (y * 3 + c * 70) & 0xFF
                else:
                    v = (c * 80 + 30) if ((x // 8 + y // 8) & 1) else (200 - c * 60)
                out[c, y, x] = v & 0xFF
    return out


def center(raw_u8):
    return (raw_u8.astype(np.int32) - 128).astype(np.int8)


# ----------------------------------------------------------------- headers
def fmt_row(row):
    return "{ " + ", ".join(f"{int(v):4d}" for v in row) + " }"


def write_synth_header(path, nc, class_names, synth_lg, pat_lg):
    L = [
        "/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.",
        " * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays",
        " * emitted into model_weights.h. Inputs are the LCG defined below, fed",
        " * planar uint8 0..255; the model centers by -128 itself.",
        " * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */",
        "",
        f"#define SYNTH_NUM_VECTORS {len(SEEDS)}",
        "static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { "
        + ", ".join(str(s) for s in SEEDS) + " };",
        "",
        f"static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][{nc}] = {{",
    ]
    for i, s in enumerate(SEEDS):
        L.append(f"    {fmt_row(synth_lg[i])}{',' if i < len(SEEDS) - 1 else ''}  /* seed {s} */")
    L += ["};", "",
          "static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { "
          + ", ".join(str(int(c)) for c in synth_lg.argmax(axis=1)) + " };",
          "",
          "/* argmax: " + ", ".join(f"seed {s} -> {class_names[int(c)]}"
                                    for s, c in zip(SEEDS, synth_lg.argmax(axis=1))) + " */",
          "",
          "/* Structured patterns -- the PRIMARY geometry check. */",
          f"#define PATTERN_NUM_VECTORS {len(PATTERN_NAMES)}",
          f"static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][{nc}] = {{"]
    for i, nm in enumerate(PATTERN_NAMES):
        L.append(f"    {fmt_row(pat_lg[i])}{',' if i < len(PATTERN_NAMES) - 1 else ''}  /* {nm} */")
    L += ["};", "",
          "static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { "
          + ", ".join(str(int(c)) for c in pat_lg.argmax(axis=1)) + " };",
          "",
          "/* argmax: " + ", ".join(f"{n} -> {class_names[int(c)]}"
                                    for n, c in zip(PATTERN_NAMES, pat_lg.argmax(axis=1))) + " */",
          ""]
    path.write_text("\n".join(L))


def write_real_header(path, nc, picked, h, w):
    L = ["/* Real test-split images for the RGB CNN end-to-end check.",
         " * Planar uint8 0..255, [channel][y][x], channel 0=R 1=G 2=B (PIL RGB",
         " * order, see dct_common/rgb_features.py). Feed straight to the model --",
         " * it centers by -128 itself. Expected logits generated from",
         " * quantized_model.npz, the same arrays emitted into model_weights.h.",
         " * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */", ""]
    for cls, _, p, arr, _ in picked:
        raw = (arr.astype(np.int32) + 128).astype(np.uint8)
        L.append(f"/* {cls}: {p.name} */")
        L.append(f"static const uint8_t REAL_IMAGE_{cls.upper()}[3][{h}][{w}] = {{")
        for c in range(3):
            L.append("  {")
            for y in range(h):
                L.append("    {" + ",".join(str(int(v)) for v in raw[c, y]) + "}"
                         + ("," if y < h - 1 else ""))
            L.append("  }" + ("," if c < 2 else ""))
        L += ["};", ""]
    L += [f"#define REAL_NUM_VECTORS {len(picked)}",
          "static const uint8_t *const REAL_IMAGES[REAL_NUM_VECTORS] = { "
          + ", ".join(f"(const uint8_t *)REAL_IMAGE_{c.upper()}" for c, _, _, _, _ in picked) + " };",
          f"static const int8_t REAL_EXPECTED_LOGITS[REAL_NUM_VECTORS][{nc}] = {{"]
    for i, (cls, _, _, _, lg) in enumerate(picked):
        L.append(f"    {fmt_row(lg)}{',' if i < len(picked) - 1 else ''}  /* {cls} */")
    L += ["};",
          "static const uint8_t REAL_EXPECTED_CLASS[REAL_NUM_VECTORS] = { "
          + ", ".join(str(ci) for _, ci, _, _, _ in picked) + " };",
          "static const char *const REAL_CLASS_NAMES[REAL_NUM_VECTORS] = { "
          + ", ".join(f'"{c}"' for c, _, _, _, _ in picked) + " };", ""]
    path.write_text("\n".join(L))


HARNESS = r'''/* Generated by verify_rgb_cnn_c_export.py -- do not edit.
 * Compiles the exported header's PORTABLE path with a plain compiler and
 * checks it against the expected logits, which came from the Python int8
 * reference. synth_fill/pattern_fill below are bit-identical to the
 * firmware's rgb_test_vectors.h. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "model_weights.h"
#include "rgb_synth_vectors.h"
#include "rgb_real_images.h"

#define RGB_BYTES (3 * MODEL_RGB_HEIGHT * MODEL_RGB_WIDTH)
static uint8_t buf[3][MODEL_RGB_HEIGHT][MODEL_RGB_WIDTH];
static const char *const PATTERN_NAMES[PATTERN_NUM_VECTORS] = {"HRAMP", "VRAMP", "CHECKER"};

static void synth_fill(uint32_t seed, uint8_t *dst) {
    uint32_t s = seed;
    for (int i = 0; i < RGB_BYTES; i++) {
        s = s * 1664525u + 1013904223u;
        dst[i] = (uint8_t)((s >> 24) & 0xFFu);
    }
}

static void pattern_fill(int idx, uint8_t *dst) {
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < MODEL_RGB_HEIGHT; y++)
            for (int x = 0; x < MODEL_RGB_WIDTH; x++) {
                int i = (c * MODEL_RGB_HEIGHT + y) * MODEL_RGB_WIDTH + x;
                switch (idx) {
                    case 0:  dst[i] = (uint8_t)((x * 2 + c * 40) & 0xFF); break;
                    case 1:  dst[i] = (uint8_t)((y * 3 + c * 70) & 0xFF); break;
                    default: dst[i] = (uint8_t)(((x / 8 + y / 8) & 1) ? (c * 80 + 30)
                                                                     : (200 - c * 60));
                }
            }
}

static int check(const char *tier, const char *name, const int8_t *got, const int8_t *want) {
    int max_diff = 0, ag = 0, aw = 0;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        int d = got[i] - want[i]; if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        if (got[i] > got[ag]) ag = i;
        if (want[i] > want[aw]) aw = i;
    }
    int ok = (max_diff == 0);
    printf("[%-7s] %-16s %s  class %s%s\n", tier, name, ok ? "bit-exact" : "MISMATCH ",
           MODEL_CLASS_NAMES[ag], ag == aw ? "" : "  (CLASS DIFFERS)");
    if (!ok) {
        printf("            got: ");
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) printf(" %d", got[i]);
        printf("\n            want:");
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) printf(" %d", want[i]);
        printf("   max diff %d\n", max_diff);
    }
    return ok;
}

int main(void) {
    int8_t logits[MODEL_NUM_CLASSES];
    int pass = 0, total = 0;
    if (!model_init()) { printf("model_init() FAILED\n"); return 2; }

    for (int t = 0; t < PATTERN_NUM_VECTORS; t++) {
        pattern_fill(t, (uint8_t *)buf);
        model_forward(buf, logits);
        pass += check("pattern", PATTERN_NAMES[t], logits, PATTERN_EXPECTED_LOGITS[t]);
        total++;
    }
    for (int t = 0; t < SYNTH_NUM_VECTORS; t++) {
        char nm[32];
        snprintf(nm, sizeof nm, "seed %u", (unsigned)SYNTH_SEEDS[t]);
        synth_fill(SYNTH_SEEDS[t], (uint8_t *)buf);
        model_forward(buf, logits);
        pass += check("synth", nm, logits, SYNTH_EXPECTED_LOGITS[t]);
        total++;
    }
    for (int t = 0; t < REAL_NUM_VECTORS; t++) {
        memcpy(buf, REAL_IMAGES[t], sizeof buf);
        model_forward(buf, logits);
        pass += check("real", REAL_CLASS_NAMES[t], logits, REAL_EXPECTED_LOGITS[t]);
        total++;
    }
    printf("\n%d/%d bit-exact against the Python int8 reference\n", pass, total);
    return pass == total ? 0 : 1;
}
'''


def main():
    if not (OUT_DIR / "quantized_model.npz").exists():
        print(f"No {OUT_DIR}/quantized_model.npz -- train train_rgb_cnn.py first")
        return 1
    if not (OUT_DIR / "model_weights.h").exists():
        print(f"No {OUT_DIR}/model_weights.h -- run export_rgb_cnn_c_weights.py first")
        return 1

    manifest = json.load(open(OUT_DIR / "model_manifest.json"))
    class_names, nc = manifest["class_names"], manifest["num_classes"]
    h, w = manifest["rgb_height"], manifest["rgb_width"]
    # A block-mean model must have its vectors built the same way the firmware
    # reduces frames, or the "expected" logits describe a different input than
    # the device ever sees.
    use_blocks = manifest.get("reduction") == "block_mean"
    bw = manifest.get("rgb_block_width", 1)
    bh = manifest.get("rgb_block_height", 1)
    extract = ((lambda p_: extract_rgb_blocks(p_, bw, bh)) if use_blocks
               else (lambda p_: extract_rgb_pixels(p_, w, h)))
    print(f"input {w}x{h}  reduction {'block mean %dx%d' % (bw, bh) if use_blocks else 'lanczos'}")
    forward = load_reference(OUT_DIR)

    synth = np.stack([center(synth_fill(s, h, w)) for s in SEEDS])
    pat = np.stack([center(pattern_fill(i, h, w)) for i in range(3)])
    synth_lg, pat_lg = forward(synth), forward(pat)
    write_synth_header(OUT_DIR / "rgb_synth_vectors.h", nc, class_names, synth_lg, pat_lg)

    picked = []
    for ci, cls in enumerate(class_names):
        d = DATA_DIR / "test" / cls
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg"))[:12]:
            arr = extract(p).astype(np.int8)[None]
            lg = forward(arr)[0]
            if int(lg.argmax()) == ci:
                picked.append((cls, ci, p, arr[0], lg))
                break
        if len(picked) == 2:
            break
    if len(picked) < 2:
        print(f"Only found {len(picked)} correctly-classified test images; need 2")
        return 1
    write_real_header(OUT_DIR / "rgb_real_images.h", nc, picked, h, w)

    harness = OUT_DIR / "verify_rgb_export.c"
    binary = OUT_DIR / "verify_rgb_export"
    harness.write_text(HARNESS)
    cc = subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-I", str(OUT_DIR),
                         "-o", str(binary), str(harness)], capture_output=True, text=True)
    if cc.returncode != 0:
        print("COMPILE FAILED\n" + cc.stderr)
        return 1
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    print(run.stdout, end="")
    if run.returncode == 0:
        print("\nVERIFICATION PASSED (rgb_cnn) -- C export is bit-exact with the "
              "Python int8 reference.\nCopy rgb_synth_vectors.h and rgb_real_images.h "
              "into esp32_cam/esp32_rgb_cnn/include/.")
    else:
        print("\nVERIFICATION FAILED (rgb_cnn)")
    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
