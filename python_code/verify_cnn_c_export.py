#!/usr/bin/env python3
"""Correctness-test a CNN's exported C header against the Python int8
reference -- no ESP32 hardware needed. The CNN analogue of
verify_c_export.py (which does this for the MLP); see that file's
docstring for the general approach, unchanged here: two independent
implementations of the same fixed-point arithmetic (Python, straight off
the saved quantized_model.npz arrays -- the ground truth train_cnn.py's/
train_ac_cnn.py's own "int8 reference test accuracy" came from -- and C,
compiled with a plain compiler and run as a subprocess) must produce
bit-identical int8 logits on real test-split images before the exported
header is trusted enough to build firmware around.

Usage:
    python export_cnn_c_weights.py     # if not already run
    python verify_cnn_c_export.py               # both output/cnn and output/cnn_ac, whichever exist
    python verify_cnn_c_export.py cnn_ac         # just one
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dct_common.config import Config
from dct_common.quantization import (
    avg_pool_int_exact,
    conv2d_int8_reference,
    dense_int8_reference_batched,
    quantize_multipliers,
)
from dct_common.splits import build_spatial_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
N_SAMPLES = 25


def quantize_planes_fixedpoint(raw: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Same per-channel int8 quantization as dct_common.quantization.quantize_planes,
    but via the Q31-mantissa fixed-point (multiplier, shift) convention
    model_quantize_dc/ac/chroma actually use in C, not exact float
    division. The two occasionally differ by +-1 (same wobble
    verify_c_export.py's MLP version calls out as informational, not a
    failure) -- using this instead of quantize_planes for the reference's
    *inputs* means a mismatch downstream is a real conv/pool arithmetic
    bug, not this expected, harmless approximation surfacing as a false
    positive. raw: int32/int64 [N, C, H, W]; scales: [C]."""
    mult, shift = quantize_multipliers(1.0 / scales)
    raw64 = raw.astype(np.int64)
    out = np.empty_like(raw64)
    for c in range(raw.shape[1]):
        total_shift = 31 - int(shift[c])
        rounding = np.int64(1) << (total_shift - 1)
        out[:, c] = (raw64[:, c] * np.int64(mult[c]) + rounding) >> total_shift
    return np.clip(out, -127, 127).astype(np.int8)


def python_int8_forward(npz, is_ac_variant: bool, q_dc, q_ac, q_chroma, num_ac_coeffs: int, use_chroma: bool):
    """Ground truth: chains dct_common.quantization's bit-exact int8 ops
    directly against the arrays saved in quantized_model.npz, mirroring
    Int8CnnReference.forward() / Int8CnnAcReference.forward() exactly."""
    h = conv2d_int8_reference(q_dc, npz["lum_weight"], npz["lum_bias"], npz["lum_mult"], npz["lum_shift"], stride=1, padding=1, output_min=0, output_max=127)
    h = conv2d_int8_reference(h, npz["stride2_weight"], npz["stride2_bias"], npz["stride2_mult"], npz["stride2_shift"], stride=2, padding=1, output_min=0, output_max=127)
    target_hw = h.shape[-2:]

    parts = [h]
    if num_ac_coeffs > 0:
        ac_pooled = avg_pool_int_exact(q_ac, target_hw).astype(np.int8)
        if is_ac_variant:
            ac_out = conv2d_int8_reference(ac_pooled, npz["ac_proj_weight"], npz["ac_proj_bias"], npz["ac_proj_mult"], npz["ac_proj_shift"], stride=1, padding=0, output_min=0, output_max=127)
        else:
            ac_out = ac_pooled
        parts.append(ac_out)
    if use_chroma:
        chroma_pooled = avg_pool_int_exact(q_chroma, target_hw).astype(np.int8)
        if is_ac_variant:
            chroma_out = conv2d_int8_reference(chroma_pooled, npz["chroma_proj_weight"], npz["chroma_proj_bias"], npz["chroma_proj_mult"], npz["chroma_proj_shift"], stride=1, padding=0, output_min=0, output_max=127)
        else:
            chroma_out = chroma_pooled
        parts.append(chroma_out)
    h = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    h = conv2d_int8_reference(h, npz["post_concat_weight"], npz["post_concat_bias"], npz["post_concat_mult"], npz["post_concat_shift"], stride=1, padding=1, output_min=0, output_max=127)

    extra_weights = list(npz["extra_weights"]) if "extra_weights" in npz.files else []
    extra_biases = list(npz["extra_biases"]) if "extra_biases" in npz.files else []
    extra_mults = list(npz["extra_mults"]) if "extra_mults" in npz.files else []
    extra_shifts = list(npz["extra_shifts"]) if "extra_shifts" in npz.files else []
    for w, b, m, s in zip(extra_weights, extra_biases, extra_mults, extra_shifts):
        h = conv2d_int8_reference(h, w, b, m, s, stride=1, padding=1, output_min=0, output_max=127)

    pooled = avg_pool_int_exact(h, (1, 1))
    flat = pooled.reshape(pooled.shape[0], pooled.shape[1])
    return dense_int8_reference_batched(flat, npz["output_weight"], npz["output_bias"], npz["output_mult"], npz["output_shift"], output_min=-128, output_max=127)


def format_nd(name: str, ctype: str, arr: np.ndarray, per_line: int = 20) -> str:
    """Recursively renders an arbitrary-rank int array as a nested C
    initializer, e.g. shape (N, C, H, W) -> `ctype name[N][C][H][W] = {...}`."""
    dims = "".join(f"[{d}]" for d in arr.shape)

    def render(a: np.ndarray, indent: str) -> str:
        if a.ndim == 1:
            values = [str(int(v)) for v in a]
            lines = [", ".join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
            return "{\n" + ",\n".join(indent + "    " + line for line in lines) + "\n" + indent + "}"
        blocks = [render(a[i], indent + "    ") for i in range(a.shape[0])]
        return "{\n" + ",\n".join(indent + "    " + b for b in blocks) + "\n" + indent + "}"

    body = render(arr, "")
    return f"static const {ctype} {name}{dims} = {body};"


def verify_one(model_dir_name: str) -> bool:
    out_dir = OUTPUT_ROOT / model_dir_name
    manifest_path = out_dir / "model_manifest.json"
    if not manifest_path.exists():
        print(f"Skipping {model_dir_name}: no {manifest_path} (train it first)")
        return True
    header_path = out_dir / "model_weights.h"
    if not header_path.exists():
        print(f"Skipping {model_dir_name}: no {header_path} -- run export_cnn_c_weights.py first.", file=sys.stderr)
        return False

    manifest = json.load(open(manifest_path))
    npz = np.load(out_dir / "quantized_model.npz", allow_pickle=True)
    is_ac_variant = "ac_proj_weight" in npz.files
    num_ac_coeffs = manifest["num_ac_coeffs"]
    use_chroma = manifest["use_chroma"]
    num_classes = manifest["num_classes"]

    cfg = Config(
        capture_width=manifest["capture_width"], capture_height=manifest["capture_height"],
        chroma_subsampling=manifest.get("chroma_subsampling", "4:2:0"),
        num_ac_coeffs=num_ac_coeffs, num_chroma_ac_coeffs=manifest.get("num_chroma_ac_coeffs"),
        use_chroma=use_chroma,
        selected_classes=tuple(manifest["class_names"]),
    )
    X_y, X_c, y_test, _ = build_spatial_split(DATA_DIR, "test", cfg.active_class_names, cfg)

    rng = np.random.default_rng(0)
    idx = rng.choice(X_y.shape[0], size=min(N_SAMPLES, X_y.shape[0]), replace=False)
    X_y, X_c = X_y[idx], X_c[idx]

    dc_scale = npz["dc_scale"]
    q_dc = quantize_planes_fixedpoint(X_y[:, 0:1], dc_scale)
    dc_raw = X_y[:, 0].astype(np.int32)  # [n, y_rows, y_cols]

    q_ac, ac_raw = None, None
    if num_ac_coeffs > 0:
        ac_scales = npz["ac_scales"]
        q_ac = quantize_planes_fixedpoint(X_y[:, 1:], ac_scales)
        ac_raw = X_y[:, 1:].astype(np.int32)  # [n, num_ac_coeffs, y_rows, y_cols]

    q_chroma, chroma_raw = None, None
    if use_chroma:
        chroma_scales = npz["chroma_scales"]
        q_chroma = quantize_planes_fixedpoint(X_c, chroma_scales)
        chroma_raw = X_c.astype(np.int32)  # [n, 2*num_coeffs, c_rows, c_cols]

    expected_logits = python_int8_forward(npz, is_ac_variant, q_dc, q_ac, q_chroma, num_ac_coeffs, use_chroma).astype(np.int32)
    expected_preds = expected_logits.argmax(axis=1)

    n = X_y.shape[0]
    lines = [
        f"#ifndef DCT_CNN_{model_dir_name.upper()}_TEST_VECTORS_H",
        f"#define DCT_CNN_{model_dir_name.upper()}_TEST_VECTORS_H",
        "/* Auto-generated by verify_cnn_c_export.py -- real test-split images.",
        " * TEST_*_RAW are the dequantized-by-its-own-JPEG-quant-table DCT",
        " * coefficient planes *before* int8 quantization (what feature",
        " * extraction produces) -- model_forward() quantizes them itself, so",
        " * this tests the full pipeline end to end, same as the MLP's verify",
        " * script. Expected logits computed by the Python int8 reference",
        " * chained directly against the arrays exported into model_weights.h. */",
        "",
        f"#define N_TEST_VECTORS {n}",
        "",
        format_nd("TEST_DC_RAW", "int32_t", dc_raw),
    ]
    if num_ac_coeffs > 0:
        lines.append(format_nd("TEST_AC_RAW", "int32_t", ac_raw))
    if use_chroma:
        lines.append(format_nd("TEST_CHROMA_RAW", "int32_t", chroma_raw))
    lines += [
        format_nd("TEST_EXPECTED_LOGITS", "int8_t", expected_logits),
        format_nd("TEST_EXPECTED_CLASS", "int32_t", expected_preds),
        "",
        f"#endif /* DCT_CNN_{model_dir_name.upper()}_TEST_VECTORS_H */",
        "",
    ]
    (out_dir / "test_vectors.h").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'test_vectors.h'} ({n} test vectors)")

    harness_lines = [
        "#include <stdio.h>",
        "#include <string.h>",
        '#include "model_weights.h"',
        '#include "test_vectors.h"',
        "",
        "int main(void) {",
        "    int failures = 0;",
        "    for (int t = 0; t < N_TEST_VECTORS; t++) {",
        "        int8_t logits[MODEL_NUM_CLASSES];",
        "        model_forward(TEST_DC_RAW[t],",
    ]
    # model_forward()'s signature always has ac_planes/chroma_planes params
    # now (see export_cnn_c_weights.py -- kept stable across --num-ac-coeffs
    # 0 / --no-chroma retrains so firmware call sites don't need editing).
    # When this model doesn't actually use one, there's no TEST_AC_RAW/
    # TEST_CHROMA_RAW array in test_vectors.h to pass -- a null pointer cast
    # to the exact parameter type satisfies the compiler and is safe, since
    # model_forward() (void)-casts and never dereferences an unused plane.
    call_args = []
    if num_ac_coeffs > 0:
        call_args.append("                      TEST_AC_RAW[t],")
    else:
        call_args.append("                      (const int32_t (*)[MODEL_Y_ROWS][MODEL_Y_COLS])0,")
    if use_chroma:
        call_args.append("                      TEST_CHROMA_RAW[t],")
    else:
        call_args.append("                      (const int32_t (*)[MODEL_C_ROWS][MODEL_C_COLS])0,")
    harness_lines += call_args
    harness_lines += [
        "                      logits);",
        "",
        "        int mismatch = memcmp(logits, TEST_EXPECTED_LOGITS[t], MODEL_NUM_CLASSES) != 0;",
        "        int pred = model_argmax(logits, MODEL_NUM_CLASSES);",
        "        int class_mismatch = pred != TEST_EXPECTED_CLASS[t];",
        "",
        "        if (mismatch || class_mismatch) {",
        "            failures++;",
        '            printf("FAIL vector %d: predicted class %d (expected %d)%s\\n",',
        '                   t, pred, TEST_EXPECTED_CLASS[t], mismatch ? ", logits differ" : "");',
        "        } else {",
        '            printf("PASS vector %d: class %d, logits bit-exact\\n", t, pred);',
        "        }",
        "    }",
        '    printf("\\n%d/%d test vectors bit-exact match end-to-end (raw planes -> fixed-point quantize -> full CNN -> logits)\\n", N_TEST_VECTORS - failures, N_TEST_VECTORS);',
        "    return failures == 0 ? 0 : 1;",
        "}",
        "",
    ]
    harness_path = out_dir / "verify_export.c"
    harness_path.write_text("\n".join(harness_lines))

    binary_path = out_dir / "verify_export"
    compile_cmd = ["gcc", "-std=c99", "-Wall", "-Wextra", "-I", str(out_dir), "-o", str(binary_path), str(harness_path)]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE FAILED ({model_dir_name}):")
        print(result.stdout)
        print(result.stderr)
        return False
    if result.stderr:
        print(f"compiler warnings ({model_dir_name}):")
        print(result.stderr)

    run_result = subprocess.run([str(binary_path)], capture_output=True, text=True)
    print(run_result.stdout)
    if run_result.returncode != 0:
        print(f"VERIFICATION FAILED ({model_dir_name}) -- C export does not match Python reference.", file=sys.stderr)
        return False
    print(f"VERIFICATION PASSED ({model_dir_name}) -- C export is bit-exact with the Python int8 reference.")
    return True


def main() -> None:
    requested = sys.argv[1:] or ["cnn", "cnn_ac"]
    ok = True
    for model_dir_name in requested:
        ok = verify_one(model_dir_name) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
