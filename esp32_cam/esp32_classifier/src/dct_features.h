#pragma once
#include <stdint.h>
#include <stddef.h>

// Extracts the DC coefficient plus the first DCT_NUM_AC_COEFFS zig-zag AC
// coefficients of every Y (luma) 8x8 block from a baseline JPEG, fully
// dequantized by that JPEG's own quantization table. Every AC symbol in a
// block is still entropy-decoded up to EOB (required to stay
// bitstream-synced -- JPEG's variable-length codes mean you can't skip a
// block without decoding it), but only zig-zag positions < DCT_NUM_COEFFS
// are kept; the rest are discarded. See jpeg_capture.md and
// minimal_decoder.md for why you can't skip straight to a coefficient
// position instead.
//
// The Cb/Cr (chroma) blocks are also fully decoded (still required for MCU
// sync regardless of whether their values are used), and their DC
// coefficient (dequantized the same way as luma's) is stored into
// chroma_planes -- chroma AC is not extracted (this project's current
// model is chroma-DC-only; see esp32_chroma_extraction_prompt.md for why
// decode_block() isn't generalized further than that).
//
// Fixed to exactly what the OV2640 produces at FRAMESIZE_QQVGA: 160x120,
// baseline (SOF0/SOF1), 3 components, 8-bit precision, 4:2:2 chroma
// subsampling (Y H=2,V=1 -- confirmed on real hardware at both 96x96 and
// 160x120, see jpeg_capture.md Sec7; not configurable via esp32-camera).
// The exact chroma sampling factors only matter here to know how many
// Cb/Cr blocks to walk past per MCU to stay synced -- their values are
// never extracted or dequantized, since this model doesn't use them at all.
//
// dc_plane[by][bx]         -- dequantized Y-block DC (zig-zag position 0).
// ac_planes[k][by][bx]     -- dequantized Y-block AC at zig-zag position
//                              k+1, for k in 0..DCT_NUM_AC_COEFFS-1.
// chroma_planes[0][my][mx] -- dequantized Cb-block DC.
// chroma_planes[1][my][mx] -- dequantized Cr-block DC.
// dc_plane/ac_planes match dct_common/features.py's extract_y_dct_planes
// layout exactly; chroma_planes matches extract_cbcr_dct_planes exactly
// (one Cb/Cr block per MCU under this project's fixed 4:2:2 sampling, so
// the MCU grid (mcu_rows x mcu_cols) IS the chroma plane's (my, mx) grid --
// DCT_C_ROWS/DCT_C_COLS below).
//
// Returns true on success. On failure (malformed JPEG, unexpected
// dimensions/sampling, unsupported marker), returns false and sets *err to
// a short diagnostic string (safe to Serial.println() directly).

// 160x120 (FRAMESIZE_QQVGA) at 4:2:2 -- 15x20 luma blocks, 15x10 chroma.
// For 320x240 (FRAMESIZE_QVGA) these are 30 / 40 / 30 / 20. They must agree
// with the exported model_weights.h's MODEL_Y_ROWS/COLS and MODEL_C_ROWS/COLS,
// and with the sensor->set_framesize() call in main.cpp -- static_asserts in
// main.cpp enforce the first of those, nothing enforces the second.
#define DCT_Y_ROWS 15
#define DCT_Y_COLS 20
#define DCT_C_ROWS 15
#define DCT_C_COLS 10
#define DCT_NUM_AC_COEFFS 3
#define DCT_NUM_COEFFS (1 + DCT_NUM_AC_COEFFS)

bool dct_extract_coeffs(const uint8_t *jpeg, size_t len,
                         int32_t dc_plane[DCT_Y_ROWS][DCT_Y_COLS],
                         int32_t ac_planes[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS],
                         int32_t chroma_planes[2][DCT_C_ROWS][DCT_C_COLS],
                         const char **err);
