"""JPEG DCT coefficient extraction -- shared dequantization, two layouts.

`jpeglib.read_dct()` reads the **quantized** coefficients straight from the
JPEG bitstream (no inverse DCT, no pixel reconstruction). Every extractor
here dequantizes by each image's own quantization table
(`raw_level * quant_table[position]`, via `get_component_qt()`) before
selecting zig-zag positions, so magnitudes are physically comparable across
images encoded with different quality settings.

Two output layouts, one per model family:
- `extract_flat_features` (MLP): every block's coefficients concatenated
  into one 1D vector -- all Y, then all Cb, then all Cr; within each
  component, block row -> block column -> zig-zag coefficient.
- `extract_y_dct_planes` / `extract_cbcr_dct_planes` (CNN): coefficients
  kept on their natural block grid, one 2D "plane" per zig-zag position,
  Y and chroma returned separately since they're different resolutions.
"""

from pathlib import Path

import jpeglib
import numpy as np

from .config import Config
from .zigzag import SCAN_ORDERS, zigzag_extract


def component_slices(cfg: Config) -> dict:
    """Y is always present; Cb/Cr are only part of the flat layout (and
    only included in this dict) when cfg.use_chroma -- mirrors the CNN's
    use_chroma toggle, extended to the MLP's flat feature vector (which
    has no separate chroma-fusion branch to switch off, so the feature
    layout itself has to change instead). See extract_flat_features."""
    y_len = cfg.y_features
    slices = {"Y": slice(0, y_len)}
    if cfg.use_chroma:
        c_len = cfg.c_features
        slices["Cb"] = slice(y_len, y_len + c_len)
        slices["Cr"] = slice(y_len + c_len, y_len + 2 * c_len)
    return slices


def extract_flat_features(jpeg_path: Path, cfg: Config) -> np.ndarray:
    """Return shape [cfg.input_size], dtype int32 -- DEQUANTIZED DCT
    coefficients, flattened (MLP layout): all Y block features, then (if
    cfg.use_chroma) all Cb, then all Cr. use_chroma=False drops Cb/Cr from
    the feature vector entirely (not just zeroed -- input_size shrinks to
    match), for the same reason the CNN's chroma-fusion branch can be
    switched off: the OV2640's chroma channel isn't captured reliably on
    this camera hardware, see issues.md."""
    im = jpeglib.read_dct(str(jpeg_path))

    if im.Y.shape[:2] != (cfg.y_rows, cfg.y_cols):
        raise ValueError(f"Unexpected Y block grid {im.Y.shape[:2]}, expected {(cfg.y_rows, cfg.y_cols)}")

    components = ["Y"]
    if cfg.use_chroma:
        if im.Cb.shape[:2] != (cfg.c_rows, cfg.c_cols):
            raise ValueError(f"Unexpected Cb block grid {im.Cb.shape[:2]}, expected {(cfg.c_rows, cfg.c_cols)}")
        if im.Cr.shape[:2] != (cfg.c_rows, cfg.c_cols):
            raise ValueError(f"Unexpected Cr block grid {im.Cr.shape[:2]}, expected {(cfg.c_rows, cfg.c_cols)}")
        components += ["Cb", "Cr"]

    order = SCAN_ORDERS[cfg.coeff_scan_order]
    component_arrays = {"Y": im.Y, "Cb": im.Cb, "Cr": im.Cr}
    component_qt_index = {"Y": 0, "Cb": 1, "Cr": 2}
    component_num_coeffs = {"Y": cfg.num_coeffs, "Cb": cfg.num_chroma_coeffs, "Cr": cfg.num_chroma_coeffs}

    feature_chunks = []
    for comp_name in components:
        component = component_arrays[comp_name]
        qt = im.get_component_qt(component_qt_index[comp_name]).astype(np.int32)
        n_coeffs = component_num_coeffs[comp_name]
        rows, cols = component.shape[:2]
        for by in range(rows):
            for bx in range(cols):
                dequantized_block = component[by, bx].astype(np.int32) * qt
                feature_chunks.append(zigzag_extract(dequantized_block, n_coeffs, order=order))

    features = np.concatenate(feature_chunks).astype(np.int32)
    assert features.shape == (cfg.input_size,), features.shape
    return features


def extract_y_dct_planes(jpeg_path: Path, cfg: Config) -> np.ndarray:
    """Return shape [cfg.num_coeffs, cfg.y_rows, cfg.y_cols], dtype int32 --
    DEQUANTIZED Y-channel DCT coefficients, one 2D plane per zig-zag
    position, still on the block grid (never flattened). Plane 0 is the DC
    plane -- an 8x-downsampled luma thumbnail of the frame."""
    im = jpeglib.read_dct(str(jpeg_path))

    if im.Y.shape[:2] != (cfg.y_rows, cfg.y_cols):
        raise ValueError(f"Unexpected Y block grid {im.Y.shape[:2]}, expected {(cfg.y_rows, cfg.y_cols)}")

    order = SCAN_ORDERS[cfg.coeff_scan_order]
    qt = im.get_component_qt(0).astype(np.int32)
    planes = np.zeros((cfg.num_coeffs, cfg.y_rows, cfg.y_cols), dtype=np.int32)
    for by in range(cfg.y_rows):
        for bx in range(cfg.y_cols):
            dequantized_block = im.Y[by, bx].astype(np.int32) * qt
            planes[:, by, bx] = zigzag_extract(dequantized_block, cfg.num_coeffs, order=order)
    return planes


def extract_cbcr_dct_planes(jpeg_path: Path, cfg: Config) -> np.ndarray:
    """Return shape [2 * cfg.num_chroma_coeffs, cfg.c_rows, cfg.c_cols],
    dtype int32 -- DEQUANTIZED Cb then Cr coefficient planes, same zig-zag
    positions as the Y planes (though possibly fewer of them --
    num_chroma_coeffs is independent of the Y planes' num_coeffs), on
    their native 4:2:0/4:2:2 half-resolution block grid."""
    im = jpeglib.read_dct(str(jpeg_path))
    order = SCAN_ORDERS[cfg.coeff_scan_order]
    component_arrays = {"Cb": im.Cb, "Cr": im.Cr}
    component_qt_index = {"Cb": 1, "Cr": 2}
    n_coeffs = cfg.num_chroma_coeffs

    planes = np.zeros((2 * n_coeffs, cfg.c_rows, cfg.c_cols), dtype=np.int32)
    for i, comp_name in enumerate(("Cb", "Cr")):
        component = component_arrays[comp_name]
        if component.shape[:2] != (cfg.c_rows, cfg.c_cols):
            raise ValueError(f"Unexpected {comp_name} block grid {component.shape[:2]}, expected {(cfg.c_rows, cfg.c_cols)}")
        qt = im.get_component_qt(component_qt_index[comp_name]).astype(np.int32)
        for by in range(cfg.c_rows):
            for bx in range(cfg.c_cols):
                dequantized_block = component[by, bx].astype(np.int32) * qt
                planes[i * n_coeffs:(i + 1) * n_coeffs, by, bx] = zigzag_extract(dequantized_block, n_coeffs, order=order)
    return planes
