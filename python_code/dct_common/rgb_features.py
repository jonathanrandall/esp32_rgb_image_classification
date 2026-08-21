"""Plain RGB pixel extraction -- for train_rgb_cnn.py's pixel-domain
baseline only. Unlike everything in features.py, this decodes the JPEG to
pixels (jpeglib.read_dct() never does) -- deliberately out of scope for
the rest of this project (see python_code/README.md's opening line: "read
image class directly from quantized JPEG DCT coefficients (never decoding
to pixels)"), which is exactly why this lives in its own module instead of
inside features.py.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def extract_rgb_pixels(jpg_path: Path, width: int, height: int) -> np.ndarray:
    """Decode `jpg_path` to RGB and return a [3, height, width] int32 array
    of CENTERED pixel values (raw 0..255 pixel minus 128, range
    [-128, 127]) -- already exactly int8-representable, no further
    quantization needed (see train_rgb_cnn.py's "Input quantization" for
    why this is simpler than the DCT models' percentile-calibrated
    scales). Resized to (width, height) with LANCZOS resampling first if
    that doesn't already match the source image's size -- this is where
    --downsample-factor's effect actually happens, downstream of
    ensure_dataset()'s own capture-resolution build."""
    with Image.open(jpg_path) as img:
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.int32)  # (H, W, 3), 0..255
    return arr.transpose(2, 0, 1) - 128  # (3, H, W), centered to [-128, 127]


def extract_rgb_blocks(jpg_path: Path, block_w: int, block_h: int) -> np.ndarray:
    """Decode `jpg_path` and reduce it by averaging each block_h x block_w
    block of pixels to a single value. Returns [3, H/block_h, W/block_w]
    int32, centered to [-128, 127] exactly as extract_rgb_pixels does.

    This is the pixel-domain equivalent of the DCT arm's DC plane: the DC
    coefficient of an 8x8 block IS that block's mean, up to a constant
    scale. At the default 8x8 it therefore produces the same spatial grid
    the compressed-domain model works on (20x15 at 160x120 capture), which
    makes it the equal-resolution control the paper's limitations call for
    -- "an equivalent-resolution pixel control remains important future
    work".

    Deliberately a box mean rather than the LANCZOS resize
    extract_rgb_pixels uses. Two reasons: it matches what the DCT arm
    actually computes, and it is one add per pixel plus a shift on the
    device, where LANCZOS is not implementable at any sensible cost.
    """
    with Image.open(jpg_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.int32)  # (H, W, 3)
    h, w = arr.shape[:2]
    if h % block_h or w % block_w:
        raise ValueError(f"{w}x{h} is not divisible by block {block_w}x{block_h}")
    oh, ow = h // block_h, w // block_w
    # (H, W, 3) -> (oh, block_h, ow, block_w, 3) -> mean over the block axes
    blocks = arr.reshape(oh, block_h, ow, block_w, 3)
    means = blocks.mean(axis=(1, 3))                     # (oh, ow, 3)
    return np.rint(means).astype(np.int32).transpose(2, 0, 1) - 128
