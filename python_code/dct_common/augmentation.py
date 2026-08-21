"""Pixel-domain data augmentation -- shared by train_rgb_cnn.py (applied
live, once per epoch, directly to decoded pixels) and dct_common/splits.py's
`--use-augmentation` path for the DCT models (applied once, offline, to a
copy of each training image, which is then re-encoded to JPEG and
re-extracted -- see splits.py's `_augmented_jpeg_path` docstring for why
the DCT path can't just perturb the already-extracted coefficients
directly).

Implements the operation set `esp32_jpeg_coefficient_mlp_guide.md`'s
"Recommended augmentation" row calls for -- crop, small rotation,
brightness/contrast, mild blur, plus a horizontal flip -- as one shared
PIL-based transform, since no other dependency does this more simply
(torchvision isn't installed in this project's venv, see
`reference_fiftyone_local_cache` project notes).
"""

import random

from PIL import Image, ImageEnhance, ImageFilter, ImageStat


def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply one random augmentation pass to `img` (a PIL Image, any mode
    -- caller should already have `.convert("RGB")`'d it), returning a NEW
    image of the SAME size (width, height unchanged) -- callers that feed
    this into a fixed-size pipeline (JPEG re-encode -> DCT extraction with
    a fixed block grid, or a fixed-shape CNN input) depend on this.

    Order: random horizontal flip, crop-then-resize (acts as a combined
    zoom/translate jitter), small rotation, brightness, contrast, then a
    coin-flip mild blur (stands in for the guide's "JPEG quality
    variation" -- actually varying JPEG quality is the caller's job, via
    the quality it re-encodes with, not this function's; blur is a cheap
    proxy for "slightly softer/lower-fidelity capture" in the meantime).

    Rotation's empty corners are filled with the image's own mean color
    (via ImageStat) rather than black -- a solid black wedge in the
    corner would otherwise be a strong, artificial, class-independent
    edge signal a small CNN could latch onto."""
    width, height = img.size

    if rng.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # Zoom/translate jitter: crop a slightly smaller random region, then
    # resize back up to the original size. crop_frac stays close to 1.0
    # (0.85-1.0) so a real subject doesn't get cropped out of frame.
    crop_frac = rng.uniform(0.85, 1.0)
    crop_w, crop_h = max(1, int(width * crop_frac)), max(1, int(height * crop_frac))
    max_x, max_y = width - crop_w, height - crop_h
    x0 = rng.randint(0, max_x) if max_x > 0 else 0
    y0 = rng.randint(0, max_y) if max_y > 0 else 0
    img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((width, height), Image.Resampling.LANCZOS)

    angle = rng.uniform(-8, 8)
    avg_color = tuple(int(c) for c in ImageStat.Stat(img).mean)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=avg_color, expand=False)

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))

    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    return img
