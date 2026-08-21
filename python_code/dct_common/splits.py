"""Walk data/<split>/<class_name>/*.jpg and stack extracted features."""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .augmentation import augment_image
from .config import Config
from .features import extract_cbcr_dct_planes, extract_flat_features, extract_y_dct_planes
from .rgb_features import extract_rgb_pixels, extract_rgb_blocks


def _augmented_jpeg_path(jpg_path: Path, cfg: Config, rng: random.Random, tmp_dir: Path) -> Path:
    """Decode jpg_path, apply one random augmentation pass
    (dct_common/augmentation.py: flip/crop/rotate/brightness/contrast/
    blur), and re-encode as a real JPEG into tmp_dir at the SAME
    quality/chroma-subsampling as the rest of this dataset
    (cfg.jpeg_quality/cfg.chroma_subsampling). Required because
    features.py reads real quantized DCT coefficients straight out of the
    JPEG bitstream (jpeglib.read_dct()), never decoding to pixels itself
    -- an augmented training example has to actually BE a re-encoded JPEG
    for that same extraction path to see coefficients that reflect the
    augmentation, not an unencoded approximation of it. Returns the temp
    file's path; caller owns tmp_dir's lifetime (features are extracted
    from this path into plain numpy arrays before tmp_dir is ever cleaned
    up, so the file doesn't need to outlive this call)."""
    with Image.open(jpg_path) as img:
        augmented = augment_image(img.convert("RGB"), rng)
    tmp_path = tmp_dir / f"{jpg_path.stem}_aug{rng.randint(0, 1_000_000_000)}.jpg"
    augmented.save(tmp_path, format="JPEG", quality=cfg.jpeg_quality, subsampling=cfg.chroma_subsampling)
    return tmp_path


def build_flat_split(data_dir: Path, split: str, class_names: list, cfg: Config,
                      augment: bool = False, augment_copies: int = 1):
    """MLP layout: returns (X, y, paths) with X.shape == (N, cfg.input_size).

    augment/augment_copies: only ever applied when split == "train" (val/
    test are never augmented, regardless of what's passed -- augmenting
    eval data would make accuracy numbers incomparable to every other run).
    When on, each original train image gets `augment_copies` extra
    augmented-and-re-encoded variants stacked in alongside it (see
    `_augmented_jpeg_path` above for why a real re-encode is required
    here, unlike train_rgb_cnn.py's live per-epoch version of this same
    idea). Off (the default) reproduces the exact previous behavior."""
    do_augment = augment and split == "train"
    xs, ys, paths = [], [], []
    rng = random.Random(cfg.seed) if do_augment else None
    n_originals = n_augmented = 0
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for class_idx, class_name in enumerate(class_names):
            class_dir = data_dir / split / class_name
            for jpg_path in sorted(class_dir.glob("*.jpg")):
                xs.append(extract_flat_features(jpg_path, cfg))
                ys.append(class_idx)
                paths.append(str(jpg_path))
                n_originals += 1
                if do_augment:
                    for _ in range(augment_copies):
                        aug_path = _augmented_jpeg_path(jpg_path, cfg, rng, tmp_dir)
                        xs.append(extract_flat_features(aug_path, cfg))
                        ys.append(class_idx)
                        paths.append(str(aug_path))
                        n_augmented += 1
    if do_augment:
        print(f"  augmentation on ({split}): {n_originals} originals + {n_augmented} augmented = {n_originals + n_augmented} total")
    X = np.stack(xs).astype(np.int32)
    y = np.array(ys, dtype=np.int64)
    assert X.shape == (len(ys), cfg.input_size)
    return X, y, paths


def build_spatial_split(data_dir: Path, split: str, class_names: list, cfg: Config,
                         augment: bool = False, augment_copies: int = 1):
    """CNN layout: returns (X_y, X_chroma, y, paths) with
    X_y.shape == (N, num_coeffs, y_rows, y_cols) and
    X_chroma.shape == (N, 2*num_chroma_coeffs, c_rows, c_cols).

    augment/augment_copies: see build_flat_split's docstring above --
    same rule (train split only), same mechanism, same off-by-default
    behavior."""
    do_augment = augment and split == "train"
    xs_y, xs_c, ys, paths = [], [], [], []
    rng = random.Random(cfg.seed) if do_augment else None
    n_originals = n_augmented = 0
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for class_idx, class_name in enumerate(class_names):
            class_dir = data_dir / split / class_name
            for jpg_path in sorted(class_dir.glob("*.jpg")):
                xs_y.append(extract_y_dct_planes(jpg_path, cfg))
                xs_c.append(extract_cbcr_dct_planes(jpg_path, cfg))
                ys.append(class_idx)
                paths.append(str(jpg_path))
                n_originals += 1
                if do_augment:
                    for _ in range(augment_copies):
                        aug_path = _augmented_jpeg_path(jpg_path, cfg, rng, tmp_dir)
                        xs_y.append(extract_y_dct_planes(aug_path, cfg))
                        xs_c.append(extract_cbcr_dct_planes(aug_path, cfg))
                        ys.append(class_idx)
                        paths.append(str(aug_path))
                        n_augmented += 1
    if do_augment:
        print(f"  augmentation on ({split}): {n_originals} originals + {n_augmented} augmented = {n_originals + n_augmented} total")
    X_y = np.stack(xs_y).astype(np.float32)
    X_c = np.stack(xs_c).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    assert X_y.shape == (len(ys), cfg.num_coeffs, cfg.y_rows, cfg.y_cols)
    assert X_c.shape == (len(ys), 2 * cfg.num_chroma_coeffs, cfg.c_rows, cfg.c_cols)
    return X_y, X_c, y, paths


def build_rgb_block_split(data_dir: Path, split: str, class_names: list,
                          block_w: int, block_h: int):
    """Block-mean equivalent of build_rgb_split -- see
    rgb_features.extract_rgb_blocks. Returns (X, y, paths) with
    X.shape == (N, 3, H/block_h, W/block_w)."""
    xs, ys, paths = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / split / class_name
        for jpg_path in sorted(class_dir.glob("*.jpg")):
            xs.append(extract_rgb_blocks(jpg_path, block_w, block_h))
            ys.append(class_idx)
            paths.append(str(jpg_path))
    X = np.stack(xs).astype(np.int32)
    y = np.array(ys, dtype=np.int64)
    return X, y, paths


def build_rgb_split(data_dir: Path, split: str, class_names: list, width: int, height: int):
    """train_rgb_cnn.py's layout: returns (X, y, paths) with
    X.shape == (N, 3, height, width), dtype int32, values in [-128, 127]
    (see rgb_features.extract_rgb_pixels). Takes width/height directly
    rather than a Config, since capture size (what data/ is built at) and
    this model's own input size (after --downsample-factor) are two
    different things here -- unlike the DCT splits above, where Config
    alone determines both."""
    xs, ys, paths = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / split / class_name
        for jpg_path in sorted(class_dir.glob("*.jpg")):
            xs.append(extract_rgb_pixels(jpg_path, width, height))
            ys.append(class_idx)
            paths.append(str(jpg_path))
    X = np.stack(xs).astype(np.int32)
    y = np.array(ys, dtype=np.int64)
    assert X.shape == (len(ys), 3, height, width)
    return X, y, paths


def load_people_labels(meta_source_dir: Path, data_split: str) -> dict:
    """{filename: 0 or 1} sourced from meta_source_dir's original
    meta/{train,val}_intersections.json (see get_everyday_openimages_data.py
    and everyday_openimages160x120/README.md for what that file records).
    `meta_source_dir` is the *original* everyday_openimages{...} directory
    build_data.py merged/filtered from -- not data_dir itself, which has no
    memory of its own provenance.

    Split mapping is NOT 1:1 with data_dir's own split names: data/train
    and data/test both derive from meta_source_dir's train/ (data/test is
    carved out of it by dct_common.dataset.build_openimages_dataset, not a
    separately-collected split), so both use train_intersections.json;
    data/val is copied straight through from meta_source_dir's val/, so it
    uses val_intersections.json. `data_split` here means data_dir's split
    name ("train"/"val"/"test"), and this function resolves it to the
    right meta file internally.

    A filename absent from the returned dict has UNKNOWN ground truth
    (most commonly: it's a `garden` image -- Places365 has no object
    detections to derive a people/no-people label from) -- callers must
    treat a missing key as "exclude from the people-presence loss/metric",
    never assume 0 (that would silently teach "garden never has people",
    which was never actually measured)."""
    meta_filename = "train_intersections.json" if data_split in ("train", "test") else "val_intersections.json"
    with open(meta_source_dir / "meta" / meta_filename) as f:
        meta = json.load(f)
    labels = {}
    for class_name, entries in meta.items():
        for filename, other_classes in entries.items():
            # An image's own class is never listed in its *own*
            # intersections list (that list only records *other* target
            # classes present) -- so "this image's class IS people" has
            # to be special-cased here rather than falling out of the
            # "people" in other_classes check below.
            labels[filename] = 1 if (class_name == "people" or "people" in other_classes) else 0
    return labels


def build_spatial_split_with_people(data_dir: Path, split: str, class_names: list, cfg: Config, meta_source_dir: Path,
                                     augment: bool = False, augment_copies: int = 1):
    """Like build_spatial_split, plus a people_present array (int64, values
    0/1/-1) parallel to y and sourced from load_people_labels -- -1 means
    unknown (see that function's docstring), not "no people". Reuses
    build_spatial_split entirely (zero duplicated feature-extraction code)
    and just looks up each of its already-returned `paths` afterward.
    Augmented images (temp files, not in meta_source_dir's original
    listing) always resolve to -1/unknown here -- correct, not a gap:
    cropping/rotating an image can add or remove a person from frame, so
    the original's people-presence label can't be assumed to still hold."""
    X_y, X_c, y, paths = build_spatial_split(data_dir, split, class_names, cfg, augment=augment, augment_copies=augment_copies)
    people_labels = load_people_labels(meta_source_dir, split)
    p = np.array([people_labels.get(Path(path).name, -1) for path in paths], dtype=np.int64)
    return X_y, X_c, y, p, paths
