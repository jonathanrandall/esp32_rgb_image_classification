#!/usr/bin/env python3

"""
Build a broad "everyday objects" dataset from Open Images V7 (plus a
Places365 pass for `garden`, same source as get_garden_data.py), at TWO
resolutions in one run: 160x120 and 96x96. Unlike get_open_image_data.py
(which builds this project's actual 5 training classes), this script is
a wider, shallower pull across ~22 household/office/kitchen/furniture
classes -- meant as a supplementary "everyday objects" dataset, not a
replacement for the curated 5-class training set.

Crop/pad pipeline (padded-crop -> expand-to-aspect -> letterbox resize)
is copied verbatim from get_open_image_data.py -- see that file's
docstrings for why (never a plain aspect-distorting .resize()).

Cross-class intersection handling:
    1. Hard: an image, once used for one class, is never reused for a
       different class (global `claimed` id set), so no two class
       folders ever contain the literal same photo.
    2. Soft preference, not exclusion: within one class's candidate
       pool, "clean" images (no other target class's label among that
       image's OTHER annotated objects) are filled in before "dirty"
       ones, but dirty candidates still get used to reach the
       1000/100 target if clean alone isn't enough. Hard-excluding
       intersections outright would starve genuinely rare classes
       (dishwasher, cupboard, ...) and, empirically, `people`
       specifically: a spot-check found people present in 40-53% of
       ALL candidates for desk/furniture classes like monitor/chair,
       not just a marginal fraction -- excluding people hard would
       cost a lot of yield there.
    3. Every saved image's *actual* other-target-classes-present is
       recorded (not just a clean/dirty bit) into
       meta/{train,val}_intersections.json: {class: {filename: [other
       classes present]}}, empty list = clean. This makes any
       filtering choice (drop people-intersections only, drop all
       intersections, or use everything) a training-time decision
       instead of one baked into the download -- avoids having to
       redownload if the desired filtering strategy changes later.
    `garden` (Places365, whole-scene labels, no bounding boxes) isn't
    cross-checked against the Open-Images classes and has no entry in
    the intersection metadata -- different source, no shared object
    annotations to compare against, same limitation get_garden_data.py
    already documents for its own pipeline.

`people` and `car` are pulled fresh here too (not copied from
data_curation/open_images_tmp/) so that both resolutions and all ~22
classes go through identical filtering/intersection logic and land on
the same 1000/100 per-class target. This re-hits Open Images V7's
"Person"/"Car" labels, but FiftyOne's local cache
(~/fiftyone/open-images-v7/) already has the raw images from the
original 5-class pull, so this is expected to mostly avoid new network
downloads, not repeat them.

Output:
    everyday_openimages160x120/{train,val}/<class>/
    everyday_openimages96x96/{train,val}/<class>/
    (both one level above python_code/)

Install:
    python3 -m pip install fiftyone pillow tqdm

Run:
    python3 get_everyday_openimages_data.py
"""

import json
from pathlib import Path
import random

from PIL import Image
from tqdm import tqdm

import fiftyone as fo
import fiftyone.zoo as foz


# ============================================================
# USER SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (output_width, output_height, output_dir_name)
RESOLUTIONS = [
    (160, 120, "everyday_openimages160x120"),
    (96, 96, "everyday_openimages96x96"),
]

TRAIN_PER_CLASS = 1000
VAL_PER_CLASS = 100

# See get_open_image_data.py for why 4.0 (download more than needed,
# top up per-class afterwards if any class still falls short).
DOWNLOAD_MULTIPLIER = 4.0

MIN_OBJECT_AREA = 0.05
MAX_OBJECT_AREA = 0.90
CROP_PADDING = 0.15
JPEG_QUALITY = 95
RANDOM_SEED = 42

VAL_FRACTION = 0.10  # garden only (Places365 has no separate val split pull here)


# ============================================================
# CLASS DEFINITIONS (Open Images V7 detection labels)
# ============================================================
# Order matters for intersection avoidance: rarer/harder classes are
# listed first so they get first pick of "clean" (non-intersecting)
# candidate images before the big common classes (people, car) run.

CLASS_MAP = {
    "dishwasher": ["Dishwasher"],
    "cupboard": ["Cupboard", "Chest of drawers"],
    "bookshelves": ["Bookcase"],
    "monitor": ["Computer monitor"],
    "mouse": ["Computer mouse"],
    "keyboard": ["Computer keyboard"],
    "fridge": ["Refrigerator"],
    "microwave": ["Microwave oven"],
    "laptop": ["Laptop"],
    "cutlery": ["Knife", "Kitchen knife", "Fork", "Spoon"],
    "plates": ["Plate"],
    "bowls": ["Bowl"],
    "cups": ["Coffee cup", "Mug"],
    "doors": ["Door"],
    "couch": ["Couch", "Sofa bed", "Studio couch", "Loveseat"],
    "table": ["Table", "Coffee table", "Kitchen & dining room table"],
    "chair": ["Chair"],
    "bottles": ["Bottle"],
    "books": ["Book"],
    "fruit": [
        "Fruit", "Apple", "Banana", "Orange", "Grape", "Strawberry",
        "Watermelon", "Pineapple", "Pear", "Lemon", "Mango", "Peach",
        "Common fig", "Grapefruit", "Pomegranate", "Cantaloupe", "Coconut",
    ],
    "people": ["Person", "Man", "Woman", "Boy", "Girl"],
    "car": ["Car"],
}

# Precomputed once: for each class, the set of OI labels belonging to
# every *other* target class -- used for the soft intersection check.
OTHER_CLASS_LABELS = {
    name: set().union(*(set(lbls) for other, lbls in CLASS_MAP.items() if other != name))
    for name in CLASS_MAP
}

# Reverse lookup: OI label -> our class name, used to translate a
# saved image's "other classes present" from labels into class names
# for the intersection metadata (meta/<split>.json).
LABEL_TO_CLASS = {label: name for name, labels in CLASS_MAP.items() for label in labels}

# Places365 categories genuinely about gardens (confirmed directly
# against the loaded Places validation split's actual label strings --
# not guessed -- excludes near-matches like beer_garden/junkyard/
# kindergarden_classroom/vineyard/courtyard). Same source and pipeline
# as get_garden_data.py, extended with 4 categories
# (greenhouse/indoor+outdoor, lawn, yard) that script's current
# CLASS_MAP doesn't have but data_curation/open_images_tmp/garden was
# actually already built with -- matched here for consistency.
GARDEN_LABELS = {
    "/b/botanical_garden": "botanical_garden",
    "/f/formal_garden": "formal_garden",
    "/g/greenhouse/indoor": "greenhouse_indoor",
    "/g/greenhouse/outdoor": "greenhouse_outdoor",
    "/j/japanese_garden": "japanese_garden",
    "/l/lawn": "lawn",
    "/r/roof_garden": "roof_garden",
    "/t/topiary_garden": "topiary_garden",
    "/v/vegetable_garden": "vegetable_garden",
    "/y/yard": "yard",
    "/z/zen_garden": "zen_garden",
}


# ============================================================
# CROP / RESIZE HELPERS (verbatim logic from get_open_image_data.py,
# parameterized by width/height instead of module-level constants)
# ============================================================

def bbox_area(bbox):
    _, _, w, h = bbox
    return w * h


def padded_crop_box(bbox, image_width, image_height):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    pad_x, pad_y = w * CROP_PADDING, h * CROP_PADDING
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return (
        int(x1 * image_width), int(y1 * image_height),
        int(x2 * image_width), int(y2 * image_height),
    )


def expand_crop_to_aspect(crop_box, image_width, image_height, target_width, target_height):
    """See get_open_image_data.py's docstring for the full explanation --
    grows the crop box (pulling in real neighboring content), shifts
    into bounds, then trims only as a last resort."""
    left, top, right, bottom = (float(v) for v in crop_box)
    target_ratio = target_width / target_height

    def resize_to_ratio(left, top, right, bottom):
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0:
            return left, top, right, bottom
        current_ratio = w / h
        if current_ratio < target_ratio:
            grow = (h * target_ratio - w) / 2
            return left - grow, top, right + grow, bottom
        elif current_ratio > target_ratio:
            grow = (w / target_ratio - h) / 2
            return left, top - grow, right, bottom + grow
        return left, top, right, bottom

    left, top, right, bottom = resize_to_ratio(left, top, right, bottom)

    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > image_width:
        left -= (right - image_width)
        right = float(image_width)
    if bottom > image_height:
        top -= (bottom - image_height)
        bottom = float(image_height)

    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(image_width), right), min(float(image_height), bottom)
    w, h = right - left, bottom - top
    if w > 0 and h > 0:
        current_ratio = w / h
        if current_ratio < target_ratio:
            trim = (h - w / target_ratio) / 2
            top += trim
            bottom -= trim
        elif current_ratio > target_ratio:
            trim = (w - h * target_ratio) / 2
            left += trim
            right -= trim

    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def contain_resize(img, target_width, target_height, fill_color=(0, 0, 0)):
    src_w, src_h = img.size
    scale = min(target_width / src_w, target_height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), fill_color)
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def save_crop(image_path, detection, output_path, width, height):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            crop_box = padded_crop_box(detection.bounding_box, w, h)
            crop_box = expand_crop_to_aspect(crop_box, w, h, width, height)
            cropped = img.crop(crop_box)
            if cropped.width < 10 or cropped.height < 10:
                return False
            cropped = contain_resize(cropped, width, height)
            cropped.save(output_path, format="JPEG", quality=JPEG_QUALITY)
        return True
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return False


# ============================================================
# OPEN IMAGES DETECTION CLASSES
# ============================================================

def build_detection_split(output_dir, width, height, oi_split, output_split, samples_per_class, claimed, meta):
    download_count = int(samples_per_class * DOWNLOAD_MULTIPLIER)

    for our_class, oi_classes in CLASS_MAP.items():
        print()
        print("=" * 70)
        print(f"{output_split}: {our_class}  ({width}x{height})")
        print(f"Open Images labels: {oi_classes}")
        print("=" * 70)

        dataset_name = f"everyday_{output_split}_{our_class}"
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=oi_split,
            label_types=["detections"],
            classes=oi_classes,
            max_samples=download_count,
            shuffle=True,
            seed=RANDOM_SEED,
            dataset_name=dataset_name,
        )

        other_labels = OTHER_CLASS_LABELS[our_class]
        clean, dirty = [], []
        for sample in dataset:
            if sample.id in claimed:
                continue
            detections = sample.ground_truth.detections
            valid = [d for d in detections if d.label in oi_classes
                     and MIN_OBJECT_AREA <= bbox_area(d.bounding_box) <= MAX_OBJECT_AREA]
            if not valid:
                continue
            # Other target classes actually present in this image's
            # detections (not just this class's own labels) -- recorded
            # per saved image below so filtering by intersection can be
            # a training-time choice instead of baked in at download
            # time (e.g. drop only people-intersections, or all
            # intersections, or none).
            other_classes_present = sorted({
                LABEL_TO_CLASS[d.label] for d in detections
                if d.label in other_labels
            })
            (dirty if other_classes_present else clean).append((sample, valid, other_classes_present))

        rng = random.Random(RANDOM_SEED)
        rng.shuffle(clean)
        rng.shuffle(dirty)
        ordered = clean + dirty  # prefer non-intersecting candidates first

        output_folder = output_dir / output_split / our_class
        class_meta = meta.setdefault(our_class, {})
        count = 0
        for sample, valid, other_classes_present in tqdm(ordered, desc=f"Creating {our_class}"):
            if count >= samples_per_class:
                break
            if sample.id in claimed:
                continue
            detection = max(valid, key=lambda d: bbox_area(d.bounding_box))
            filename = f"{our_class}_{count:06d}.jpg"
            success = save_crop(sample.filepath, detection, output_folder / filename, width, height)
            if success:
                class_meta[filename] = other_classes_present
                count += 1
                claimed.add(sample.id)

        print(f"Created {count} / {samples_per_class} {our_class} images "
              f"({len(dirty)} of {len(clean) + len(dirty)} candidates were cross-class intersections)")

        fo.delete_dataset(dataset_name)


# ============================================================
# GARDEN (Places365, whole-scene, no bounding boxes)
# ============================================================

def build_garden_split(output_dir, width, height):
    print()
    print("=" * 70)
    print(f"garden  ({width}x{height}, Places365)")
    print("=" * 70)

    dataset = foz.load_zoo_dataset(
        "places", split="validation", dataset_name="everyday_places_garden_source",
    )

    by_category = {name: [] for name in GARDEN_LABELS.values()}
    for sample in dataset:
        name = GARDEN_LABELS.get(sample.ground_truth.label)
        if name is not None:
            by_category[name].append(sample.filepath)

    train_count = val_count = 0
    for category, filepaths in by_category.items():
        filepaths = sorted(filepaths)
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(filepaths)
        n_val = max(1, round(len(filepaths) * VAL_FRACTION))
        val_paths, train_paths = filepaths[:n_val], filepaths[n_val:]

        for split, paths in (("train", train_paths), ("val", val_paths)):
            for i, path in enumerate(tqdm(paths, desc=f"{split}/garden/{category}")):
                try:
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        resized = contain_resize(img, width, height)
                        out_path = output_dir / split / "garden" / f"garden_{category}_{i:04d}.jpg"
                        resized.save(out_path, format="JPEG", quality=JPEG_QUALITY)
                    if split == "train":
                        train_count += 1
                    else:
                        val_count += 1
                except Exception as e:
                    print(f"  skipping unreadable image {path}: {e!r}")

    fo.delete_dataset("everyday_places_garden_source")
    print(f"Created {train_count} train / {val_count} val garden images (Places365-capped, not a shortfall)")


# ============================================================
# MAIN
# ============================================================

def prepare_directories(output_dir):
    class_names = list(CLASS_MAP.keys()) + ["garden"]
    for split in ["train", "val"]:
        for cls in class_names:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def main():
    random.seed(RANDOM_SEED)

    print()
    print("Everyday Open Images dataset builder")
    print(f"Classes: {', '.join(list(CLASS_MAP.keys()) + ['garden'])}")
    print(f"Train/class: {TRAIN_PER_CLASS}   Val/class: {VAL_PER_CLASS}")
    print(f"Resolutions: {[(w, h) for w, h, _ in RESOLUTIONS]}")
    print()

    for width, height, dirname in RESOLUTIONS:
        output_dir = PROJECT_ROOT / dirname
        print()
        print("#" * 70)
        print(f"# RESOLUTION {width}x{height} -> {output_dir}")
        print("#" * 70)

        prepare_directories(output_dir)

        claimed_train = set()
        claimed_val = set()
        meta_train = {}
        meta_val = {}

        build_detection_split(output_dir, width, height, "train", "train", TRAIN_PER_CLASS, claimed_train, meta_train)
        build_detection_split(output_dir, width, height, "validation", "val", VAL_PER_CLASS, claimed_val, meta_val)
        build_garden_split(output_dir, width, height)

        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "train_intersections.json", "w") as f:
            json.dump(meta_train, f, indent=2, sort_keys=True)
        with open(meta_dir / "val_intersections.json", "w") as f:
            json.dump(meta_val, f, indent=2, sort_keys=True)
        print(f"Wrote intersection metadata to {meta_dir}")
        print("(garden has no metadata -- Places365 whole-scene labels, no object")
        print(" detections to cross-check against; see build_garden_split's docstring)")

        print()
        print(f"DONE with {width}x{height} -> {output_dir.resolve()}")

    print()
    print("=" * 70)
    print("ALL RESOLUTIONS DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
