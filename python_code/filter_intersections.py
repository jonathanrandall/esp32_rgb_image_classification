#!/usr/bin/env python3
"""Build a filtered copy of an everyday_openimages<res>/ dataset that drops
images flagged by meta/{train,val}_intersections.json (see
get_everyday_openimages_data.py and everyday_openimages160x120/README.md
for what that file records: per saved image, which *other* target classes
were also annotated in its source photo -- recorded but never excluded at
download time, so filtering could be a training-time decision) AND, since
2026-08-14c, meta/{train,val}_yolo_people.json (see detect_people_yolo.py
-- a real YOLO11m detection pass over the actual saved images, catching
people Open Images' own annotators never boxed).

Motivated directly by the first CNN run on the unfiltered 23-class
dataset: `table` was confused with `chair` more than any other pair
(22/150 test images), and `people` was the single biggest driver of
cross-class intersections generally (see everyday_openimages160x120/
README.md's spot-check numbers, ~40-53% of candidates for desk/furniture
classes). This is a real experiment testing whether removing those
specific intersections improves the confusion, not a blanket
"maximally clean data" pass -- only what's actually described below is
excluded.

The intersections-based people filter alone turned out to be incomplete
in practice -- a visible fraction of `furniture` images still had a
person in frame after that filter, because Open Images' own detection
annotations don't box every person in every photo (see
detect_people_yolo.py's module docstring for the full explanation). The
YOLO-based check below closes that gap with an actual object detector
run on the real saved pixels, independent of what Open Images happened
to annotate.

Rules:
    - Every class except `people` drops any image whose intersections
      list includes "people" (Open-Images-annotation-based, as before).
    - Every class except `people` ALSO drops any image where
      YOLO_PERSON_CONF_THRESHOLD or higher person-detection confidence
      was recorded in meta/{split}_yolo_people.json (YOLO-based, new).
      These two checks are independent ORs -- either one alone is enough
      to drop an image.
    - `garden` has no intersections meta entry (Places365 whole-scene
      labels, no object detections) but DOES get the YOLO check (it runs
      directly on pixels, no Open-Images-annotation dependency).
      `people` itself is structurally never flagged for "people" by the
      intersections check (an image's intersections list only records
      *other* target classes, never its own) and is explicitly skipped
      by the YOLO check too -- both by design, not incidentally: a
      person detected in a `people`-class photo is exactly the point of
      that class, not something to filter out.

Output: <source>_filtered/{train,val}/<class>/*.jpg -- same filenames,
a subset of the source. Point --dataset-source at this directory name
in train_cnn.py/train_ac_cnn.py/train_mlp.py to train on it. Note:
dct_common.dataset.ensure_dataset() only checks that data/'s class set,
image size, and chroma layout match the current Config -- it does NOT
compare against the source directory's content, so switching
--dataset-source to this filtered copy will NOT trigger a data/ rebuild
by itself if a data/ built from the unfiltered source already exists at
the same resolution. Delete or rename data/ before training on this
output if that's the case.

This module's `build_filtered_dataset`/`copy_filtered_class` are also
reused directly (not duplicated) by `build_data.py`, which additionally
merges several source classes into one destination class (e.g.
chair+couch+table -> furniture) on top of this same filtering.

Usage:
    python filter_intersections.py [--source everyday_openimages160x120]
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every class drops images intersecting with these labels...
DEFAULT_EXCLUDE = {"people"}
# ...plus these additional per-class exclusions. Empty as of the
# 10-class taxonomy in build_data.py's CLASS_MERGE_MAP (2026-08-14):
# the old "table" drops "chair"-intersecting images" rule is moot now
# that table/chair/couch/bookshelves are merged into one "furniture"
# class -- dropping table-chair intersections asymmetrically no longer
# makes sense when both source classes land in the same destination.
EXTRA_EXCLUDE = {}

# Minimum YOLO11m person-detection confidence (meta/{split}_yolo_people.json,
# see detect_people_yolo.py) for an image to be dropped as "has a person" --
# below Ultralytics' own default detection confidence (0.25) is never
# recorded at all (no box survives that low), so this threshold only
# matters within the confidence range YOLO actually returns. 0.25 (no
# extra tightening beyond YOLO's own default) errs toward dropping more
# borderline detections rather than fewer, on the theory that a
# false-positive "person" drop just costs one training image, while a
# false-negative leaves an actual, unfiltered person in the class it's
# not supposed to be in -- the exact problem this whole mechanism exists
# to fix. Tune here (not by re-running detect_people_yolo.py) since the
# meta file stores the raw confidence, not a baked-in yes/no.
YOLO_PERSON_CONF_THRESHOLD = 0.25


def copy_filtered_class(
    source_dir: Path, split: str, src_class_name: str, dst_class_dir: Path,
    meta: dict, apply_filter: bool = True,
    yolo_meta: dict | None = None, apply_yolo_filter: bool = True,
) -> tuple:
    """Copy one source class's `split` *.jpg files into dst_class_dir
    (created if needed), dropping any whose meta-recorded intersections
    hit DEFAULT_EXCLUDE/EXTRA_EXCLUDE for `src_class_name`, OR whose
    yolo_meta-recorded person-detection confidence is at least
    YOLO_PERSON_CONF_THRESHOLD (skipped entirely for src_class_name ==
    "people" -- see module docstring). Returns (kept, dropped). This is
    the one place the actual copy/filter decision lives --
    build_filtered_dataset (below, 1:1 filtering) and build_data.py
    (many-to-one class merging + filtering) both call it, rather than
    each having their own copy of this loop."""
    exclude_labels = (DEFAULT_EXCLUDE | EXTRA_EXCLUDE.get(src_class_name, set())) if apply_filter else set()
    class_meta = meta.get(src_class_name, {})  # {} for garden -- no meta, nothing filtered
    class_yolo_meta = (yolo_meta or {}).get(src_class_name, {}) if src_class_name != "people" else {}
    check_yolo = apply_yolo_filter and src_class_name != "people"
    src_class_dir = source_dir / split / src_class_name
    dst_class_dir.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    for jpg_path in sorted(src_class_dir.glob("*.jpg")):
        other_classes = set(class_meta.get(jpg_path.name, []))
        if other_classes & exclude_labels:
            dropped += 1
            continue
        if check_yolo and class_yolo_meta.get(jpg_path.name, 0.0) >= YOLO_PERSON_CONF_THRESHOLD:
            dropped += 1
            continue
        shutil.copy2(jpg_path, dst_class_dir / jpg_path.name)
        kept += 1
    return kept, dropped


def build_filtered_dataset(source_dir: Path, dest_dir: Path, class_map: dict | None = None,
                            apply_filter: bool = True, apply_yolo_filter: bool = True) -> None:
    """class_map: {dest_class_name: [src_class_name, ...]} -- defaults to
    an identity mapping over every class folder found under
    source_dir/train (one source class per dest class, same name), this
    script's own standalone behavior. build_data.py passes a real
    many-to-one map to merge several source classes into one dest class
    (e.g. {"furniture": ["chair", "couch", "table"]}) on top of the same
    filtering, reusing copy_filtered_class per source class rather than
    duplicating it. apply_yolo_filter requires
    meta/{split}_yolo_people.json to exist (run detect_people_yolo.py
    first) -- missing/empty is treated as "no YOLO detections recorded",
    same as the intersections meta's own missing-file fallback, not an
    error."""
    if class_map is None:
        class_map = {p.name: [p.name] for p in (source_dir / "train").iterdir() if p.is_dir()}

    for split in ("train", "val"):
        meta_path = source_dir / "meta" / f"{split}_intersections.json"
        meta = json.load(open(meta_path)) if meta_path.exists() else {}
        yolo_meta_path = source_dir / "meta" / f"{split}_yolo_people.json"
        yolo_meta = json.load(open(yolo_meta_path)) if yolo_meta_path.exists() else {}
        if apply_yolo_filter and not yolo_meta_path.exists():
            print(f"  NOTE: {yolo_meta_path} not found -- run detect_people_yolo.py first for the YOLO-based "
                  f"people filter to have any effect on {split}; continuing with intersections-only filtering.")

        for dst_class_name, src_class_names in sorted(class_map.items()):
            dst_class_dir = dest_dir / split / dst_class_name
            total_kept = total_dropped = 0
            for src_class_name in src_class_names:
                kept, dropped = copy_filtered_class(source_dir, split, src_class_name, dst_class_dir, meta, apply_filter,
                                                      yolo_meta, apply_yolo_filter)
                total_kept += kept
                total_dropped += dropped
            label = dst_class_name if src_class_names == [dst_class_name] else f"{dst_class_name} <- {'+'.join(src_class_names)}"
            print(f"  {split}/{label}: kept {total_kept}, dropped {total_dropped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="everyday_openimages160x120",
                         help="source directory name, relative to the project root (default: everyday_openimages160x120)")
    parser.add_argument("--dest", default=None,
                         help="destination directory name, relative to the project root (default: <source>_filtered)")
    parser.add_argument("--no-yolo-filter", action="store_true",
                         help="skip the YOLO11m person-detection filter, keep only the Open-Images-intersections one")
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / args.source
    dest_dir = PROJECT_ROOT / (args.dest or f"{args.source}_filtered")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    print(f"Filtering {source_dir} -> {dest_dir}")
    build_filtered_dataset(source_dir, dest_dir, apply_yolo_filter=not args.no_yolo_filter)
    print("done")


if __name__ == "__main__":
    main()
