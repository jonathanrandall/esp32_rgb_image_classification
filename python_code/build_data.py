#!/usr/bin/env python3
"""Build ../data/{train,val,test}/<class>/*.jpg -- the single entry point
for producing training data with cross-class-intersection filtering,
class merging (combining several source classes into one new class, e.g.
chair+couch -> seat), AND dropping specific classes entirely (CLASS_DROP)
all applied together.

Reuses existing code rather than reimplementing it:
    - filter_intersections.py's `build_filtered_dataset`/
      `copy_filtered_class` do the actual per-image copy-or-drop
      decision (same meta/{train,val}_intersections.json-driven rules),
      generalized there to support a many-to-one class_map instead of
      only 1:1 filtering.
    - dct_common/dataset.py's `build_openimages_dataset` does the
      resize/re-encode-to-capture-size and stratified train/test split,
      generalized there to accept an explicit `class_names` list instead
      of always using CLASS_NAMES, so a merged taxonomy doesn't need its
      own copy of that logic either.

Two-step pipeline:
    1. Merge + filter `--source` into an intermediate
       `<source>_processed/` directory (same shape as `--source`, but
       CLASS_MERGE_MAP's groups combined into single class folders, and
       people/chair intersections dropped per filter_intersections.py's
       rules unless --no-filter).
    2. Build `../data/{train,val,test}/` from that intermediate
       directory at `--capture-width`/`--capture-height`/
       `--chroma-subsampling`, using the merged class list.

`../data/` is always rebuilt from scratch here (no dataset_matches_config
caching check like ensure_dataset() uses) -- this script is for
deliberately (re)building data/ with a specific taxonomy, not a fast-path
called at the top of every training run.

CLASS_NAMES is kept in sync automatically: after a successful build, the
resolved class list is written to dct_common/class_names.json (a small,
git-tracked sidecar file `dct_common/config.py`'s CLASS_NAMES loads from
if present -- see that file), so train_*.py's own ensure_dataset() calls
recognize the new taxonomy instead of seeing a mismatch and rebuilding
data/ from the unmerged --dataset-source. Pass --no-write-class-names for
a preview-only run that builds data/ but leaves class_names.json alone.

Customizing filtering/merging: EXTRA_EXCLUDE (in filter_intersections.py)
and CLASS_MERGE_MAP (below) are both edit-the-script config, not CLI
flags -- same convention as get_everyday_openimages_data.py's CLASS_MAP.
For example, to filter ONLY "people" out of every non-people class (no
other per-class exclusion) and merge "couch"+"chair" into a new class
called "couch_chair" (leaving "table" standalone):
    1. In filter_intersections.py, set `EXTRA_EXCLUDE = {}` (removes the
       table-drops-chair-intersections rule; DEFAULT_EXCLUDE = {"people"}
       already covers "people out of every non-people class" on its own).
    2. In this file, set:
           CLASS_MERGE_MAP = {
               "couch_chair": ["couch", "chair"],
           }
    3. Run: python build_data.py

Usage:
    python build_data.py                       # merge/filter + build data/ + update class_names.json
    python build_data.py --no-filter            # merge only, keep all images
    python build_data.py --no-write-class-names  # build data/ but don't touch class_names.json
    python build_data.py --capture-width 96 --capture-height 96 --chroma-subsampling 4:2:0
"""

import argparse
import json
import shutil
from pathlib import Path

from dct_common.config import Config, _CLASS_NAMES_FILE
from dct_common.dataset import build_openimages_dataset
from filter_intersections import build_filtered_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# dest_class_name -> [src_class_name, ...]. Every source class NOT
# mentioned here (and not in CLASS_DROP below) passes through unchanged
# (dest name == source name). Edit this to change what gets merged --
# not a CLI flag, same edit-the-script convention as
# filter_intersections.py's EXTRA_EXCLUDE and
# get_everyday_openimages_data.py's CLASS_MAP.
#
# 9-class taxonomy (2026-08-14b), superseding the same-day 10-class
# version above it in history: `crockery` (bowls+plates+cups) and the
# standalone `cutlery` class are now merged into one `eating_tools`
# class. Motivated directly by a concrete failure in that 10-class
# taxonomy's train_rgb_cnn.py run: `cutlery` got 0% test recall despite
# having as many train images as several classes that trained fine
# (850, same ballpark as car/doors/fruit/people) -- its confusion matrix
# showed 90/150 cutlery test images misclassified as `crockery`
# specifically, i.e. not a data-shortage problem, a same-context
# (kitchen tableware) visual-similarity problem the model couldn't
# separate. Merging the two removes that particular decision boundary
# instead of asking the model to keep drawing it.
CLASS_MERGE_MAP = {
    # 2026-08-22: `bookshelves` split out of `furniture` and `cutlery` out of
    # what was `eating_tools`, which is renamed `crockery` now that it holds
    # only dishes.
    #
    # Both merges had been grouping things that share a word but not an
    # appearance, which is what the model actually sees -- and at 32x24 input
    # it sees very little. A bookshelf and a couch have no common shape,
    # texture or context; forks and plates likewise. `furniture` was the
    # weakest non-`people` class in the 7-class model (precision 63.4%,
    # recall 69.2%) and the first to collapse under fine-tuning (recall 51%),
    # which is the signature of a label covering several visual concepts.
    #
    # `bookshelves` and `cutlery` are NOT dropped -- they pass through as
    # their own classes (build_data.py's default for anything neither merged
    # nor dropped), so they stay available and can be excluded at training
    # time with --classes rather than by another rebuild.
    "furniture": ["table", "chair", "couch"],
    "computer": ["laptop", "keyboard", "monitor"],
    "crockery": ["bowls", "plates", "cups"],
}

# Source classes dropped entirely -- not merged into anything, not
# passed through, absent from the final class list. `books`/`cupboard`
# joined the drop list in the 10-class taxonomy above (neither fit
# naturally into `furniture` or any other merged category); `microwave`/
# `dishwasher`/`bottles` remain dropped for the same data-shortfall/
# confusion-magnet reasons as before (see
# everyday_openimages160x120/README.md's shortfall table). `bookshelves`,
# `table`, `chair`, and `couch` are no longer in this set -- they're
# merged into `furniture` above instead of dropped.
CLASS_DROP = {"books", "cupboard", "dishwasher", "fridge", "microwave", "bottles"}


def resolve_class_map(source_dir: Path, merge_map: dict, drop: set) -> dict:
    """{dest_class_name: [src_class_name, ...]} for every class folder
    found under source_dir/train, excluding anything in `drop` -- merge_map
    entries win (a merge group must not reference a dropped class), every
    other surviving discovered class passes through 1:1 (dest name ==
    source name)."""
    all_source_classes = sorted(p.name for p in (source_dir / "train").iterdir() if p.is_dir())
    merged_sources = {s for srcs in merge_map.values() for s in srcs}
    dropped_but_merged = merged_sources & drop
    if dropped_but_merged:
        raise ValueError(f"CLASS_MERGE_MAP references dropped class(es) {sorted(dropped_but_merged)} -- "
                          f"remove them from CLASS_DROP or from the merge group, not both.")
    class_map = {dst: list(srcs) for dst, srcs in merge_map.items()}
    for name in all_source_classes:
        if name in drop:
            continue
        if name not in merged_sources:
            class_map[name] = [name]
    return class_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="everyday_openimages160x120",
                         help="source directory name, relative to the project root (default: everyday_openimages160x120)")
    parser.add_argument("--capture-width", type=int, default=160, help="multiple of 16 (default: 160)")
    parser.add_argument("--capture-height", type=int, default=120, help="multiple of 16 under 4:2:0, multiple of 8 under 4:2:2 (default: 120)")
    parser.add_argument("--chroma-subsampling", type=str, default="4:2:2", choices=["4:2:0", "4:2:2"],
                         help="default: 4:2:2, matches the real OV2640 camera")
    parser.add_argument("--no-filter", action="store_true",
                         help="skip meta/-based intersection filtering -- merge classes but keep every image")
    parser.add_argument("--no-yolo-filter", action="store_true",
                         help="skip the YOLO11m person-detection filter (meta/{split}_yolo_people.json, see "
                              "detect_people_yolo.py) -- keep only the Open-Images-intersections-based people filter")
    parser.add_argument("--no-curation-filter", action="store_true",
                         help="skip the hand-curation reject list (meta/{split}_curation_rejects.json, see "
                              "apply_curation_rejects.py) -- keep images marked bad in manual review")
    parser.add_argument("--no-write-class-names", action="store_true",
                         help="build data/ but don't write dct_common/class_names.json -- preview-only run, "
                              "CLASS_NAMES stays whatever it currently is")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                         help="fraction of each merged/filtered class's train images carved into data/test/ (default: 0.15)")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = PROJECT_ROOT / args.source
    class_map = resolve_class_map(source_dir, CLASS_MERGE_MAP, CLASS_DROP)
    class_names = sorted(class_map.keys())

    print("=" * 70)
    print("build_data.py")
    print("=" * 70)
    print(f"source: {source_dir}")
    print(f"capture: {args.capture_width}x{args.capture_height} ({args.chroma_subsampling})")
    print(f"filtering: {'off (--no-filter)' if args.no_filter else 'on (meta/-based, see filter_intersections.py)'}")
    print(f"YOLO people filter: {'off (--no-yolo-filter)' if args.no_yolo_filter else 'on (see detect_people_yolo.py)'}")
    print(f"hand-curation filter: {'off (--no-curation-filter)' if args.no_curation_filter else 'on (see apply_curation_rejects.py)'}")
    print(f"class merges: {CLASS_MERGE_MAP if CLASS_MERGE_MAP else '(none)'}")
    print(f"classes dropped: {sorted(CLASS_DROP) if CLASS_DROP else '(none)'}")
    print(f"final class list ({len(class_names)}): {class_names}")
    print()

    processed_dir = PROJECT_ROOT / f"{args.source}_processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    print(f"-- Step 1: merging + filtering {source_dir} -> {processed_dir} --")
    build_filtered_dataset(source_dir, processed_dir, class_map=class_map, apply_filter=not args.no_filter,
                            apply_yolo_filter=not args.no_yolo_filter,
                            apply_curation=not args.no_curation_filter)

    cfg = Config(
        capture_width=args.capture_width, capture_height=args.capture_height,
        chroma_subsampling=args.chroma_subsampling, seed=args.seed,
    )
    print()
    print(f"-- Step 2: building {DATA_DIR} from {processed_dir} --")
    build_openimages_dataset(DATA_DIR, cfg, test_fraction=args.test_fraction, source_dir=processed_dir, class_names=class_names)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"data/ written with class list: {class_names}")
    print(f"intermediate merged/filtered source kept at: {processed_dir}")
    print()

    if args.no_write_class_names:
        print("--no-write-class-names passed -- dct_common/class_names.json NOT touched.")
        print("CLASS_NAMES is still whatever it currently resolves to (see dct_common/config.py);")
        print(f"it will NOT match this data/ build's class list ({class_names}) until you either")
        print("re-run without --no-write-class-names, or edit class_names.json by hand.")
    else:
        with open(_CLASS_NAMES_FILE, "w") as f:
            json.dump(class_names, f, indent=2)
        print(f"CLASS_NAMES UPDATED -- wrote {_CLASS_NAMES_FILE}:")
        print(f"  {class_names}")
        print("dct_common/config.py now loads CLASS_NAMES from this file (falls back to its")
        print("hardcoded default only if the file is missing) -- every script that imports")
        print("dct_common.config from here on, in a fresh process, will see this list. Review")
        print(f"the diff on {_CLASS_NAMES_FILE.relative_to(PROJECT_ROOT)} like any other code change.")


if __name__ == "__main__":
    main()
