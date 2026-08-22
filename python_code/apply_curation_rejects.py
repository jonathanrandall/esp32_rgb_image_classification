#!/usr/bin/env python3
"""Fold make_gallery.py's exported reject lists into the source dataset's
curation metadata, so hand review survives a rebuild.

THE PROBLEM THIS SOLVES. `build_data.py` wipes and rebuilds both
`<source>_processed/` and `data/` from scratch on every run
(shutil.rmtree, build_data.py step 1). Curation applied by deleting or
moving files in either of those directories is therefore destroyed the
next time the taxonomy, a filter threshold, or the capture size changes
-- which in this project has been roughly weekly. The only durable place
to record "this image is bad" is beside the source, as metadata.

So: review wherever is convenient, export the reject list, and run this.
The decision lands in
    <source>/meta/{split}_curation_rejects.json
        {class_name: [filename, ...]}
which `filter_intersections.copy_filtered_class()` consults as a third
independent drop rule alongside the Open-Images-intersections check and
the YOLO person check. Nothing is deleted, ever -- the source images stay
put, and removing a name from the JSON un-rejects it.

REVIEW WHEREVER YOU LIKE. Reject lists are just filenames, and this
script resolves each one back to its SOURCE class by indexing the source
tree, so it does not matter which directory you actually looked at:

  - Reviewing `<source>_processed/train/furniture/` (post-merge) is
    fine -- each filename resolves back to whichever of chair/couch/
    table/cupboard/bookshelves it really came from, and the reject is
    recorded there. That survives a later change to the merge map.
  - Reviewing `data/train/furniture/` is fine too, even though those
    JPEGs have been re-encoded to 4:2:2 -- the filenames are unchanged.
  - Reviewing the source directly is fine and is the only way to see
    classes that the people filter would otherwise have hidden from you.

Reviewing a merged or filtered directory is usually the better use of
time: it only shows images that actually reach `data/`, so you are not
grading ~7,000 images the people filter is going to discard anyway.

Usage:
    # one exported list
    python apply_curation_rejects.py ~/Downloads/rejected_..._train_furniture.txt

    # everything exported so far, in one go
    python apply_curation_rejects.py ~/Downloads/rejected_*.txt

    # see what would change without writing
    python apply_curation_rejects.py --dry-run ~/Downloads/rejected_*.txt

    # change your mind: un-reject the listed files
    python apply_curation_rejects.py --remove ~/Downloads/rejected_....txt

Then rebuild:
    python build_data.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val")


def index_source(source_dir: Path) -> dict:
    """{split: {filename: class_name}}, indexed PER SPLIT.

    Filenames are unique within a split but NOT across splits:
    get_everyday_openimages_data.py runs train and val through
    build_detection_split() with separate `claimed` sets and a counter
    that restarts, so `books_000013.jpg` exists in both train/books and
    val/books. Roughly 1,550 names collide this way in the current pull.
    A flat filename->(split, class) index would therefore silently
    misfile every colliding train reject as a val reject, so the split
    has to come from the reviewed directory instead -- see
    split_from_list_name()."""
    index = {}
    for split in SPLITS:
        split_dir = source_dir / split
        if not split_dir.is_dir():
            continue
        per_split = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for jpg in class_dir.glob("*.jpg"):
                per_split[jpg.name] = class_dir.name
        index[split] = per_split
    return index


def split_from_list_name(list_path: Path) -> str | None:
    """Recover which split a reject list was reviewed from, out of
    make_gallery.py's exported name.

    That name is `rejected_<key>.txt` where <key> is the reviewed
    directory's last three path components joined by "_", e.g.
    `rejected_data_train_furniture.txt` or
    `rejected_everyday_openimages160x120_processed_val_car.txt`. Matching
    on the bracketed split component is safe even though class names
    contain underscores (`eating_tools`), because the split always sits
    between two underscores and class names never contain "train"/"val"/
    "test" as a whole component.

    data/test/ maps back to the source TRAIN split: build_openimages_dataset()
    carves data/test/ out of source train/ by a stratified shuffle, so a
    reject found there is a train-split image. Getting this backwards
    would write the name into the val list, where it would never match
    anything and would silently fail to filter."""
    stem = list_path.stem
    for token, split in (("_test_", "train"), ("_train_", "train"), ("_val_", "val")):
        if token in stem:
            return split
    return None


def read_reject_list(path: Path) -> list:
    """One filename per line, blanks and #-comments ignored. Accepts bare
    names or full paths (make_gallery.py exports bare names; a
    hand-written list might not)."""
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(Path(line).name)
    return names


def load_existing(source_dir: Path, split: str) -> dict:
    path = source_dir / "meta" / f"{split}_curation_rejects.json"
    if not path.exists():
        return {}
    return {cls: list(names) for cls, names in json.load(open(path)).items()}


def write_rejects(source_dir: Path, split: str, rejects: dict) -> Path:
    """Sorted names within sorted classes, empty classes pruned -- so the
    file is stable across runs and diffs cleanly in git."""
    meta_dir = source_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{split}_curation_rejects.json"
    cleaned = {cls: sorted(set(names)) for cls, names in sorted(rejects.items()) if names}
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lists", nargs="+", type=str,
                        help="exported rejected_*.txt file(s), or a directory containing them")
    parser.add_argument("--source", default="everyday_openimages160x120",
                        help="source directory name, relative to the project root (default: everyday_openimages160x120)")
    parser.add_argument("--split", choices=SPLITS, default=None,
                        help="force which source split these lists were reviewed from, instead of "
                             "recovering it from the rejected_*.txt filename. Note data/test/ belongs "
                             "to the source TRAIN split.")
    parser.add_argument("--remove", action="store_true",
                        help="un-reject the listed files instead of rejecting them")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / args.source
    if not source_dir.is_dir():
        raise SystemExit(f"source directory not found: {source_dir}")

    list_paths = []
    for raw in args.lists:
        p = Path(raw).expanduser()
        if p.is_dir():
            list_paths.extend(sorted(p.glob("rejected_*.txt")))
        elif p.is_file():
            list_paths.append(p)
        else:
            print(f"  not found, skipping: {p}", file=sys.stderr)
    if not list_paths:
        raise SystemExit("no reject lists found")

    print(f"source: {source_dir}")
    print("indexing source tree...")
    index = index_source(source_dir)
    for split in SPLITS:
        print(f"  {split}: {len(index.get(split, {})):,} images")
    print()

    # split -> class -> set(names)
    incoming = {split: defaultdict(set) for split in SPLITS}
    unresolved = []
    total_listed = 0
    bad_split = False

    for lp in list_paths:
        split = args.split or split_from_list_name(lp)
        if split is None:
            print(f"  {lp.name}: CANNOT TELL WHICH SPLIT -- skipped", file=sys.stderr)
            print("      The filename carries no _train_/_val_/_test_ component. Either keep "
                  "make_gallery.py's exported name, or pass --split.", file=sys.stderr)
            bad_split = True
            continue

        names = read_reject_list(lp)
        total_listed += len(names)
        per_split = index.get(split, {})
        hits = 0
        for name in names:
            class_name = per_split.get(name)
            if class_name is None:
                unresolved.append((lp.name, split, name))
                continue
            incoming[split][class_name].add(name)
            hits += 1
        origin = "--split" if args.split else "filename"
        print(f"  {lp.name}: split={split} (from {origin}), {len(names)} listed, {hits} resolved")

    if bad_split:
        raise SystemExit("aborting: at least one list had an undeterminable split, and guessing it "
                         "wrong would file rejects where they never match anything.")

    if unresolved:
        print()
        print(f"WARNING: {len(unresolved)} filename(s) were not found in the split they were attributed to.")
        print("         These are ignored. Usual causes: the list came from a different --source or a")
        print("         different resolution (96x96 vs 160x120), or the split was misattributed.")
        for lname, split, name in unresolved[:5]:
            print(f"         {lname}: {name} (looked in {split})")
        if len(unresolved) > 5:
            print(f"         ... and {len(unresolved) - 5} more")

    print()
    verb = "un-rejecting" if args.remove else "rejecting"
    wrote_any = False
    for split in SPLITS:
        if not incoming[split]:
            continue
        existing = load_existing(source_dir, split)
        before = sum(len(v) for v in existing.values())
        merged = {cls: set(names) for cls, names in existing.items()}
        for cls, names in incoming[split].items():
            cur = merged.setdefault(cls, set())
            if args.remove:
                merged[cls] = cur - names
            else:
                merged[cls] = cur | names
        after = sum(len(v) for v in merged.values())

        print(f"[{split}] {verb} -- {before} rejected before, {after} after ({after - before:+d})")
        for cls in sorted(incoming[split]):
            n_new = len(incoming[split][cls])
            print(f"    {cls:<16} {n_new:>5} name(s) in list, {len(merged.get(cls, ())):>5} rejected total")

        if args.dry_run:
            continue
        path = write_rejects(source_dir, split, merged)
        print(f"    wrote {path}")
        wrote_any = True

    print()
    if args.dry_run:
        print("--dry-run: nothing written.")
    elif wrote_any:
        print(f"Done. {total_listed} name(s) processed. Now rebuild:")
        print("    python build_data.py")
        print()
        print("The rebuild prints per-class drop reasons, so you can confirm the")
        print("'curation N' counts match what you just recorded.")
    else:
        print("Nothing matched -- no files written.")


if __name__ == "__main__":
    main()
