#!/usr/bin/env python3
"""Resolve a reviewed batch: accepted images into data_hand_curated/,
rejected ones into data_rejected/, and the rejection recorded durably.

Step 3 of the loop begun by curation_pull.py. For every
curation_review/<split>/<class>/ folder, looks for a reject list in two
places:

  1. in the folder itself (any rejected_*.txt -- save the browser
     download there), or
  2. ~/Downloads/rejected_curation_review_<split>_<class>.txt, which is
     exactly what make_gallery.py names it, so a fresh download works
     with no file shuffling.

A folder with no list found in either place is left completely alone and
reported as unreviewed -- NOT treated as "everything accepted". Silently
accepting an unreviewed batch would quietly defeat the whole exercise.

Nothing is deleted. Rejects move to data_rejected/, so a mis-click is
recoverable by moving the file back.

TWO RECORDS ARE WRITTEN, deliberately:

  - data_hand_curated/ gets the accepted images. This is the fine-tuning
    set, and it is a working directory -- disposable, rebuildable.
  - <source>/meta/{split}_curation_rejects.json gets the rejected
    FILENAMES, resolved back to their original source class. This is the
    durable record. It survives `build_data.py` wiping and rebuilding
    data/, and it is small enough to commit.

The second is what makes the effort permanent. Without it, a rebuild of
data/ silently reintroduces every image you just rejected. See
filter_intersections.load_curation_rejects().

Usage:
    python curation_resolve.py
    python curation_resolve.py --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")
DOWNLOADS_DIR = Path.home() / "Downloads"


def list_images(d: Path) -> list:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def find_reject_list(review_class_dir: Path, split: str, cls: str) -> Path | None:
    local = sorted(review_class_dir.glob("rejected_*.txt"))
    if local:
        return local[0]
    # make_gallery.py keys a directory by its last three path components.
    downloaded = DOWNLOADS_DIR / f"rejected_curation_review_{split}_{cls}.txt"
    return downloaded if downloaded.is_file() else None


def read_reject_list(path: Path) -> set:
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(Path(line).name)
    return names


def index_source(source_dir: Path) -> dict:
    """{split: {filename: source_class}} -- see apply_curation_rejects.py
    for why this is indexed per split rather than flat (filenames repeat
    across train and val)."""
    index = {}
    for split in SPLITS:
        split_dir = source_dir / split
        if not split_dir.is_dir():
            continue
        index[split] = {jpg.name: cd.name
                        for cd in sorted(p for p in split_dir.iterdir() if p.is_dir())
                        for jpg in cd.glob("*.jpg")}
    return index


def record_rejects(source_dir: Path, split: str, names: set, index: dict, dry_run: bool) -> tuple:
    """Merge `names` into <source>/meta/{split}_curation_rejects.json,
    filed under each image's ORIGINAL source class -- which is not the
    class folder it was reviewed in, once build_data.py's merge map has
    folded chair/couch/table into `furniture`. Recording against the
    merged name would silently stop matching the moment the merge map
    changes."""
    path = source_dir / "meta" / f"{split}_curation_rejects.json"
    existing = json.load(open(path)) if path.exists() else {}
    merged = {c: set(v) for c, v in existing.items()}

    per_split = index.get(split, {})
    unresolved = []
    for name in names:
        src_class = per_split.get(name)
        if src_class is None:
            unresolved.append(name)
            continue
        merged.setdefault(src_class, set()).add(name)

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = {c: sorted(v) for c, v in sorted(merged.items()) if v}
        path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n")
    return sum(len(v) for v in merged.values()), unresolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="everyday_openimages160x120",
                        help="source dataset whose meta/ holds the durable reject record "
                             "(default: everyday_openimages160x120)")
    parser.add_argument("--dry-run", action="store_true", help="report what would happen, change nothing")
    args = parser.parse_args()

    review_dir = PROJECT_ROOT / "curation_review"
    curated_dir = PROJECT_ROOT / "data_hand_curated"
    rejected_dir = PROJECT_ROOT / "data_rejected"
    source_dir = PROJECT_ROOT / args.source
    archive_dir = review_dir / "_resolved_rejected_lists"

    if not review_dir.is_dir():
        raise SystemExit(f"{review_dir} not found -- run curation_pull.py first")
    if not source_dir.is_dir():
        raise SystemExit(f"source dataset not found: {source_dir}")

    print(f"indexing {source_dir.name}/ to map filenames back to source classes...")
    index = index_source(source_dir)
    print(f"  train {len(index.get('train', {})):,}, val {len(index.get('val', {})):,}")
    print()

    accepted_total = rejected_total = 0
    unreviewed = []
    all_unresolved = []

    for split in SPLITS:
        split_dir = review_dir / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            images = list_images(class_dir)
            if not images:
                continue

            list_path = find_reject_list(class_dir, split, cls)
            if list_path is None:
                unreviewed.append(f"{split}/{cls} ({len(images)} images)")
                continue

            rejects = read_reject_list(list_path)
            present = {p.name for p in images}
            stale = rejects - present
            rejects &= present

            n_rej = len(rejects)
            n_acc = len(images) - n_rej
            print(f"  {split}/{cls}: {n_acc} accepted, {n_rej} rejected"
                  + (f"  [{len(stale)} name(s) in the list not in this folder, ignored]" if stale else ""))

            if not args.dry_run:
                for p in images:
                    dst_root = rejected_dir if p.name in rejects else curated_dir
                    dst = dst_root / split / cls / p.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dst))
                # the gallery references files that have just moved away
                gal = class_dir / "_gallery.html"
                if gal.exists():
                    gal.unlink()
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(list_path), str(archive_dir / f"{split}_{cls}_{list_path.name}"))

            total, unresolved = record_rejects(source_dir, split, rejects, index, args.dry_run)
            all_unresolved.extend((split, n) for n in unresolved)
            accepted_total += n_acc
            rejected_total += n_rej

    print()
    if all_unresolved:
        print(f"WARNING: {len(all_unresolved)} rejected filename(s) were not found in {source_dir.name}/,")
        print("         so they went into data_rejected/ but NOT into the durable record. A rebuild")
        print("         of data/ would bring them back. Likely a --source mismatch.")
        for split, n in all_unresolved[:5]:
            print(f"         {split}: {n}")
        print()

    if unreviewed:
        print(f"Left untouched -- no reject list found ({len(unreviewed)}):")
        for u in unreviewed:
            print(f"    {u}")
        print("    (export a list from the gallery, or save an empty rejected_*.txt in the folder")
        print("     if the batch genuinely has nothing wrong with it)")
        print()

    if args.dry_run:
        print("--dry-run: nothing moved, nothing written.")
        return

    if accepted_total or rejected_total:
        n_cur = sum(len(list_images(c)) for s in SPLITS for c in (curated_dir / s).iterdir()
                    if (curated_dir / s).is_dir() and c.is_dir())
        print(f"Resolved: {accepted_total} accepted, {rejected_total} rejected.")
        print(f"data_hand_curated/ now holds {n_cur} image(s) total.")
        print()
        print("Next: pull another batch (python python_code/curation_pull.py), or if you have")
        print("enough, fine-tune:")
        print("    python python_code/train_rgb_cnn.py \\")
        print("        --data-dir data_hand_curated --fine-tune-from rgb_cnn_5x5 \\")
        print("        --rgb-block-width 5 --rgb-block-height 5 \\")
        print("        --classes people,computer,doors,fruit,car \\")
        print("        --artifacts-name rgb_cnn_5x5_finetuned")
    else:
        print("Nothing resolved.")


if __name__ == "__main__":
    main()
