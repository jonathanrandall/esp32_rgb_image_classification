#!/usr/bin/env python3

"""
Resolve a reviewed batch: move accepted images back into openimages_160x120/,
rejected images into openimages_160x120_rejected/, based on the
"rejected_<key>.txt" list exported by python_code/make_gallery.py's
"Export rejected list" button (see python_code/documents/gallery_instructions.md).

For each <split>/<class> folder under select_images/ (i.e. next to this
script), looks for a reject list in two places:
  1. right there in select_images/<split>/<class>/ itself (any
     "rejected_*.txt" file -- move/save the browser download there
     before running this script), or
  2. ~/Downloads/rejected_<key>.txt, where <key> is exactly what
     make_gallery.py names it for that directory
     ("select_images_<split>_<class>") -- so this also works straight
     off a fresh browser download with no manual file-moving, as long
     as the export happened on this machine.

A folder with NO reject list found in either place is left completely
untouched and skipped (not treated as "everything accepted") -- that
means it hasn't been reviewed yet.

Once resolved, a class/split folder's images are gone from
select_images/ (moved out, not copied), its stale _gallery.html is
deleted (it would otherwise reference files that no longer exist there),
and the reject list itself is archived into
select_images/_resolved_rejected_lists/ rather than left where it was
found, so re-running this script doesn't re-process an already-resolved
folder.

After resolving everything it can, deletes the project's stale
../../data/ directory (built from a previous, uncurated pull) so the
next training run is forced to rebuild it fresh from whatever's
currently in openimages_160x120/ -- see the printed reminder at the end
for the exact command.

Usage:
    python resolve_selection.py
"""

import shutil
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

HERE = Path(__file__).resolve().parent                  # data_curation/select_images/
DATA_CURATION_ROOT = HERE.parent                         # data_curation/
PROJECT_ROOT = DATA_CURATION_ROOT.parent                 # project root
OPENIMAGES_DIR = PROJECT_ROOT / "openimages_160x120"
OPENIMAGES_REJECTED_DIR = PROJECT_ROOT / "openimages_160x120_rejected"
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = Path.home() / "Downloads"
ARCHIVE_DIR = HERE / "_resolved_rejected_lists"


def list_images(d: Path) -> list:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def find_reject_list(split: str, cls: str, class_dir: Path) -> Path | None:
    local_matches = sorted(class_dir.glob("rejected_*.txt"))
    if local_matches:
        return local_matches[0]
    key = f"select_images_{split}_{cls}"
    downloaded = DOWNLOADS_DIR / f"rejected_{key}.txt"
    if downloaded.is_file():
        return downloaded
    return None


def main():
    if not OPENIMAGES_DIR.is_dir():
        raise SystemExit(f"{OPENIMAGES_DIR} not found -- expected the (possibly empty) curated-dataset skeleton to exist")

    ARCHIVE_DIR.mkdir(exist_ok=True)

    total_accepted = 0
    total_rejected = 0
    resolved_any = False

    for split_dir in sorted(HERE.iterdir()):
        if not split_dir.is_dir() or split_dir.name.startswith("_"):
            continue
        split = split_dir.name
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cls = class_dir.name

            reject_list_path = find_reject_list(split, cls, class_dir)
            if reject_list_path is None:
                images_here = list_images(class_dir)
                if images_here:
                    print(f"{split}/{cls}: no exported reject list found ({len(images_here)} images waiting) -- skipping")
                continue

            rejected_names = {
                line.strip() for line in reject_list_path.read_text().splitlines() if line.strip()
            }

            images = list_images(class_dir)
            accept_dir = OPENIMAGES_DIR / split / cls
            reject_dir = OPENIMAGES_REJECTED_DIR / split / cls
            accept_dir.mkdir(parents=True, exist_ok=True)
            reject_dir.mkdir(parents=True, exist_ok=True)

            n_accepted = 0
            n_rejected = 0
            for p in images:
                if p.name in rejected_names:
                    shutil.move(str(p), str(reject_dir / p.name))
                    n_rejected += 1
                else:
                    shutil.move(str(p), str(accept_dir / p.name))
                    n_accepted += 1

            gallery_html = class_dir / "_gallery.html"
            if gallery_html.is_file():
                gallery_html.unlink()

            archived_list = ARCHIVE_DIR / f"select_images_{split}_{cls}.txt"
            shutil.move(str(reject_list_path), str(archived_list))

            print(f"{split}/{cls}: {n_accepted} accepted, {n_rejected} rejected "
                  f"(reject list: {reject_list_path.name}, archived)")
            total_accepted += n_accepted
            total_rejected += n_rejected
            resolved_any = True

    print(f"\nTotal: {total_accepted} accepted, {total_rejected} rejected")

    if resolved_any and DATA_DIR.is_dir():
        shutil.rmtree(DATA_DIR)
        print(f"\nDeleted stale {DATA_DIR} (built from the pre-curation dataset).")
        print("Rebuild it by running any train_*.py script from python_code/ -- e.g. the "
              "'Current best config' command in python_code/README.md -- which will build "
              "data/ fresh from openimages_160x120/'s current (curated) contents.")
    elif resolved_any:
        print(f"\n{DATA_DIR} doesn't exist yet -- nothing to invalidate.")
    else:
        print("\nNothing resolved this run -- no reject lists found anywhere.")


if __name__ == "__main__":
    main()
