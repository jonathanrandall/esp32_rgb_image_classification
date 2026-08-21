#!/usr/bin/env python3

"""
Pull a review batch out of open_images_tmp/ into select_images/.

Moves up to --count (default 100) random images per class, per split
(train and val, both processed by default), from
../open_images_tmp/<split>/<class>/ into ./<split>/<class>/ (this
script lives in data_curation/select_images/, so "." is select_images
itself).

Idempotent per (split, class): if select_images/<split>/<class>/ already
has >= --count images sitting in it (a batch already pulled and not yet
resolved), that class/split is skipped rather than topped up further --
run resolve_selection.py first to clear a batch before pulling a new one.
(This is what "unless there are already 100 images ... for that class"
in the original spec means in this implementation -- interpreted as
select_images already having a full batch, not open_images_tmp, since
checking open_images_tmp would produce the opposite, useless behavior:
skipping exactly when there's plenty left to pull from. Flag this to
whoever's using this if that's not the intended behavior.)

If fewer than --count images remain in open_images_tmp for a class/split,
whatever's left is moved and a warning is printed -- it does not skip
just because it can't reach the full count.

Usage:
    python select_batch.py                          # all classes, both splits, batch of 100
    python select_batch.py --classes birds,car       # just these classes
    python select_batch.py --count 50 --splits val   # smaller batch, val only
    python select_batch.py --seed 42                 # reproducible sample
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

HERE = Path(__file__).resolve().parent          # data_curation/select_images/
DATA_CURATION_ROOT = HERE.parent                # data_curation/
TMP_ROOT = DATA_CURATION_ROOT / "open_images_tmp"
SELECT_ROOT = HERE


def list_images(d: Path) -> list:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def discover_classes() -> list:
    train_dir = TMP_ROOT / "train"
    if train_dir.is_dir():
        return sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if TMP_ROOT.is_dir():
        # fall back to whichever split exists
        for split_dir in TMP_ROOT.iterdir():
            if split_dir.is_dir():
                return sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=str, default=None,
                         help="comma-separated class names (default: all classes found in open_images_tmp/train)")
    parser.add_argument("--splits", type=str, default="train,val", help="comma-separated splits (default: train,val)")
    parser.add_argument("--count", type=int, default=100, help="target batch size per class/split (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for a reproducible sample (default: unseeded)")
    args = parser.parse_args()

    if not TMP_ROOT.is_dir():
        raise SystemExit(f"open_images_tmp not found at {TMP_ROOT} -- run the data_curation setup steps first")

    rng = random.Random(args.seed)

    classes = args.classes.split(",") if args.classes else discover_classes()
    if not classes:
        raise SystemExit(f"no classes found under {TMP_ROOT} and none given via --classes")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"classes: {', '.join(classes)}")
    print(f"splits: {', '.join(splits)}")
    print(f"target batch size: {args.count}\n")

    for split in splits:
        for cls in classes:
            src_dir = TMP_ROOT / split / cls
            dst_dir = SELECT_ROOT / split / cls

            existing = len(list_images(dst_dir))
            if existing >= args.count:
                print(f"{split}/{cls}: select_images already has {existing} >= {args.count}, skipping")
                continue

            need = args.count - existing
            available = list_images(src_dir)
            if not available:
                print(f"{split}/{cls}: nothing left in open_images_tmp, skipping (0 available)")
                continue

            chosen = rng.sample(available, min(need, len(available)))
            if len(available) < need:
                print(f"{split}/{cls}: WARNING only {len(available)} available in open_images_tmp, "
                      f"wanted {need} more (had {existing} already) -- moving all of them")

            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in chosen:
                shutil.move(str(p), str(dst_dir / p.name))

            print(f"{split}/{cls}: moved {len(chosen)} (had {existing}, now {existing + len(chosen)})")


if __name__ == "__main__":
    main()
