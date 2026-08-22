#!/usr/bin/env python3
"""Pull a review batch out of data_tmp/ into curation_review/.

Step 1 and 2 of the hand-curation loop that feeds fine-tuning. The full
loop, and why the directories are shaped this way:

    data/                     built by build_data.py. NEVER touched here.
                              Stays the untouched set the base model
                              trains on, and supplies the test split.

    data_tmp/                 working pool, created here as a copy of
                              data/{train,val} on first run. Images move
                              OUT of it, so what remains is exactly
                              "not yet reviewed".

    curation_review/          the batch currently under review. Empty
                              between batches.

    data_hand_curated/        accepted images accumulate here across
                              batches. This is the fine-tuning set.

    data_rejected/            rejected images land here. Nothing is
                              deleted, so any call is reversible.

Why a separate `curation_review/` rather than pulling straight into
data_hand_curated/: data_hand_curated/ grows every batch, so "does this
folder already hold a full batch?" stops being answerable once it has
300 accepted images in it. Keeping the pending batch in its own
directory makes the idempotency check trivial and unambiguous -- a
non-empty curation_review/<split>/<class>/ means "reviewed or not, this
batch is unresolved", and curation_resolve.py is what clears it.

test/ is deliberately never pulled from. The whole point of the exercise
is to compare a fine-tuned model against the base one, which requires
both to be scored on the same untouched test split.

Usage:
    python curation_pull.py                       # 100 train + 20 val per class
    python curation_pull.py --train-count 50 --val-count 10
    python curation_pull.py --classes people,car   # just these
    python curation_pull.py --seed 42              # reproducible sample

Then review, per the printed instructions, and run curation_resolve.py.
"""

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")


def list_images(d: Path) -> list:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def seed_working_pool(data_dir: Path, tmp_dir: Path) -> None:
    """Copy data/{train,val} -> data_tmp/ on first run only.

    Copy, not move: data/ has to stay intact, both because it is what the
    base model trains on and because build_data.py's output is the only
    record of what the untouched dataset was."""
    if tmp_dir.exists():
        return
    print(f"-- First run: seeding working pool {tmp_dir.name}/ from {data_dir.name}/ --")
    for split in SPLITS:
        src = data_dir / split
        if not src.is_dir():
            raise SystemExit(f"{src} not found -- run build_data.py first")
        dst = tmp_dir / split
        shutil.copytree(src, dst)
        n = sum(len(list_images(c)) for c in dst.iterdir() if c.is_dir())
        print(f"   {split}: copied {n:,} images")
    print(f"   (test/ deliberately not copied -- it stays untouched for evaluation)")
    print()


def discover_classes(tmp_dir: Path) -> list:
    for split in SPLITS:
        d = tmp_dir / split
        if d.is_dir():
            return sorted(p.name for p in d.iterdir() if p.is_dir())
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data",
                        help="untouched dataset to seed the working pool from (default: data)")
    parser.add_argument("--classes", type=str, default=None,
                        help="comma-separated class names (default: every class in the working pool)")
    parser.add_argument("--train-count", type=int, default=100, help="train images per class per batch (default: 100)")
    parser.add_argument("--val-count", type=int, default=20, help="val images per class per batch (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for a reproducible sample (default: unseeded)")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    tmp_dir = PROJECT_ROOT / "data_tmp"
    review_dir = PROJECT_ROOT / "curation_review"

    seed_working_pool(data_dir, tmp_dir)

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else discover_classes(tmp_dir)
    if not classes:
        raise SystemExit(f"no classes found under {tmp_dir}")

    rng = random.Random(args.seed)
    per_split_count = {"train": args.train_count, "val": args.val_count}

    print(f"classes: {', '.join(classes)}")
    print(f"batch size: train {args.train_count}, val {args.val_count} per class")
    print()

    pulled_dirs = []
    total_moved = 0
    for split in SPLITS:
        want = per_split_count[split]
        if want <= 0:
            continue
        for cls in classes:
            src_dir = tmp_dir / split / cls
            dst_dir = review_dir / split / cls

            pending = list_images(dst_dir)
            if pending:
                print(f"  {split}/{cls}: {len(pending)} already awaiting review -- skipping "
                      f"(resolve it first)")
                pulled_dirs.append(dst_dir)
                continue

            available = list_images(src_dir)
            if not available:
                print(f"  {split}/{cls}: working pool exhausted, nothing left to pull")
                continue

            chosen = rng.sample(available, min(want, len(available)))
            if len(available) < want:
                print(f"  {split}/{cls}: WARNING only {len(available)} left in the pool, wanted {want} "
                      f"-- taking all of them")
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in chosen:
                shutil.move(str(p), str(dst_dir / p.name))
            total_moved += len(chosen)
            pulled_dirs.append(dst_dir)
            print(f"  {split}/{cls}: pulled {len(chosen)}, {len(available) - len(chosen)} left in pool")

    print()
    if not pulled_dirs:
        print("Nothing to review. The working pool is empty -- every image has been through curation.")
        return

    print("=" * 70)
    print(f"{total_moved} image(s) pulled for review. Next:")
    print("=" * 70)
    print()
    print("1. Build the contact sheets:")
    print(f"   python python_code/make_gallery.py \\")
    for d in pulled_dirs:
        print(f"       {d.relative_to(PROJECT_ROOT)} \\")
    print()
    print("2. Open each _gallery.html, click the bad images, hit 'Export rejected list'.")
    print("   Keep the downloaded filename as-is -- curation_resolve.py reads the split")
    print("   and class out of it.")
    print()
    print("3. Resolve the batch:")
    print("   python python_code/curation_resolve.py")


if __name__ == "__main__":
    main()
