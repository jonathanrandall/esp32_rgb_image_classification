"""openimages_8class-to-JPEG dataset build, shared by both training scripts.

Expects/produces `data/{train,val,test}/<class_name>/*.jpg` at the project
root (one level up from python_code/), resized+re-encoded at
`Config.capture_width x capture_height`. Source images come from
`openimages_8class/{train,val}/<class_name>/*.jpg` (built by
`get_open_image_data.py`; the "background" folder ships empty -- 0 images
in both train/ and val/ -- and is excluded, same as CLASS_NAMES in
config.py). Unlike the project's earlier CIFAR-100/Caltech-256 sources,
this dataset already ships a train/val split -- val is used as-is (not
re-split), and a test split is carved out of train only, since no test
split is provided. Cached the same way as before: auto-rebuilt whenever
the on-disk JPEGs don't match the current Config (wrong size or wrong
class set), otherwise reused as-is.
"""

import math
import shutil
from pathlib import Path

import jpeglib
import numpy as np
from PIL import Image

from .config import CLASS_NAMES, Config

OPENIMAGES_SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "openimages_8class"


def make_capture_resolution_jpeg(src_image: Image.Image, dst_path: Path, cfg: Config) -> None:
    """Resize (never crop) to exactly capture_width x capture_height and
    re-encode as a fixed-quality baseline JPEG at cfg.chroma_subsampling.
    Must be used identically for train/val/test data, and identically by
    both models, so resize target, JPEG quality, and chroma subsampling
    never diverge from what a real camera path would assume -- done
    unconditionally (even when the source is already at the target size)
    so the fixed pipeline assumptions hold regardless of whatever settings
    get_open_image_data.py's JPEG encoder used, and so a different
    --capture-width/--capture-height/--chroma-subsampling still works
    correctly. PIL's `subsampling` save parameter accepts "4:2:0"/"4:2:2"
    strings directly, matching Config.chroma_subsampling's values."""
    rgb = src_image.convert("RGB")
    resized = rgb.resize((cfg.capture_width, cfg.capture_height), Image.BILINEAR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(dst_path, format="JPEG", quality=cfg.jpeg_quality, subsampling=cfg.chroma_subsampling)


def dataset_matches_config(data_dir: Path, cfg: Config) -> bool:
    """True if data/{train,val,test} exist, CONTAIN the classes this run
    actually needs, those JPEGs are already sized cfg.capture_width x
    cfg.capture_height, AND their actual chroma block grid matches
    cfg.c_rows x cfg.c_cols -- this last check is what catches a data/
    built under a different chroma_subsampling (e.g. 4:2:0 vs 4:2:2) at the
    *same* pixel resolution, which the size check alone can't see (see
    issues.md's camera chroma-subsampling mismatch for why this distinction
    matters).

    CLASS CHECK IS A SUBSET TEST AGAINST cfg.active_class_names, NOT
    EQUALITY AGAINST CLASS_NAMES. It used to require data/ to hold exactly
    the full CLASS_NAMES list, which asked the wrong question twice:

      - Training a 5-class subset (`--classes people,computer,...`) against
        a data/ containing only those 5 was reported as a MISMATCH, even
        though every class the run needs is present. The "fix" was a
        rebuild -- and a rebuild deletes data/ (see
        build_openimages_dataset) before it can fail.
      - It ignored --classes entirely, so the decision to destroy and
        rebuild was driven by classes the run was never going to touch.

    Extra classes in data/ are harmless: build_spatial_split only reads
    cfg.active_class_names. What matters is that nothing needed is missing,
    which is exactly what a subset test asks."""
    required = set(cfg.active_class_names)
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            return False
        class_dirs = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if not required <= class_dirs:
            return False
        jpgs = list(split_dir.rglob("*.jpg"))
        if not jpgs:
            return False
        with Image.open(jpgs[0]) as im:
            if im.size != (cfg.capture_width, cfg.capture_height):
                return False
        dct = jpeglib.read_dct(str(jpgs[0]))
        if dct.Cb.shape[:2] != (cfg.c_rows, cfg.c_cols):
            return False
    return True


def _discover_openimages_class_dirs(source_dir: Path, split: str, class_names: list | None = None) -> dict:
    """{class_name: folder_path} for one of source_dir's "train"/"val"
    subfolders. `class_names` defaults to CLASS_NAMES (every existing
    caller's behavior, unchanged); build_data.py passes a different list
    when building a merged/renamed class taxonomy from an already
    merged+filtered source directory. Raises if a folder is missing or
    empty (an empty class folder -- like the old "background" -- silently
    producing zero training examples is exactly the kind of bug this
    project has hit before; better to fail loudly at dataset-build time)."""
    class_names = class_names if class_names is not None else CLASS_NAMES
    split_dir = source_dir / split
    found = {}
    for name in class_names:
        class_dir = split_dir / name
        if not class_dir.is_dir():
            raise ValueError(f"{class_dir} does not exist -- the requested class list and the dataset on disk have diverged.")
        n_images = sum(1 for _ in class_dir.glob("*.jpg"))
        if n_images == 0:
            raise ValueError(
                f"{class_dir} has no images -- if this is intentional (like the excluded "
                f"'background' folder), remove it from the class list instead of "
                f"letting it silently produce zero training examples for that class."
            )
        found[name] = class_dir
    return found


def _stratified_two_way_split(n: int, test_fraction: float, seed: int) -> tuple:
    """Shuffled indices for one class's train-split images, split into
    (train, test) index arrays -- used to carve a test set out of
    openimages_8class's train/ folder, since it ships no test split of its
    own (val/ is used as-is, see build_openimages_dataset)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, round(n * test_fraction)) if n > 1 else 0
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def build_openimages_dataset(
    data_dir: Path, cfg: Config, test_fraction: float = 0.15,
    source_dir: Path | None = None, class_names: list | None = None,
) -> None:
    """(Re)build data/{train,val,test}/<class_name>/*.jpg from
    `source_dir` (default OPENIMAGES_SOURCE_DIR, i.e. openimages_8class/),
    resized to cfg.capture_width x capture_height. Always writes all of
    `class_names` (default CLASS_NAMES) regardless of cfg.selected_classes
    -- filtering to a subset happens later, when reading, so switching
    subsets never requires a rebuild (same pattern used for every dataset
    this project has had). `class_names` lets a caller (build_data.py)
    build data/ under a different class taxonomy than CLASS_NAMES -- e.g.
    a merged class list -- from an already-prepared source_dir whose
    folder names match that list, without duplicating this function's
    resize/test-split logic. source_dir/val/ is copied through as
    data/val/ unchanged (it's already a real, separately-collected
    validation set -- splitting it further would just shrink it for no
    reason); data/test/ is carved out of source_dir/train/ by a per-class
    stratified shuffle, and what's left becomes data/train/."""
    source_dir = source_dir if source_dir is not None else OPENIMAGES_SOURCE_DIR
    class_names = class_names if class_names is not None else CLASS_NAMES
    train_class_dirs = _discover_openimages_class_dirs(source_dir, "train", class_names)
    val_class_dirs = _discover_openimages_class_dirs(source_dir, "val", class_names)

    if data_dir.exists():
        shutil.rmtree(data_dir)

    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for class_idx, class_name in enumerate(class_names):
        train_paths = sorted(train_class_dirs[class_name].glob("*.jpg"))
        train_idx, test_idx = _stratified_two_way_split(len(train_paths), test_fraction, cfg.seed + class_idx)

        for split, indices in (("train", train_idx), ("test", test_idx)):
            for i in indices:
                src_path = train_paths[i]
                try:
                    with Image.open(src_path) as img:
                        make_capture_resolution_jpeg(img, data_dir / split / class_name / src_path.name, cfg)
                    split_counts[split] += 1
                except Exception as exc:
                    skipped += 1
                    print(f"  skipping unreadable image {src_path}: {exc!r}")

        for src_path in sorted(val_class_dirs[class_name].glob("*.jpg")):
            try:
                with Image.open(src_path) as img:
                    make_capture_resolution_jpeg(img, data_dir / "val" / class_name / src_path.name, cfg)
                split_counts["val"] += 1
            except Exception as exc:
                skipped += 1
                print(f"  skipping unreadable image {src_path}: {exc!r}")

    for split in ("train", "val", "test"):
        print(f"  {split}: {split_counts[split]} images written at {cfg.capture_width}x{cfg.capture_height}")
    if skipped:
        print(f"  skipped {skipped} unreadable image(s)")


def generate_synthetic_dataset(data_dir: Path, cfg: Config, samples_per_class: int = 120) -> None:
    """Placeholder dataset: structured noise with a class-dependent
    low-frequency bias, so DC/low-AC coefficients carry a learnable signal.
    Fallback only, used if openimages_8class/ can't be found or a class
    folder is missing/empty."""
    print(f"  openimages_8class source unavailable -- generating a synthetic placeholder dataset under {data_dir}.")
    rng = np.random.default_rng(cfg.seed)
    split_counts = {
        "train": samples_per_class,
        "val": max(1, samples_per_class // 4),
        "test": max(1, samples_per_class // 4),
    }

    if data_dir.exists():
        shutil.rmtree(data_dir)

    for split, n_per_class in split_counts.items():
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                h, w = cfg.capture_height * 2, cfg.capture_width * 2
                base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8).astype(np.float32)
                yy, xx = np.mgrid[0:h, 0:w]
                bias = 40.0 * np.sin((class_idx + 1) * (xx / w) * math.pi)
                base[..., 0] += bias
                base = np.clip(base, 0, 255).astype(np.uint8)
                img = Image.fromarray(base)
                make_capture_resolution_jpeg(img, class_dir / f"{class_name}_{split}_{i:04d}.jpg", cfg)

    print(f"  Synthetic dataset written to {data_dir}")


def ensure_dataset(data_dir: Path, cfg: Config, source_dir: Path | None = None) -> None:
    """Build data/ if missing, sized differently than cfg, or built under a
    different chroma_subsampling than cfg; no-op (fast) if it already
    matches. Safe to call at the top of every training run. `source_dir`
    picks which openimages-shaped source directory to build from (default
    OPENIMAGES_SOURCE_DIR, i.e. openimages_8class/)."""
    if dataset_matches_config(data_dir, cfg):
        print(f"Found existing dataset under {data_dir} already at {cfg.capture_width}x{cfg.capture_height} ({cfg.chroma_subsampling})")
        return
    print(f"Dataset under {data_dir} missing, wrong size, or wrong chroma subsampling for "
          f"{cfg.capture_width}x{cfg.capture_height} ({cfg.chroma_subsampling}) -- rebuilding.")
    try:
        build_openimages_dataset(data_dir, cfg, source_dir=source_dir)
    except Exception as exc:
        # DO NOT fall back to generate_synthetic_dataset() here. It used to,
        # and the failure was silent and expensive:
        #
        #   1. build_openimages_dataset() rmtree's data/ BEFORE it can fail,
        #      so the real dataset is already gone by this point.
        #   2. The fallback refilled data/ with random-noise images carrying
        #      a per-class brightness bias.
        #   3. Training then ran to completion and printed a plausible
        #      accuracy table computed entirely on noise.
        #
        # The only signal was one line of console output. Worse, the rebuild
        # CANNOT succeed for the current taxonomy: data/'s merged classes
        # (computer, furniture, crockery) exist only as build_data.py's
        # output, never in the raw --dataset-source, so discovery always
        # raises. That makes this path purely destructive.
        raise SystemExit(
            f"\nRebuilding {data_dir} from {source_dir} FAILED:\n"
            f"  {exc}\n\n"
            f"{data_dir} has already been deleted -- that happens before the failure.\n\n"
            f"Rebuild it with build_data.py, which is the only thing that can produce it\n"
            f"(it merges source classes into the taxonomy in dct_common/class_names.json,\n"
            f"e.g. laptop+keyboard+monitor -> computer; those merged names do not exist in\n"
            f"the raw source directory):\n\n"
            f"    cd python_code && python build_data.py\n\n"
            f"Anything you added to {data_dir} by hand is gone -- board captures live in\n"
            f"board_captures/ and must be re-copied after the rebuild.\n"
        ) from exc
