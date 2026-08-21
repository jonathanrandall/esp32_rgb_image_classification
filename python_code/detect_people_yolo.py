#!/usr/bin/env python3
"""Run YOLO11m person detection over every class folder in an
everyday_openimages<res>/ source directory, and write
meta/{train,val}_yolo_people.json -- a per-image "how confident is YOLO
that there's a person in this photo" signal, independent of and more
reliable than the existing meta/{train,val}_intersections.json's
people-intersection flag.

Why this exists: get_everyday_openimages_data.py's intersections.json is
built from Open Images V7's OWN detection annotations (see that script's
`label_types=["detections"]` FiftyOne call) -- and those annotations are
known to be incomplete. Open Images' crowd-sourced/automated labeling
boxes salient/queried objects, not necessarily every person in every
photo (background people, partial people at the frame edge, small/blurry
people are frequently left unboxed). filter_intersections.py's people
filter can only drop what Open Images actually annotated -- confirmed
directly: after that filter, a visible fraction of `furniture` images
still had unboxed people in them. Running a real, general-purpose object
detector (YOLO11m, COCO-pretrained, 80 classes, "person" = class 0) over
the actual saved dataset images closes that gap independent of whatever
Open Images' annotators happened to box.

Design mirrors get_everyday_openimages_data.py's own philosophy: detect
once, record the raw signal (here: max person-detection CONFIDENCE per
image, not a baked-in yes/no), and let filtering apply whatever
confidence threshold it wants later without re-running YOLO -- see
filter_intersections.py's YOLO_PERSON_CONF_THRESHOLD.

Weights: python_code/yolo11m.pt (a standard Ultralytics COCO11m
checkpoint, copied in from an existing local cache rather than
re-downloaded -- see this repo's notes on uncertain internet egress).
Runs on GPU automatically if available (this machine has one), CPU
otherwise -- see get_device()-style fallback below.

Usage:
    python detect_people_yolo.py [--source everyday_openimages160x120] [--batch-size 64]

Writes <source>/meta/{train,val}_yolo_people.json:
    {class_name: {filename: max_person_confidence_float}}
0.0 means "no person detected at all" (YOLO's own internal NMS/objectness
threshold still applies before any box is returned), not "detected with
confidence 0".
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = Path(__file__).resolve().parent / "yolo11m.pt"
PERSON_CLASS_ID = 0  # COCO "person" -- standard across every YOLO11 COCO-pretrained checkpoint


def detect_split(model: YOLO, device: str, source_dir: Path, split: str, batch_size: int) -> dict:
    result = {}
    class_dirs = sorted(p for p in (source_dir / split).iterdir() if p.is_dir())
    for class_dir in class_dirs:
        class_name = class_dir.name
        jpg_paths = sorted(class_dir.glob("*.jpg"))
        confs = {}
        for i in range(0, len(jpg_paths), batch_size):
            batch = jpg_paths[i:i + batch_size]
            preds = model.predict([str(p) for p in batch], device=device, verbose=False)
            for jpg_path, pred in zip(batch, preds):
                person_mask = pred.boxes.cls == PERSON_CLASS_ID
                person_confs = pred.boxes.conf[person_mask]
                confs[jpg_path.name] = float(person_confs.max()) if len(person_confs) else 0.0
        n_detected = sum(1 for c in confs.values() if c > 0.0)
        print(f"  {split}/{class_name}: {n_detected}/{len(confs)} images with >=1 person detection (any confidence)")
        result[class_name] = confs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="everyday_openimages160x120",
                         help="source directory name, relative to the project root (default: everyday_openimages160x120)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / args.source
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {WEIGHTS_PATH} on device={device}")
    model = YOLO(str(WEIGHTS_PATH))

    meta_dir = source_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    for split in ("train", "val"):
        print(f"=== {split} ===")
        result = detect_split(model, device, source_dir, split, args.batch_size)
        out_path = meta_dir / f"{split}_yolo_people.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
