#!/usr/bin/env python3
"""One command to put a trained RGB CNN run onto the firmware.

Replaces the five-step sequence that has to be done in the right order and
has bitten this project more than once:

    1. export_rgb_cnn_c_weights.py <run>   -> model_weights.h, network_info.json
    2. verify_rgb_cnn_c_export.py  <run>   -> rgb_synth_vectors.h, rgb_real_images.h
    3. back up the firmware's current headers
    4. copy all four files into esp32_cam/esp32_rgb_cnn/
    5. rebuild

WHY A SCRIPT. Two failure modes, both silent, both previously hit:

  * The exporter writes model_weights.h; the VERIFIER writes the two self-test
    vector headers, one directory up. Copying only model_weights.h leaves the
    boot self-test comparing a new model against another model's expected
    logits, which fails on device with no hint as to why. Doing it by hand
    means remembering that the two commands write to different places.

  * Both underlying scripts default to the `rgb_cnn` output directory when
    given no argument, so a bare run silently rebuilds from whatever is in
    there rather than from the run you just trained.

This script does the steps in order, refuses to install anything the verifier
did not pass, checks the four files actually agree with each other, and backs
up what it replaces.

It also prints a per-class precision and predicted-share table, because
accuracy alone hides the failure that matters most here: a model that
over-predicts one class scores *well* on balanced accuracy (which is macro
recall) while being obviously wrong on camera. See --help output of the
warning it prints.

Usage:
    python export_to_firmware.py                 # the `rgb_cnn` run
    python export_to_firmware.py rgb_5x5_ft      # a specific run
    python export_to_firmware.py --build         # also run `pio run`
    python export_to_firmware.py --upload        # build and flash
    python export_to_firmware.py --dry-run       # show what would change
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "output"
DEFAULT_FIRMWARE = PROJECT_ROOT / "esp32_cam" / "esp32_rgb_cnn"

# (source path relative to the run directory, destination relative to the
# firmware root). model_weights.h comes from esp32/ -- that is the deployment
# bundle the exporter writes -- while the vector headers sit one level up,
# where the verifier puts them.
FILES = [
    ("esp32/model_weights.h", "include/model_weights.h"),
    ("rgb_synth_vectors.h", "include/rgb_synth_vectors.h"),
    ("rgb_real_images.h", "include/rgb_real_images.h"),
    ("esp32/network_info.json", "network_info.json"),
]


def run_step(label: str, argv: list) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    result = subprocess.run([sys.executable, *argv], cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"\n{label} FAILED (exit {result.returncode}) -- nothing installed.")


def report_model(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "model_manifest.json").read_text())
    print(f"\n  classes ({manifest['num_classes']}): {manifest['class_names']}")
    print(f"  input   : {manifest['rgb_width']}x{manifest['rgb_height']}"
          f"  reduction={manifest.get('reduction', 'lanczos')}"
          f"  block={manifest.get('rgb_block_width', '-')}x{manifest.get('rgb_block_height', '-')}")
    print(f"  accuracy: float={manifest['float_test_accuracy']:.4f} "
          f"qat={manifest['qat_test_accuracy']:.4f} "
          f"int8={manifest['int8_reference_test_accuracy']:.4f} "
          f"agreement={manifest['int8_vs_qat_agreement']:.4f}")
    if manifest.get("fine_tuned_from"):
        print(f"  fine-tuned from '{manifest['fine_tuned_from']}' on "
              f"'{manifest.get('data_dir')}' at lr={manifest.get('fine_tune_lr')}")
    return manifest


def report_per_class(run_dir: Path, manifest: dict) -> None:
    """Per-class precision/recall and predicted-vs-actual share.

    Accuracy and balanced accuracy both hide over-prediction: balanced
    accuracy is macro-averaged RECALL, so a model that shouts one class
    scores well on it. The predicted-share column is what exposes that."""
    cm_path = run_dir / "confusion_matrix.json"
    if not cm_path.exists():
        return
    cm = json.loads(cm_path.read_text())["test"]["int8"]
    names = manifest["class_names"]
    total = sum(sum(r) for r in cm)
    print(f"\n  {'class':<14}{'recall':>8}{'prec':>8}{'pred%':>8}{'actual%':>9}")
    print("  " + "-" * 47)
    flagged = []
    for i, n in enumerate(names):
        support = sum(cm[i])
        predicted = sum(r[i] for r in cm)
        recall = cm[i][i] / support if support else 0.0
        prec = cm[i][i] / predicted if predicted else 0.0
        pshare, ashare = predicted / total * 100, support / total * 100
        print(f"  {n:<14}{recall * 100:>7.1f}%{prec * 100:>7.1f}%{pshare:>7.1f}%{ashare:>8.1f}%")
        # A class predicted much more often than it occurs is being
        # over-predicted; on camera that reads as "it thinks everything is X".
        if pshare > ashare * 1.3 and pshare - ashare > 2.0:
            flagged.append((n, pshare, ashare, prec))
    for n, p, a, prec in flagged:
        print(f"\n  WARNING: '{n}' is predicted on {p:.1f}% of test images but is only "
              f"{a:.1f}% of them\n           (precision {prec * 100:.1f}%). Expect it to be "
              f"over-called on camera.")
    if flagged:
        print("           Common cause: fine-tuning on a class-balanced set "
              "(data_hand_curated is\n           balanced by construction) strips the "
              "majority-class prior.")


def check_consistency(run_dir: Path, manifest: dict) -> None:
    """The four files must describe one model. Cheap to check, and the
    failure it prevents (mismatched vector headers) shows up only as a
    device-side self-test failure with no explanation."""
    weights = (run_dir / "esp32" / "model_weights.h").read_text()
    n = manifest["num_classes"]
    if f"#define MODEL_NUM_CLASSES {n}" not in weights:
        raise SystemExit(f"model_weights.h does not declare MODEL_NUM_CLASSES {n}")
    for h in ("rgb_synth_vectors.h", "rgb_real_images.h"):
        text = (run_dir / h).read_text()
        if f"[{n}]" not in text:
            raise SystemExit(
                f"{h} has no [{n}]-wide logit rows -- it was generated for a different "
                f"model. Re-run verify_rgb_cnn_c_export.py {run_dir.name}.")
    net = json.loads((run_dir / "esp32" / "network_info.json").read_text())
    if net["class_names"] != manifest["class_names"]:
        raise SystemExit("network_info.json disagrees with model_manifest.json on class names")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default="rgb_cnn",
                        help="output/ subdirectory to deploy (default: rgb_cnn)")
    parser.add_argument("--firmware", default=str(DEFAULT_FIRMWARE),
                        help=f"PlatformIO project root (default: {DEFAULT_FIRMWARE})")
    parser.add_argument("--skip-export", action="store_true",
                        help="reuse the existing generated headers instead of re-running the exporter")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do NOT do this: skips the bit-exactness check against the Python int8 reference")
    parser.add_argument("--force", action="store_true",
                        help="install even if the firmware already has these exact headers")
    parser.add_argument("--build", action="store_true", help="run `pio run` afterwards")
    parser.add_argument("--upload", action="store_true", help="run `pio run -t upload` afterwards")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, copy nothing")
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.run
    firmware = Path(args.firmware).resolve()
    if not (run_dir / "model_manifest.json").exists():
        raise SystemExit(f"no trained run at {run_dir} (missing model_manifest.json)")
    if not (firmware / "platformio.ini").exists():
        raise SystemExit(f"{firmware} is not a PlatformIO project")

    print(f"run     : {run_dir}")
    print(f"firmware: {firmware}")
    manifest = report_model(run_dir)
    report_per_class(run_dir, manifest)

    if not args.skip_export:
        run_step("STEP 1/4  export C weights", ["export_rgb_cnn_c_weights.py", args.run])
    if not args.skip_verify:
        run_step("STEP 2/4  verify bit-exactness (also writes the self-test vectors)",
                 ["verify_rgb_cnn_c_export.py", args.run])
    else:
        print("\n!! --skip-verify: installing weights that have NOT been checked against "
              "the Python\n   int8 reference. The device self-test is then the only check left.")

    missing = [s for s, _ in FILES if not (run_dir / s).exists()]
    if missing:
        raise SystemExit(f"missing generated file(s): {missing} -- run without --skip-export/--skip-verify")
    check_consistency(run_dir, manifest)

    changed = [(s, d) for s, d in FILES
               if not (firmware / d).exists()
               or (run_dir / s).read_bytes() != (firmware / d).read_bytes()]
    print(f"\n{'=' * 70}\nSTEP 3/4  install\n{'=' * 70}")
    if not changed and not args.force:
        print("  Firmware already has this exact model -- nothing to copy.")
        print("  (--force to copy anyway.)")
    elif args.dry_run:
        for s, d in changed:
            print(f"  would replace {d}")
        print("\n--dry-run: nothing copied.")
        return
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = firmware / f"_headers_backup_{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        for _, d in FILES:
            src = firmware / d
            if src.exists():
                shutil.copy2(src, backup / Path(d).name)
        print(f"  backed up current headers -> {backup.name}/")
        for s, d in (FILES if args.force else changed):
            (firmware / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / s, firmware / d)
            print(f"  installed {d}")

    print(f"\n{'=' * 70}\nSTEP 4/4  build\n{'=' * 70}")
    if args.build or args.upload:
        pio = Path.home() / ".platformio" / "penv" / "bin" / "pio"
        pio = str(pio) if pio.exists() else "pio"
        cmd = [pio, "run"] + (["-t", "upload"] if args.upload else [])
        if subprocess.run(cmd, cwd=firmware).returncode != 0:
            raise SystemExit("build failed")
    else:
        print(f"  skipped (--build to compile, --upload to flash)")
        print(f"  manually:  cd {firmware}  &&  pio run -t upload")

    print(f"\nDone. Firmware carries '{args.run}': {manifest['num_classes']} classes, "
          f"int8 test {manifest['int8_reference_test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
