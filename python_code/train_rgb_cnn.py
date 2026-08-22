#!/usr/bin/env python3
"""Train a plain RGB-pixel CNN: float -> QAT -> bit-exact int8, same
three-stage discipline as train_cnn.py, but reading full decoded RGB
pixels instead of JPEG DCT coefficients. This is the pixel-domain arm of
the DCT-vs-RGB comparison: it exists to be measured against the DCT
models on both accuracy and, more importantly, real on-device latency.

It IS deployed. Export with export_rgb_cnn_c_weights.py (ESP-NN
accelerated, NHWC activations) into `output/rgb_cnn/esp32/`, and the
firmware lives in `esp32_cam/esp32_rgb_cnn/`. An earlier version of this
docstring said no exporter existed or was planned and that the model was
not a firmware target -- both are now false.

Note the deployment does NOT do a JPEG-to-RGB decode, which is what that
old note assumed. The firmware captures RGB565 directly, so the pipeline
cost lands on the other side: an RGB capture can no longer double as the
JPEG stream, so streaming needs a software JPEG ENCODE (~17-20 ms/frame,
flat across quality) that the DCT pipeline gets for free from the one
capture. That asymmetry is a headline result of the comparison, not a
footnote -- see the RGB frame-rate project notes.

Reuses `../data/{train,val,test}/<class>/*.jpg` -- the exact same JPEGs
every other script in this project trains on, built the same way
(ensure_dataset(), --dataset-source, etc.). The only new knob is
--downsample-factor: data/ is still built at the full --capture-width x
--capture-height (so it stays shared with every other script regardless
of this setting), and this script resizes DOWN from that at feature-
extraction time (see dct_common/rgb_features.py). --downsample-factor 1
(default) trains on full-resolution decoded pixels; --downsample-factor 4
on a capture-width=160/height=120 image trains on 40x30 pixels, etc.

Architecture mirrors train_cnn.py's conv-stack idea (configurable channel
widths via CLI, BatchNorm+ReLU per stage, optional extra same-resolution
conv stages, global average pool -> Dropout -> Linear) but as ONE
variable-length stack instead of train_cnn.py's fixed lum/stride2/
post_concat shape -- that shape exists specifically to fuse the DCT
model's several separate coefficient-plane inputs, which doesn't apply
here (RGB pixels are already one plain tensor). See
dct_common/models/rgb_cnn.py's module docstring for the full design note.

Examples:
    python train_rgb_cnn.py
    python train_rgb_cnn.py --downsample-factor 4
    python train_rgb_cnn.py --conv-channels 16,32,64,128 --extra-conv-channels ""
    python train_rgb_cnn.py --classes car,garden,people,seat --epochs 15

See README.md for what the printed sections mean (same shape as
train_cnn.py's).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dct_common.augmentation import augment_image
from dct_common.config import CLASS_NAMES, Config
from dct_common.dataset import ensure_dataset
from dct_common.device import benchmark_cpu_inference, get_device
from dct_common.metrics import balanced_accuracy, confusion_matrix, print_accuracy_table, print_classification_report
from dct_common.models.rgb_cnn import (
    Int8RgbCnnReference,
    RgbCnnClassifier,
    RgbCnnClassifierQat,
    init_rgb_cnn_qat_from_float,
)
from dct_common.quantization import ragged_object_array
from dct_common.seeding import set_seed
from dct_common.splits import build_rgb_split, build_rgb_block_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "output" / "rgb_cnn"   # overridden by --artifacts-name in main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-width", type=int, default=160, help="data/'s build resolution, multiple of 16 (default: 160)")
    parser.add_argument("--capture-height", type=int, default=120, help="data/'s build resolution (default: 120)")
    parser.add_argument("--chroma-subsampling", type=str, default="4:2:2", choices=["4:2:0", "4:2:2"],
                         help="only affects ensure_dataset()'s data/ build/reuse check, irrelevant to this "
                              "model's own RGB decode (default: 4:2:2)")
    parser.add_argument("--dataset-source", type=str, default="everyday_openimages160x120",
                         help="source directory name (relative to the project root) to build data/ from -- "
                              "see train_cnn.py's --dataset-source help for the full list of options")
    parser.add_argument("--rgb-block-width", type=int, default=8,
                        help="average each block of this many pixels horizontally into one "
                             "input value (default: 8). An 8x8 block mean is exactly the DCT "
                             "arm's DC coefficient, so the default makes this the "
                             "equal-resolution pixel control -- 20x15 at 160x120 capture. "
                             "Use 1 (with --rgb-block-height 1) for full-resolution pixels, "
                             "which is what the paper's RGB arm used.")
    parser.add_argument("--rgb-block-height", type=int, default=8,
                        help="as --rgb-block-width, vertically (default: 8)")
    parser.add_argument("--artifacts-name", type=str, default="rgb_cnn",
                        help="subdirectory of output/ to write to (default: rgb_cnn). Use a "
                             "different name to avoid overwriting an existing trained model.")
    parser.add_argument("--downsample-factor", type=int, default=1,
                         help="resize decoded pixels down to (capture_width/factor) x (capture_height/factor) "
                             "before training -- 1 = full capture resolution (default: 1)")
    parser.add_argument("--classes", type=str, default=None, help="comma-separated subset of class names (default: all)")
    parser.add_argument("--conv-channels", type=str, default="16,32,64",
                         help="comma-separated channel widths for the main conv stack -- first stage is stride 1, "
                              "every later stage is stride 2 (downsamples) (default: 16,32,64)")
    parser.add_argument("--extra-conv-channels", type=str, default="32",
                         help="comma-separated extra stride-1 conv stage widths after the main stack, "
                              "e.g. '64,64' (empty string for none) (default: 32)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="data",
                         help="dataset directory to train on, relative to the project root (default: data). "
                              "Point this at data_hand_curated for fine-tuning. Any value other than 'data' "
                              "SKIPS ensure_dataset() -- a curated directory is hand-built and must never be "
                              "regenerated from --dataset-source.")
    parser.add_argument("--test-data-dir", type=str, default=None,
                         help="where to read the test split from (default: --data-dir if it has a test/, "
                              "else data/). Fine-tuning sets have no test/ of their own on purpose: the "
                              "fine-tuned and base models must be scored on the same untouched split for "
                              "the comparison to mean anything.")
    parser.add_argument("--fine-tune-from", type=str, default=None,
                         help="output/ subdirectory holding a trained model to initialize from, e.g. "
                              "rgb_cnn_5x5. Loads its float_model.pt, verifies its manifest matches this "
                              "run's architecture/reduction/class list, and trains at --fine-tune-lr "
                              "instead of from scratch.")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4,
                         help="learning rate for --fine-tune-from (default: 1e-4, ~10x below the from-scratch "
                              "rate). QAT still runs afterwards at its own rate.")
    parser.add_argument("--use-augmentation", action="store_true",
                         help="live train-split augmentation (flip/crop/rotate/brightness/contrast/blur, see "
                              "dct_common/augmentation.py), applied fresh every epoch directly to decoded pixels "
                              "-- unlike the DCT scripts' offline --use-augmentation (which has to re-encode to a "
                              "real JPEG and re-extract coefficients), this model reads pixels already, so no "
                              "re-encode step is needed and augmentation can vary every epoch instead of being "
                              "fixed at feature-extraction time. Off by default; val/test are never augmented either way.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Reuses Config only for what's genuinely shared (dataset build knobs,
    training hyperparameters, dropout, classes) -- the RGB-specific conv
    architecture (conv_channels/extra_conv_channels) is passed to the
    model directly instead of through Config, since those aren't
    meaningful to the DCT models Config also serves; every DCT-specific
    field here (num_ac_coeffs etc.) just sits at its default, unused."""
    selected = tuple(c.strip() for c in args.classes.split(",")) if args.classes else None
    return Config(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        chroma_subsampling=args.chroma_subsampling,
        selected_classes=selected,
        epochs=args.epochs,
        dropout=args.dropout,
        seed=args.seed,
    )


class RgbImageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 1234) -> None:
        # X is already centered pixel values (-128..127), exactly int8-
        # representable -- no separate quantize-then-dequantize float
        # conversion needed (unlike the DCT models), see rgb_features.py.
        self.x = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        # One shared, instance-level Random rather than reseeding per
        # __getitem__ call -- reseeding per call would make every epoch's
        # augmentation IDENTICAL (same idx -> same seed -> same draw),
        # defeating the point of live per-epoch augmentation.
        self.rng = random.Random(seed) if augment else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), int(self.y[idx])

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """x is (3, H, W), centered [-128, 127]. Round-trips through PIL
        to reuse dct_common.augmentation.augment_image directly (same
        flip/crop/rotate/brightness/contrast/blur ops the DCT models'
        offline augmentation path uses) rather than reimplementing them
        as raw tensor math."""
        uint8_img = (x.transpose(1, 2, 0) + 128).astype(np.uint8)  # (H, W, 3), 0..255
        pil_img = Image.fromarray(uint8_img, mode="RGB")
        augmented = augment_image(pil_img, self.rng)
        arr = np.asarray(augmented, dtype=np.float32)  # (H, W, 3), 0..255
        return arr.transpose(2, 0, 1) - 128.0  # (3, H, W), centered to [-128, 127]


def batched_int8_predict(int8_ref: Int8RgbCnnReference, q_rgb: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Chunks calls to int8_ref.predict() instead of passing a whole split
    at once. Necessary (not just a nice-to-have) for this model
    specifically: conv2d_int8_reference (quantization.py) builds one
    im2col buffer sized for its *entire* input batch with no internal
    chunking -- fine for the DCT models' 15x20 block grids, but at this
    model's full 160x120 pixel resolution, the first conv layer's im2col
    buffer for an ~13,000-image split is ~54GB (13000 * 3*3*3 * 120*160 *
    8 bytes) -- confirmed as the actual cause of two silent, traceback-
    free process kills (OOM, not a VSCode/session issue as first
    suspected) before this fix. Not touching conv2d_int8_reference itself
    since every DCT model's usage of it is safe as-is; the fix belongs at
    this model's call sites instead."""
    preds = []
    for i in range(0, len(q_rgb), batch_size):
        preds.append(int8_ref.predict(q_rgb[i:i + batch_size]))
    return np.concatenate(preds)


def batched_int8_accuracy(int8_ref: Int8RgbCnnReference, q_rgb: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    return float((batched_int8_predict(int8_ref, q_rgb, batch_size) == y).mean())


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(yb.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float((preds == labels).mean()),
        "preds": preds,
        "labels": labels,
    }


def verify_finetune_compatible(manifest: dict, args, conv_channels: tuple, extra_conv_channels: tuple,
                                cfg: Config) -> None:
    """Refuse to fine-tune across a configuration change.

    Everything checked here would otherwise fail either silently or far
    downstream. A different class ORDER is the nastiest: the state dict
    loads cleanly, training converges, and the exported header's labels
    are simply wrong. A different reduction or architecture at least
    fails loudly on the state-dict load, but the message is unreadable."""
    problems = []

    def check(label, want, got):
        if want != got:
            problems.append(f"  {label}: source run has {want!r}, this run has {got!r}")

    check("conv_channels", list(manifest["conv_channels"]), list(conv_channels))
    check("extra_conv_channels", list(manifest["extra_conv_channels"]), list(extra_conv_channels))
    check("num_classes", manifest["num_classes"], cfg.num_classes)
    # Order matters, not just membership -- the final Linear's rows are
    # positional and the exported MODEL_CLASS_NAMES follows this list.
    check("class_names (order matters)", list(manifest["class_names"]), list(cfg.active_class_names))
    check("rgb_width", manifest["rgb_width"], args.capture_width // args.rgb_block_width
          if args.rgb_block_width > 1 or args.rgb_block_height > 1
          else args.capture_width // args.downsample_factor)
    check("rgb_height", manifest["rgb_height"], args.capture_height // args.rgb_block_height
          if args.rgb_block_width > 1 or args.rgb_block_height > 1
          else args.capture_height // args.downsample_factor)
    if "rgb_block_width" in manifest:
        check("rgb_block_width", manifest["rgb_block_width"], args.rgb_block_width)
        check("rgb_block_height", manifest["rgb_block_height"], args.rgb_block_height)

    if problems:
        raise SystemExit(
            f"--fine-tune-from {args.fine_tune_from}: the source run's configuration does not match "
            f"this one.\n" + "\n".join(problems) +
            "\n\nFine-tuning only means anything when the model is otherwise identical. "
            "Match the flags to the source run, or train from scratch.")


def load_finetune_init(args, conv_channels: tuple, extra_conv_channels: tuple, cfg: Config) -> dict:
    """Read and validate the source run, returning its float state dict."""
    src_dir = Path(__file__).resolve().parent / "output" / args.fine_tune_from
    manifest_path = src_dir / "model_manifest.json"
    weights_path = src_dir / "float_model.pt"
    for path in (manifest_path, weights_path):
        if not path.exists():
            raise SystemExit(f"--fine-tune-from {args.fine_tune_from}: {path} not found")
    with open(manifest_path) as f:
        manifest = json.load(f)
    verify_finetune_compatible(manifest, args, conv_channels, extra_conv_channels, cfg)
    print(f"  initializing from {src_dir.relative_to(Path(__file__).resolve().parent)}/float_model.pt")
    print(f"    source run: float={manifest['float_test_accuracy']:.4f} "
          f"qat={manifest['qat_test_accuracy']:.4f} int8={manifest['int8_reference_test_accuracy']:.4f}")
    return torch.load(weights_path, map_location="cpu")


def train_float_model(cfg: Config, conv_channels: tuple, extra_conv_channels: tuple, num_classes: int,
                       device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                       init_state: dict | None = None, lr: float | None = None) -> nn.Module:
    set_seed(cfg.seed)
    model = RgbCnnClassifier(conv_channels, extra_conv_channels, cfg.dropout, num_classes).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    lr = cfg.learning_rate if lr is None else lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    if init_state is not None:
        # Score the starting point, so "did fine-tuning help?" is answerable
        # from this log alone rather than by comparing two separate runs.
        start = evaluate(model, val_loader, criterion)
        print(f"  starting val_acc={start['accuracy']:.4f} (before any fine-tuning steps)")
        best_val_acc = start["accuracy"]
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  best={best_val_acc:.4f}")
        if epochs_without_improvement >= cfg.early_stop_patience:
            print(f"  early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


def train_qat_model(
    float_model: nn.Module, cfg: Config, conv_channels: tuple, extra_conv_channels: tuple, num_classes: int,
    device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
    qat_epochs: int, qat_lr: float = 1e-4,
) -> nn.Module:
    set_seed(cfg.seed)
    qat_model = RgbCnnClassifierQat(conv_channels, extra_conv_channels, num_classes).to(device)
    init_rgb_cnn_qat_from_float(qat_model, float_model)

    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=qat_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_val_acc, best_state = -1.0, None

    for epoch in range(qat_epochs):
        qat_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(qat_model(xb), yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(qat_model, val_loader, criterion)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in qat_model.state_dict().items()}

        if epoch % 2 == 0 or epoch == qat_epochs - 1:
            print(f"  [QAT] epoch {epoch:3d}  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  best={best_val_acc:.4f}")

    qat_model.load_state_dict(best_state)
    return qat_model


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    global ARTIFACTS_DIR
    ARTIFACTS_DIR = Path(__file__).resolve().parent / "output" / args.artifacts_name
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    conv_channels = tuple(int(c) for c in args.conv_channels.split(",") if c.strip())
    extra_conv_channels = tuple(int(c) for c in args.extra_conv_channels.split(",") if c.strip()) if args.extra_conv_channels else ()
    # Block averaging supersedes --downsample-factor when either block dimension
    # is above 1. The two are different reductions -- a box mean over a fixed
    # block versus a LANCZOS resize -- and combining them would be meaningless.
    use_blocks = args.rgb_block_width > 1 or args.rgb_block_height > 1
    if use_blocks:
        if args.downsample_factor != 1:
            raise SystemExit("--downsample-factor cannot be combined with --rgb-block-*; "
                             "they are alternative reductions. Use one or the other.")
        rgb_width = args.capture_width // args.rgb_block_width
        rgb_height = args.capture_height // args.rgb_block_height
    else:
        rgb_width = args.capture_width // args.downsample_factor
        rgb_height = args.capture_height // args.downsample_factor

    print("=" * 70)
    print("RGB-PIXEL CNN (pixel-domain arm of the DCT-vs-RGB comparison)")
    print("=" * 70)
    print(cfg)
    print(f"conv_channels={conv_channels}  extra_conv_channels={extra_conv_channels}")
    print(f"capture: {args.capture_width}x{args.capture_height}  downsample_factor={args.downsample_factor}  "
          f"-> model input {rgb_width}x{rgb_height}")
    print(f"classes ({cfg.num_classes}/{len(CLASS_NAMES)}): {cfg.active_class_names}")
    set_seed(cfg.seed)
    device = get_device()

    print("\n-- Dataset --")
    train_data_dir = PROJECT_ROOT / args.data_dir
    if args.data_dir == "data":
        ensure_dataset(train_data_dir, cfg, source_dir=PROJECT_ROOT / args.dataset_source)
    else:
        # A curated directory is assembled by hand (curation_pull.py ->
        # curation_resolve.py). ensure_dataset() would see a class/size
        # mismatch and rebuild it from --dataset-source, silently throwing
        # the curation away and replacing it with the uncurated data.
        if not train_data_dir.is_dir():
            raise SystemExit(f"--data-dir {args.data_dir}: {train_data_dir} not found")
        print(f"  using {train_data_dir} as-is (ensure_dataset skipped -- hand-built directory)")

    # The test split comes from wherever it is untouched. A fine-tuning set
    # has no test/ of its own by design: scoring the fine-tuned model on a
    # different test split than the base model would make the comparison
    # meaningless.
    if args.test_data_dir is not None:
        test_data_dir = PROJECT_ROOT / args.test_data_dir
    elif (train_data_dir / "test").is_dir():
        test_data_dir = train_data_dir
    else:
        test_data_dir = DATA_DIR
    if test_data_dir != train_data_dir:
        print(f"  test split read from {test_data_dir} (--data-dir has no test/ of its own)")
    if not (test_data_dir / "test").is_dir():
        raise SystemExit(f"no test split found at {test_data_dir / 'test'}")

    print("\n-- Feature extraction --")
    def load_split(dd, split):
        return (build_rgb_block_split(dd, split, cfg.active_class_names, args.rgb_block_width, args.rgb_block_height)
                if use_blocks else
                build_rgb_split(dd, split, cfg.active_class_names, rgb_width, rgb_height))
    X_train, y_train, _ = load_split(train_data_dir, "train")
    X_val, y_val, _ = load_split(train_data_dir, "val")
    X_test, y_test, _ = load_split(test_data_dir, "test")
    for name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        print(f"  {name}: X={X.shape} y={y.shape} class balance={np.bincount(y)}")

    print("\n-- Input quantization --")
    # No calibration step, unlike the DCT models -- centered pixel values
    # (-128..127) are already exactly int8-representable by construction
    # (see rgb_features.py), so "quantize then dequantize" is the identity
    # here. Just cast straight to float32 for training.
    print("  fixed scale=1.0 (exact, no percentile calibration needed -- see rgb_features.py)")

    print(f"  augmentation: {'on (live, per epoch, train split only)' if args.use_augmentation else 'off'}")
    train_ds = RgbImageDataset(X_train, y_train, augment=args.use_augmentation, seed=cfg.seed)
    val_ds = RgbImageDataset(X_val, y_val)
    test_ds = RgbImageDataset(X_test, y_test)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    init_state = None
    if args.fine_tune_from:
        print("\n-- Fine-tune initialization --")
        init_state = load_finetune_init(args, conv_channels, extra_conv_channels, cfg)

    print("\n-- Float training --" if not args.fine_tune_from
          else f"\n-- Float fine-tuning (lr={args.fine_tune_lr:g}) --")
    float_model = train_float_model(cfg, conv_channels, extra_conv_channels, cfg.num_classes, device,
                                     train_loader, val_loader, init_state=init_state,
                                     lr=args.fine_tune_lr if args.fine_tune_from else None)
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    test_metrics = evaluate(float_model, test_loader, eval_criterion)
    print(f"  float model test accuracy: {test_metrics['accuracy']:.4f}")
    print_classification_report(test_metrics["preds"], test_metrics["labels"], cfg.active_class_names)
    torch.save(float_model.state_dict(), ARTIFACTS_DIR / "float_model.pt")

    print("\n-- QAT fine-tuning --")
    qat_model = train_qat_model(float_model, cfg, conv_channels, extra_conv_channels, cfg.num_classes,
                                 device, train_loader, val_loader, args.qat_epochs)
    qat_test_metrics = evaluate(qat_model, test_loader, eval_criterion)
    print(f"  QAT model test accuracy: {qat_test_metrics['accuracy']:.4f}  (float was {test_metrics['accuracy']:.4f})")
    print_classification_report(qat_test_metrics["preds"], qat_test_metrics["labels"], cfg.active_class_names)

    float_model = float_model.to("cpu")
    qat_model = qat_model.to("cpu")

    print("\n-- CPU inference latency (float model) --")
    sample_inputs = (torch.from_numpy(X_test[0:1].astype(np.float32)),)
    cpu_timing = benchmark_cpu_inference(float_model, sample_inputs)
    print(f"  {cpu_timing['ms_per_inference']:.3f} ms/image  ({cpu_timing['fps']:.1f} fps)")

    print("\n-- Bit-exact int8 NumPy reference --")
    int8_ref = Int8RgbCnnReference(qat_model, train_loader, percentile=cfg.calibration_percentile)
    Q_train, Q_val, Q_test = X_train.astype(np.int8), X_val.astype(np.int8), X_test.astype(np.int8)
    int8_test_accuracy = batched_int8_accuracy(int8_ref, Q_test, y_test)
    int8_test_preds = batched_int8_predict(int8_ref, Q_test)
    agreement = float((int8_test_preds == qat_test_metrics["preds"]).mean())
    print(f"  int8 reference test accuracy: {int8_test_accuracy:.4f}")
    print(f"  agreement (int8 vs QAT float-sim): {agreement:.4f}")

    print("\n-- Saving artifacts --")
    np.savez(
        ARTIFACTS_DIR / "quantized_model.npz",
        conv_weights=ragged_object_array([w for w, _, _, _, _ in int8_ref.conv_layers_q]),
        conv_biases=ragged_object_array([b for _, b, _, _, _ in int8_ref.conv_layers_q]),
        conv_mults=ragged_object_array([m for _, _, m, _, _ in int8_ref.conv_layers_q]),
        conv_shifts=ragged_object_array([s for _, _, _, s, _ in int8_ref.conv_layers_q]),
        conv_strides=np.array([st for _, _, _, _, st in int8_ref.conv_layers_q], dtype=np.int32),
        output_weight=int8_ref.w_out_q, output_bias=int8_ref.b_out_q, output_mult=int8_ref.mult_out, output_shift=int8_ref.shift_out,
    )
    manifest = {
        "format_version": 1,
        "capture_width": cfg.capture_width, "capture_height": cfg.capture_height,
        "downsample_factor": args.downsample_factor, "rgb_width": rgb_width, "rgb_height": rgb_height,
        "rgb_block_width": args.rgb_block_width, "rgb_block_height": args.rgb_block_height,
        "reduction": "block_mean" if use_blocks else "lanczos",
        "conv_channels": list(conv_channels), "extra_conv_channels": list(extra_conv_channels),
        "num_classes": cfg.num_classes, "class_names": cfg.active_class_names,
        "output_scale": int8_ref.output_scale,
        "float_test_accuracy": test_metrics["accuracy"],
        "qat_test_accuracy": qat_test_metrics["accuracy"],
        "int8_reference_test_accuracy": int8_test_accuracy,
        "int8_vs_qat_agreement": agreement,
        "cpu_inference_ms": cpu_timing["ms_per_inference"],
        # Provenance. Recorded because "which data was this trained on, and
        # was it fine-tuned?" is otherwise unanswerable from the artifacts --
        # the same gap that made the shipped 5x5 model's epoch count and
        # augmentation setting unrecoverable. argv is the whole invocation,
        # so nothing else here has to be kept in sync by hand.
        "data_dir": args.data_dir,
        "test_data_dir": str(test_data_dir.relative_to(PROJECT_ROOT)),
        "fine_tuned_from": args.fine_tune_from,
        "fine_tune_lr": args.fine_tune_lr if args.fine_tune_from else None,
        "epochs": cfg.epochs,
        "qat_epochs": args.qat_epochs,
        "use_augmentation": args.use_augmentation,
        "seed": cfg.seed,
        "argv": sys.argv,
    }
    with open(ARTIFACTS_DIR / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    torch.save(qat_model.state_dict(), ARTIFACTS_DIR / "qat_model.pt")
    print(f"  saved to {ARTIFACTS_DIR}  (no C export for this model -- see module docstring)")
    for p in sorted(ARTIFACTS_DIR.glob("*")):
        if p.is_file():
            print("   ", p.name)

    print("\n" + "=" * 70)
    print("ACCURACY SUMMARY (RGB-pixel CNN)")
    print("=" * 70)
    train_float = evaluate(float_model, train_loader, eval_criterion)
    train_qat = evaluate(qat_model, train_loader, eval_criterion)
    train_int8_preds = batched_int8_predict(int8_ref, Q_train)
    val_float = evaluate(float_model, val_loader, eval_criterion)
    val_qat = evaluate(qat_model, val_loader, eval_criterion)
    val_int8_preds = batched_int8_predict(int8_ref, Q_val)

    accuracy_table = {
        "train": {"float": train_float["accuracy"], "qat": train_qat["accuracy"], "int8": float((train_int8_preds == y_train).mean())},
        "val": {"float": val_float["accuracy"], "qat": val_qat["accuracy"], "int8": float((val_int8_preds == y_val).mean())},
        "test": {"float": test_metrics["accuracy"], "qat": qat_test_metrics["accuracy"], "int8": int8_test_accuracy},
    }
    print_accuracy_table(accuracy_table)

    balanced_table = {
        "train": {"float": balanced_accuracy(train_float["preds"], train_float["labels"], cfg.num_classes),
                   "qat": balanced_accuracy(train_qat["preds"], train_qat["labels"], cfg.num_classes),
                   "int8": balanced_accuracy(train_int8_preds, y_train, cfg.num_classes)},
        "val": {"float": balanced_accuracy(val_float["preds"], val_float["labels"], cfg.num_classes),
                 "qat": balanced_accuracy(val_qat["preds"], val_qat["labels"], cfg.num_classes),
                 "int8": balanced_accuracy(val_int8_preds, y_val, cfg.num_classes)},
        "test": {"float": balanced_accuracy(test_metrics["preds"], test_metrics["labels"], cfg.num_classes),
                  "qat": balanced_accuracy(qat_test_metrics["preds"], qat_test_metrics["labels"], cfg.num_classes),
                  "int8": balanced_accuracy(int8_test_preds, y_test, cfg.num_classes)},
    }
    print("\nBALANCED ACCURACY (macro-avg per-class recall -- every class weighted equally)")
    print_accuracy_table(balanced_table)

    with open(ARTIFACTS_DIR / "accuracy_table.json", "w") as f:
        json.dump({**accuracy_table, "balanced": balanced_table}, f, indent=2)

    confusion_matrices = {
        "class_names": cfg.active_class_names,
        "train": {"float": confusion_matrix(train_float["preds"], train_float["labels"], cfg.num_classes).tolist(),
                   "qat": confusion_matrix(train_qat["preds"], train_qat["labels"], cfg.num_classes).tolist(),
                   "int8": confusion_matrix(train_int8_preds, y_train, cfg.num_classes).tolist()},
        "val": {"float": confusion_matrix(val_float["preds"], val_float["labels"], cfg.num_classes).tolist(),
                 "qat": confusion_matrix(val_qat["preds"], val_qat["labels"], cfg.num_classes).tolist(),
                 "int8": confusion_matrix(val_int8_preds, y_val, cfg.num_classes).tolist()},
        "test": {"float": confusion_matrix(test_metrics["preds"], test_metrics["labels"], cfg.num_classes).tolist(),
                  "qat": confusion_matrix(qat_test_metrics["preds"], qat_test_metrics["labels"], cfg.num_classes).tolist(),
                  "int8": confusion_matrix(int8_test_preds, y_test, cfg.num_classes).tolist()},
    }
    with open(ARTIFACTS_DIR / "confusion_matrix.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)
    print(f"  wrote {ARTIFACTS_DIR / 'confusion_matrix.json'}")


if __name__ == "__main__":
    main()
