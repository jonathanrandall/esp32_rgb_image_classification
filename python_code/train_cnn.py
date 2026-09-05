#!/usr/bin/env python3
"""Train the spatial CNN over JPEG-DCT coefficient planes (Y + chroma fusion):
float -> QAT -> bit-exact int8.

Examples:
    python train_cnn.py
    python train_cnn.py --epochs 30 --num-ac-coeffs 8
    python train_cnn.py --classes cats,dogs,birds
    python train_cnn.py --no-chroma
    python train_cnn.py --capture-width 160 --capture-height 120 --chroma-subsampling 4:2:2 --dataset-source openimages_160x120 --no-chroma

See README.md for what the printed sections mean and how to read the final
accuracy table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dct_common import config as config_module
from dct_common.config import CLASS_NAMES, Config
from dct_common.dataset import ensure_dataset
from dct_common.device import benchmark_cpu_inference, get_device
from dct_common.metrics import balanced_accuracy, confusion_matrix, print_accuracy_table, print_classification_report
from dct_common.models.cnn import DctCnnClassifier, DctCnnClassifierQat, Int8CnnReference, init_cnn_qat_from_float
from dct_common.quantization import calibrate_channel_scales, quantize_planes, ragged_object_array
from dct_common.seeding import set_seed
from dct_common.splits import build_spatial_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "output" / "cnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-width", type=int, default=160, help="multiple of 16 (default: 160)")
    parser.add_argument("--capture-height", type=int, default=120, help="multiple of 16 under 4:2:0, multiple of 8 under 4:2:2 (default: 120)")
    parser.add_argument("--chroma-subsampling", type=str, default="4:2:2", choices=["4:2:0", "4:2:2"],
                         help="JPEG chroma subsampling used to build/re-encode data/ -- 4:2:0 (original assumption) or "
                              "4:2:2 (matches real OV2640 camera output, see issues.md) (default: 4:2:2)")
    parser.add_argument("--dataset-source", type=str, default="everyday_openimages160x120",
                         help="source directory name (relative to the project root, one level up from python_code/) "
                              "to build data/ from -- everyday_openimages160x120 is the current 23-class dataset "
                              "(see its own README.md for provenance), everyday_openimages160x120_filtered is the "
                              "same data with people/chair intersections dropped (see filter_intersections.py), "
                              "openimages_160x120 is the older 6-class dataset (incompatible with the current "
                              "CLASS_NAMES in dct_common/config.py -- would need CLASS_NAMES reverted to use it), "
                              "data_pretraining is the earlier COCO-based pull kept for a possible pretrain/fine-tune "
                              "experiment (default: everyday_openimages160x120)")
    parser.add_argument("--num-ac-coeffs", type=int, default=3, help="Y (luma) AC coefficients kept per block on top of DC (default: 3)")
    parser.add_argument("--num-chroma-ac-coeffs", type=int, default=None,
                         help="Cb/Cr (chroma) AC coefficients kept per block on top of DC, independent of "
                              "--num-ac-coeffs -- e.g. 0 for chroma DC-only. Defaults to matching "
                              "--num-ac-coeffs (only meaningful with chroma enabled, i.e. without --no-chroma).")
    parser.add_argument("--data-dir", type=str, default="data",
                         help="dataset directory to train on, relative to the project root (default: data). Any "
                              "value other than 'data' SKIPS ensure_dataset(), so a deliberately-built directory "
                              "(e.g. a --split-by-people build) is never silently regenerated from --dataset-source.")
    parser.add_argument("--classes", type=str, default=None,
                         help="comma-separated subset of class names (default: all 23)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-chroma", action="store_true", help="disable Cb/Cr fusion (Y-only, for comparison)")
    parser.add_argument("--lum-channels", type=int, default=16, help="lum_conv output channels (default: 16)")
    parser.add_argument("--stride2-channels", type=int, default=32, help="stride2_conv output channels (default: 32)")
    parser.add_argument("--post-concat-channels", type=int, default=64, help="post_concat_conv output channels (default: 64)")
    parser.add_argument("--extra-conv-channels", type=str, default="32", help="comma-separated extra conv stage widths, e.g. '64,64' (empty string for none)")
    parser.add_argument("--coeff-scan-order", type=str, default="zigzag", choices=["zigzag", "axis_first"],
                         help="block coefficient traversal order: standard JPEG zig-zag, or pure horizontal/vertical frequencies first (default: zigzag)")
    parser.add_argument("--use-augmentation", action="store_true",
                         help="offline train-split augmentation (flip/crop/rotate/brightness/contrast/blur, see "
                              "dct_common/augmentation.py) -- each train image is decoded, augmented, and "
                              "re-encoded to a real JPEG before its DCT coefficients are extracted, since this "
                              "project reads coefficients straight from the JPEG bitstream rather than computing "
                              "them itself. Off by default; val/test are never augmented either way.")
    parser.add_argument("--augment-copies", type=int, default=1,
                         help="extra augmented variants generated per train image when --use-augmentation is passed (default: 1)")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    # Config validates selected_classes against dct_common.config.CLASS_NAMES,
    # which is loaded from class_names.json -- and that file describes data/.
    # An alternative --data-dir (a --split-by-people build, a curated set) has
    # its own class list, so take the truth from the directory itself rather
    # than from a sidecar written for a different dataset. Rebinding the module
    # global is what the validation actually reads; the local `from ... import
    # CLASS_NAMES` binding above is only used for the all-classes default.
    if args.data_dir != "data":
        d = PROJECT_ROOT / args.data_dir / "train"
        if not d.is_dir():
            raise SystemExit(f"--data-dir {args.data_dir}: {d} not found")
        discovered = sorted(p.name for p in d.iterdir() if p.is_dir())
        config_module.CLASS_NAMES = discovered
        print(f"  class list taken from {args.data_dir}/train ({len(discovered)}): {discovered}")

    selected = tuple(c.strip() for c in args.classes.split(",")) if args.classes else None
    extra = tuple(int(c) for c in args.extra_conv_channels.split(",") if c.strip()) if args.extra_conv_channels else ()
    return Config(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        chroma_subsampling=args.chroma_subsampling,
        num_ac_coeffs=args.num_ac_coeffs,
        num_chroma_ac_coeffs=args.num_chroma_ac_coeffs,
        selected_classes=selected,
        epochs=args.epochs,
        dropout=args.dropout,
        use_chroma=not args.no_chroma,
        lum_channels=args.lum_channels,
        stride2_channels=args.stride2_channels,
        post_concat_channels=args.post_concat_channels,
        extra_conv_channels=extra,
        coeff_scan_order=args.coeff_scan_order,
        seed=args.seed,
    )


class DctPlaneDataset(Dataset):
    def __init__(self, X: np.ndarray, Xc: np.ndarray, y: np.ndarray) -> None:
        self.x = X.astype(np.float32)
        self.xc = Xc.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.xc[idx]), int(self.y[idx])


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, xcb, yb in loader:
            xb, xcb, yb = xb.to(device, non_blocking=True), xcb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb, xcb)
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


def train_cnn_model(cfg: Config, device: torch.device, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    set_seed(cfg.seed)
    model = DctCnnClassifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, xcb, yb in train_loader:
            xb, xcb, yb = xb.to(device, non_blocking=True), xcb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb, xcb), yb)
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


def train_cnn_qat_model(
    float_model: nn.Module, cfg: Config, device: torch.device,
    train_loader: DataLoader, val_loader: DataLoader,
    qat_epochs: int, qat_lr: float = 1e-4,
) -> nn.Module:
    set_seed(cfg.seed)
    qat_model = DctCnnClassifierQat(cfg).to(device)
    init_cnn_qat_from_float(qat_model, float_model)

    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=qat_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_val_acc, best_state = -1.0, None

    for epoch in range(qat_epochs):
        qat_model.train()
        for xb, xcb, yb in train_loader:
            xb, xcb, yb = xb.to(device, non_blocking=True), xcb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(qat_model(xb, xcb), yb)
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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JPEG-DCT CNN")
    print("=" * 70)
    print(cfg)
    print(f"Y block grid: {cfg.y_rows}x{cfg.y_cols}  Cb=Cr block grid: {cfg.c_rows}x{cfg.c_cols}  "
          f"num_coeffs={cfg.num_coeffs} (DC + {cfg.num_ac_coeffs} AC)  "
          f"num_chroma_coeffs={cfg.num_chroma_coeffs} (DC + {cfg.num_chroma_ac_coeffs} AC)")
    print(f"classes ({cfg.num_classes}/{len(CLASS_NAMES)}): {cfg.active_class_names}")
    set_seed(cfg.seed)
    device = get_device()

    print("\n-- Dataset --")
    data_dir = PROJECT_ROOT / args.data_dir
    if args.data_dir == "data":
        ensure_dataset(data_dir, cfg, source_dir=PROJECT_ROOT / args.dataset_source)
    else:
        # Same reasoning as train_rgb_cnn.py's --data-dir: an alternative
        # dataset (a people-split build, a hand-curated set) is assembled
        # deliberately, and ensure_dataset() would see a class-list mismatch
        # and silently rebuild it from --dataset-source, throwing that away.
        if not data_dir.is_dir():
            raise SystemExit(f"--data-dir {args.data_dir}: {data_dir} not found")
        print(f"  using {data_dir} as-is (ensure_dataset skipped -- alternative dataset)")

    print("\n-- Feature extraction --")
    X_train, Xc_train, y_train, _ = build_spatial_split(data_dir, "train", cfg.active_class_names, cfg,
                                                         augment=args.use_augmentation, augment_copies=args.augment_copies)
    X_val, Xc_val, y_val, _ = build_spatial_split(data_dir, "val", cfg.active_class_names, cfg)
    X_test, Xc_test, y_test, _ = build_spatial_split(data_dir, "test", cfg.active_class_names, cfg)
    for name, X, Xc, y in (("train", X_train, Xc_train, y_train), ("val", X_val, Xc_val, y_val), ("test", X_test, Xc_test, y_test)):
        print(f"  {name}: Y={X.shape} chroma={Xc.shape} y={y.shape} class balance={np.bincount(y)}")

    print("\n-- Input quantization calibration --")
    # Same idea as the MLP's input calibration (percentile-of-|value|,
    # symmetric int8), just per spatial-plane channel instead of per flat
    # feature: one scale for the DC plane, one per Y AC coefficient
    # position, one per Cb/Cr coefficient position. Calibrated off train
    # only. The float/QAT models then train on the quantized-then-
    # dequantized values (`Q * scale`), not raw floats -- exactly the MLP's
    # "train on what export will reconstruct" trick (see models/mlp.py),
    # so folding these scales into lum_conv/post_concat_conv's weights at
    # export time is an exact identity, not an approximation.
    dc_scales = calibrate_channel_scales(X_train[:, 0:1], cfg.calibration_percentile)
    ac_scales = calibrate_channel_scales(X_train[:, 1:], cfg.calibration_percentile) if cfg.num_ac_coeffs > 0 else np.zeros(0)
    chroma_scales = calibrate_channel_scales(Xc_train, cfg.calibration_percentile)
    print(f"  dc_scale={dc_scales[0]:.4f}")
    if cfg.num_ac_coeffs > 0:
        print(f"  ac_scales={np.array2string(ac_scales, precision=4)}")
    print(f"  chroma_scales={np.array2string(chroma_scales, precision=4)}")

    def quantize_and_dequantize(X: np.ndarray, Xc: np.ndarray) -> tuple:
        q_dc = quantize_planes(X[:, 0:1], dc_scales)
        q_ac = quantize_planes(X[:, 1:], ac_scales) if cfg.num_ac_coeffs > 0 else np.zeros((X.shape[0], 0, cfg.y_rows, cfg.y_cols), dtype=np.int8)
        q_chroma = quantize_planes(Xc, chroma_scales)

        x_dq = np.concatenate([
            q_dc.astype(np.float32) * dc_scales.astype(np.float32)[None, :, None, None],
            q_ac.astype(np.float32) * (ac_scales.astype(np.float32)[None, :, None, None] if cfg.num_ac_coeffs > 0 else 1.0),
        ], axis=1)
        xc_dq = q_chroma.astype(np.float32) * chroma_scales.astype(np.float32)[None, :, None, None]
        return q_dc, q_ac, q_chroma, x_dq, xc_dq

    Q_dc_train, Q_ac_train, Q_chroma_train, X_train_dq, Xc_train_dq = quantize_and_dequantize(X_train, Xc_train)
    Q_dc_val, Q_ac_val, Q_chroma_val, X_val_dq, Xc_val_dq = quantize_and_dequantize(X_val, Xc_val)
    Q_dc_test, Q_ac_test, Q_chroma_test, X_test_dq, Xc_test_dq = quantize_and_dequantize(X_test, Xc_test)

    train_ds = DctPlaneDataset(X_train_dq, Xc_train_dq, y_train)
    val_ds = DctPlaneDataset(X_val_dq, Xc_val_dq, y_val)
    test_ds = DctPlaneDataset(X_test_dq, Xc_test_dq, y_test)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print("\n-- Float training --")
    float_model = train_cnn_model(cfg, device, train_loader, val_loader)
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    test_metrics = evaluate(float_model, test_loader, eval_criterion)
    print(f"  float model test accuracy: {test_metrics['accuracy']:.4f}")
    print_classification_report(test_metrics["preds"], test_metrics["labels"], cfg.active_class_names)
    torch.save(float_model.state_dict(), ARTIFACTS_DIR / "float_model.pt")

    print("\n-- QAT fine-tuning --")
    qat_model = train_cnn_qat_model(float_model, cfg, device, train_loader, val_loader, args.qat_epochs)
    qat_test_metrics = evaluate(qat_model, test_loader, eval_criterion)
    print(f"  QAT model test accuracy: {qat_test_metrics['accuracy']:.4f}  (float was {test_metrics['accuracy']:.4f})")
    print_classification_report(qat_test_metrics["preds"], qat_test_metrics["labels"], cfg.active_class_names)

    # Everything from here on (bit-exact int8 reference, weight export, the
    # accuracy summary) is CPU/NumPy work by design -- it's what actually
    # runs on the ESP32, not a torch/CUDA graph -- so both models move to
    # CPU now and stay there for the rest of the script.
    float_model = float_model.to("cpu")
    qat_model = qat_model.to("cpu")

    print("\n-- CPU inference latency (float model) --")
    sample_inputs = (
        torch.from_numpy(X_test_dq[0:1]),
        torch.from_numpy(Xc_test_dq[0:1]),
    )
    cpu_timing = benchmark_cpu_inference(float_model, sample_inputs)
    print(f"  {cpu_timing['ms_per_inference']:.3f} ms/image  ({cpu_timing['fps']:.1f} fps)")

    print("\n-- Bit-exact int8 NumPy reference --")
    int8_ref = Int8CnnReference(qat_model, float(dc_scales[0]), ac_scales, chroma_scales, cfg, train_loader)
    int8_test_accuracy = int8_ref.accuracy(Q_dc_test, Q_ac_test, Q_chroma_test, y_test)
    int8_test_preds = int8_ref.predict(Q_dc_test, Q_ac_test, Q_chroma_test)
    agreement = float((int8_test_preds == qat_test_metrics["preds"]).mean())
    print(f"  int8 reference test accuracy: {int8_test_accuracy:.4f}")
    print(f"  agreement (int8 vs QAT float-sim): {agreement:.4f}")

    print("\n-- Saving artifacts --")
    np.savez(
        ARTIFACTS_DIR / "quantized_model.npz",
        lum_weight=int8_ref.w1_q, lum_bias=int8_ref.b1_q, lum_mult=int8_ref.mult1, lum_shift=int8_ref.shift1,
        stride2_weight=int8_ref.w2_q, stride2_bias=int8_ref.b2_q, stride2_mult=int8_ref.mult2, stride2_shift=int8_ref.shift2,
        post_concat_weight=int8_ref.w3_q, post_concat_bias=int8_ref.b3_q, post_concat_mult=int8_ref.mult3, post_concat_shift=int8_ref.shift3,
        extra_weights=ragged_object_array([w for w, _, _, _ in int8_ref.extra_layers_q]),
        extra_biases=ragged_object_array([b for _, b, _, _ in int8_ref.extra_layers_q]),
        extra_mults=ragged_object_array([m for _, _, m, _ in int8_ref.extra_layers_q]),
        extra_shifts=ragged_object_array([s for _, _, _, s in int8_ref.extra_layers_q]),
        output_weight=int8_ref.w_out_q, output_bias=int8_ref.b_out_q, output_mult=int8_ref.mult_out, output_shift=int8_ref.shift_out,
        dc_scale=dc_scales, ac_scales=ac_scales, chroma_scales=chroma_scales,
    )
    manifest = {
        "format_version": 1,
        "capture_width": cfg.capture_width, "capture_height": cfg.capture_height,
        "chroma_subsampling": cfg.chroma_subsampling,
        "y_rows": cfg.y_rows, "y_cols": cfg.y_cols,
        "c_rows": cfg.c_rows, "c_cols": cfg.c_cols,
        "num_ac_coeffs": cfg.num_ac_coeffs, "num_coeffs": cfg.num_coeffs,
        "num_chroma_ac_coeffs": cfg.num_chroma_ac_coeffs, "num_chroma_coeffs": cfg.num_chroma_coeffs,
        "coefficient_order": cfg.coeff_scan_order,
        "use_chroma": cfg.use_chroma,
        "lum_channels": cfg.lum_channels, "stride2_channels": cfg.stride2_channels,
        "post_concat_channels": cfg.post_concat_channels, "extra_conv_channels": list(cfg.extra_conv_channels),
        "num_classes": cfg.num_classes, "class_names": cfg.active_class_names,
        "act_scale_lum": int8_ref.act_scale_lum, "act_scale_stride2": int8_ref.act_scale_stride2,
        "act_scale_postconcat": int8_ref.act_scale_postconcat, "output_scale": int8_ref.output_scale,
        "float_test_accuracy": test_metrics["accuracy"],
        "qat_test_accuracy": qat_test_metrics["accuracy"],
        "int8_reference_test_accuracy": int8_test_accuracy,
        "int8_vs_qat_agreement": agreement,
        "cpu_inference_ms": cpu_timing["ms_per_inference"],
    }
    with open(ARTIFACTS_DIR / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    torch.save(qat_model.state_dict(), ARTIFACTS_DIR / "qat_model.pt")
    print(f"  saved to {ARTIFACTS_DIR}")
    for p in sorted(ARTIFACTS_DIR.glob("*")):
        if p.is_file():
            print("   ", p.name)

    print("\n" + "=" * 70)
    print("ACCURACY SUMMARY (CNN)")
    print("=" * 70)
    train_float = evaluate(float_model, train_loader, eval_criterion)
    train_qat = evaluate(qat_model, train_loader, eval_criterion)
    train_int8_preds = int8_ref.predict(Q_dc_train, Q_ac_train, Q_chroma_train)
    val_float = evaluate(float_model, val_loader, eval_criterion)
    val_qat = evaluate(qat_model, val_loader, eval_criterion)
    val_int8_preds = int8_ref.predict(Q_dc_val, Q_ac_val, Q_chroma_val)

    accuracy_table = {
        "train": {
            "float": train_float["accuracy"],
            "qat": train_qat["accuracy"],
            "int8": int8_ref.accuracy(Q_dc_train, Q_ac_train, Q_chroma_train, y_train),
        },
        "val": {
            "float": val_float["accuracy"],
            "qat": val_qat["accuracy"],
            "int8": int8_ref.accuracy(Q_dc_val, Q_ac_val, Q_chroma_val, y_val),
        },
        "test": {
            "float": test_metrics["accuracy"],
            "qat": qat_test_metrics["accuracy"],
            "int8": int8_test_accuracy,
        },
    }
    print_accuracy_table(accuracy_table)

    # Macro-average per-class recall -- every class weighted equally
    # regardless of test-set size, unlike plain accuracy above (which is
    # dominated by whichever classes have the most examples). The metric
    # to watch given this dataset's ~9x class-size imbalance -- see
    # dct_common/metrics.py:balanced_accuracy's docstring.
    balanced_table = {
        "train": {
            "float": balanced_accuracy(train_float["preds"], train_float["labels"], cfg.num_classes),
            "qat": balanced_accuracy(train_qat["preds"], train_qat["labels"], cfg.num_classes),
            "int8": balanced_accuracy(train_int8_preds, y_train, cfg.num_classes),
        },
        "val": {
            "float": balanced_accuracy(val_float["preds"], val_float["labels"], cfg.num_classes),
            "qat": balanced_accuracy(val_qat["preds"], val_qat["labels"], cfg.num_classes),
            "int8": balanced_accuracy(val_int8_preds, y_val, cfg.num_classes),
        },
        "test": {
            "float": balanced_accuracy(test_metrics["preds"], test_metrics["labels"], cfg.num_classes),
            "qat": balanced_accuracy(qat_test_metrics["preds"], qat_test_metrics["labels"], cfg.num_classes),
            "int8": balanced_accuracy(int8_test_preds, y_test, cfg.num_classes),
        },
    }
    print("\nBALANCED ACCURACY (macro-avg per-class recall -- every class weighted equally)")
    print_accuracy_table(balanced_table)

    with open(ARTIFACTS_DIR / "accuracy_table.json", "w") as f:
        json.dump({**accuracy_table, "balanced": balanced_table}, f, indent=2)

    # Rows=true class, cols=predicted class, same convention as
    # print_classification_report above -- one matrix per split x variant
    # so a shift specifically introduced by quantization (float -> qat ->
    # int8) is visible, not just the scalar accuracy drop.
    confusion_matrices = {
        "class_names": cfg.active_class_names,
        "train": {
            "float": confusion_matrix(train_float["preds"], train_float["labels"], cfg.num_classes).tolist(),
            "qat": confusion_matrix(train_qat["preds"], train_qat["labels"], cfg.num_classes).tolist(),
            "int8": confusion_matrix(train_int8_preds, y_train, cfg.num_classes).tolist(),
        },
        "val": {
            "float": confusion_matrix(val_float["preds"], val_float["labels"], cfg.num_classes).tolist(),
            "qat": confusion_matrix(val_qat["preds"], val_qat["labels"], cfg.num_classes).tolist(),
            "int8": confusion_matrix(val_int8_preds, y_val, cfg.num_classes).tolist(),
        },
        "test": {
            "float": confusion_matrix(test_metrics["preds"], test_metrics["labels"], cfg.num_classes).tolist(),
            "qat": confusion_matrix(qat_test_metrics["preds"], qat_test_metrics["labels"], cfg.num_classes).tolist(),
            "int8": confusion_matrix(int8_test_preds, y_test, cfg.num_classes).tolist(),
        },
    }
    with open(ARTIFACTS_DIR / "confusion_matrix.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)
    print(f"  wrote {ARTIFACTS_DIR / 'confusion_matrix.json'}")


if __name__ == "__main__":
    main()
