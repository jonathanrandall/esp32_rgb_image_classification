"""Shared configuration for both the MLP and CNN classifiers.

One dataclass covers both models -- capture size, dataset, and quantization
fields are used by both; CNN-only fields (channel widths, chroma, dropout)
are simply unused by the MLP script and vice versa. Keeping a single Config
means dataset/feature code never has to guess which model it's serving.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .zigzag import SCAN_ORDERS

# The 23 everyday_openimages{160x120,96x96}/{train,val}/<name> classes --
# see everyday_openimages160x120/README.md for provenance (Open Images V7
# detections, plus a Places365 pass for "garden") and the meta/
# intersection-tracking format. Supersedes the earlier 6-class
# cocoimages-based list (appliances, birds, car, furniture, garden,
# people): that list's "appliances"/"birds"/"furniture" have no
# equivalent here (furniture was split into its actual constituent
# classes -- chair/couch/table/cupboard/bookshelves/doors -- instead of
# one broad label), while car/garden/people carried over unchanged. This
# is the fallback default -- see CLASS_NAMES below for where the actual
# value comes from.
_DEFAULT_CLASS_NAMES = [
    "books", "bookshelves", "bottles", "bowls", "car", "chair", "couch",
    "cupboard", "cups", "cutlery", "dishwasher", "doors", "fridge",
    "fruit", "garden", "keyboard", "laptop", "microwave", "monitor",
    "mouse", "people", "plates", "table",
]

# CLASS_NAMES is read from this small, git-tracked JSON sidecar file if it
# exists (written by build_data.py after a merge/filter run -- see that
# script's top-of-file comment), falling back to _DEFAULT_CLASS_NAMES
# above otherwise. This must match what dct_common/dataset.py discovers
# on disk; it's read once at import time from a tiny static file (not
# derived from the dataset directory itself, which can be missing/absent)
# so Config validation still works even when the multi-GB dataset
# directory isn't there -- same reasoning every CLASS_NAMES list this has
# replaced already had, just now automatable instead of a manual Python
# source edit every time the class list changes.
_CLASS_NAMES_FILE = Path(__file__).resolve().parent / "class_names.json"


def _load_class_names() -> list:
    if _CLASS_NAMES_FILE.exists():
        with open(_CLASS_NAMES_FILE) as f:
            return json.load(f)
    return list(_DEFAULT_CLASS_NAMES)


CLASS_NAMES = _load_class_names()


@dataclass(frozen=True)
class Config:
    # Whole captured frame, resized (never cropped) to this size before JPEG
    # encoding. capture_width must always be a multiple of 16 (exact
    # horizontal chroma block tiling under either subsampling mode below);
    # capture_height must be a multiple of 16 under 4:2:0 (vertical chroma
    # downsampling too) but only needs to be a multiple of 8 under 4:2:2
    # (no vertical chroma downsampling) -- see chroma_subsampling. 96x96
    # matched the ESP32 camera driver's original custom FRAMESIZE_96X96
    # config; 160x120 matches the OV2640's native FRAMESIZE_QQVGA.
    capture_width: int = 96
    capture_height: int = 96

    # Real camera hardware (OV2640, confirmed by inspecting live-captured
    # JPEG headers) outputs 4:2:2 chroma subsampling (H=2,V=1 -- horizontal
    # downsampling only), not the 4:2:0 (H=2,V=2) this pipeline originally
    # assumed unconditionally -- see issues.md. "4:2:0" is kept as the
    # default so existing 96x96 configs/results are unaffected; new runs
    # targeting real camera deployment should use "4:2:2".
    chroma_subsampling: str = "4:2:0"

    # Coefficients retained per Y (luma) block: position 0 (DC) +
    # num_ac_coeffs AC positions in zig-zag order. 0 is valid (DC-only).
    num_ac_coeffs: int = 4

    # Same idea, independently, for Cb/Cr (chroma) blocks -- lets luma and
    # chroma carry different AC coefficient counts (e.g. lum DC+7AC with
    # chroma DC-only: num_ac_coeffs=7, num_chroma_ac_coeffs=0). None (the
    # default) means "match num_ac_coeffs", preserving the original
    # single-knob behavior for any existing caller that only sets
    # num_ac_coeffs. Only meaningful when use_chroma is True. See
    # num_chroma_coeffs/c_features below and features.py's per-component
    # coefficient counts.
    num_chroma_ac_coeffs: int | None = None

    jpeg_quality: int = 80

    # Which (row, col) traversal order picks the first num_coeffs positions
    # out of each 8x8 block -- "zigzag" (standard JPEG scan, default) or
    # "axis_first" (pure horizontal/vertical frequencies tried before any
    # diagonal/mixed term, see dct_common/zigzag.py:generate_axis_first_order).
    # Same meaning for both the MLP's flat layout and the CNN's plane layout.
    coeff_scan_order: str = "zigzag"

    # Percentile (of |value|) used for both input-scale and output-scale
    # int8 calibration, in both models -- see calibrate_input_scales in
    # quantization.py and calibrate_output_scale in models/mlp.py.
    calibration_percentile: float = 99.9

    # --- MLP-only ---
    hidden_1: int = 64
    hidden_2: int = 32

    # Whether to include Cb/Cr chroma at all -- shared by both model
    # families, though it means something different in each: for the CNN
    # it toggles the chroma-fusion branch off; for the MLP (no separate
    # branch to switch off) it drops Cb/Cr from the flat feature vector
    # entirely, shrinking input_size to match (see extract_flat_features
    # in features.py). False for both matches real camera hardware: the
    # OV2640's chroma channel isn't captured reliably, see issues.md.
    use_chroma: bool = True

    # --- CNN-only ---
    lum_channels: int = 16
    stride2_channels: int = 32
    post_concat_channels: int = 64
    extra_conv_channels: tuple = (64,)
    dropout: float = 0.3
    label_smoothing: float = 0.05

    # --- DctCnnMlpClassifier-only: hidden layer widths for the MLP head
    # (Linear->ReLU->Dropout per entry) that replaces the plain single-
    # Linear head after global pooling. Empty tuple reproduces the plain
    # CNN's head exactly.
    cnn_mlp_hidden: tuple = (64,)

    # --- DctCnnAcClassifier-only: output depth of the learned 1x1 conv
    # projections applied to the AC-luma and chroma planes before they're
    # concatenated with stride2_conv's output, replacing the plain CNN's
    # raw/unlearned pass-through at that merge point. 8 beat 16 in testing
    # (better test accuracy, smaller train/test gap, cheaper) -- more
    # channels here gave the model more room to overfit the training set
    # without a matching generalization gain, on this dataset size.
    ac_proj_channels: int = 8
    chroma_proj_channels: int = 8

    # Subset of CLASS_NAMES to train/evaluate on -- None means all 7.
    selected_classes: tuple | None = None

    # --- shared training knobs ---
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    early_stop_patience: int = 12
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.num_chroma_ac_coeffs is None:
            object.__setattr__(self, "num_chroma_ac_coeffs", self.num_ac_coeffs)
        if self.chroma_subsampling not in ("4:2:0", "4:2:2"):
            raise ValueError(f"chroma_subsampling must be '4:2:0' or '4:2:2', got {self.chroma_subsampling!r}")
        if self.capture_width % 16 != 0:
            raise ValueError(
                f"capture_width must be a multiple of 16 for exact horizontal chroma "
                f"block tiling, got {self.capture_width}"
            )
        if self.chroma_subsampling == "4:2:0" and self.capture_height % 16 != 0:
            raise ValueError(
                f"capture_height must be a multiple of 16 under 4:2:0 chroma subsampling "
                f"(vertical downsampling too), got {self.capture_height}"
            )
        if self.chroma_subsampling == "4:2:2" and self.capture_height % 8 != 0:
            raise ValueError(
                f"capture_height must be a multiple of 8 under 4:2:2 chroma subsampling "
                f"(no vertical chroma downsampling), got {self.capture_height}"
            )
        if self.coeff_scan_order not in SCAN_ORDERS:
            raise ValueError(f"coeff_scan_order must be one of {sorted(SCAN_ORDERS)}, got {self.coeff_scan_order!r}")
        if self.selected_classes is not None:
            if len(self.selected_classes) == 0:
                raise ValueError("selected_classes must be non-empty (or None to use all classes)")
            unknown = set(self.selected_classes) - set(CLASS_NAMES)
            if unknown:
                raise ValueError(f"selected_classes has unknown class name(s): {sorted(unknown)}")
            if len(set(self.selected_classes)) != len(self.selected_classes):
                raise ValueError(f"selected_classes has duplicate name(s): {self.selected_classes}")

    @property
    def active_class_names(self) -> list:
        """Class names used this run, in label-index order (0..num_classes-1)."""
        return list(self.selected_classes) if self.selected_classes is not None else list(CLASS_NAMES)

    @property
    def num_classes(self) -> int:
        return len(self.active_class_names)

    @property
    def num_coeffs(self) -> int:
        return 1 + self.num_ac_coeffs

    @property
    def num_chroma_coeffs(self) -> int:
        return 1 + self.num_chroma_ac_coeffs

    @property
    def y_rows(self) -> int:
        return self.capture_height // 8

    @property
    def y_cols(self) -> int:
        return self.capture_width // 8

    @property
    def c_rows(self) -> int:
        """Chroma block-grid rows. Under 4:2:0, chroma is downsampled 2x
        vertically like luma is block-tiled (//16); under 4:2:2 there's no
        vertical chroma downsampling, so this matches y_rows exactly (//8)."""
        if self.chroma_subsampling == "4:2:2":
            return self.capture_height // 8
        return self.capture_height // 16

    @property
    def c_cols(self) -> int:
        """Chroma block-grid columns -- horizontal downsampling by 2 is the
        same under both 4:2:0 and 4:2:2."""
        return self.capture_width // 16

    @property
    def y_features(self) -> int:
        return self.y_rows * self.y_cols * self.num_coeffs

    @property
    def c_features(self) -> int:
        return self.c_rows * self.c_cols * self.num_chroma_coeffs

    @property
    def input_size(self) -> int:
        """Flattened MLP input size: all Y, then (if use_chroma) all Cb,
        then all Cr. use_chroma=False shrinks this to y_features only --
        see extract_flat_features in features.py."""
        return self.y_features + (2 * self.c_features if self.use_chroma else 0)
