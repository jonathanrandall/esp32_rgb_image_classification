"""Spatial CNN over Y-channel DCT coefficient planes, fused with Cb/Cr chroma:
the float model, its QAT twin, and the bit-exact int8 NumPy reference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..config import Config
from ..qat import QatConv2d, QatLinear
from ..quantization import (
    check_accumulator_overflow,
    check_conv_accumulator_overflow,
    conv2d_int8_reference,
    dense_int8_reference_batched,
    avg_pool_int_exact,
    quantize_multipliers,
)


class DctCnnClassifier(nn.Module):
    """
    CNN over Y-channel DCT coefficient planes (kept in their natural block
    grid, never flattened), fused with Cb/Cr chroma planes:

      Y DC plane (1 channel)
        -> conv "lum" stage                               [y_rows x y_cols]
        -> conv "stride 2" stage (downsamples)             [~half resolution]
        -> concat first `num_ac_coeffs` Y AC planes (adaptive-avg-pooled to
           match) and, if use_chroma, the Cb/Cr planes (already ~matching
           resolution at 96x96 capture)
        -> conv "post-concat" stage (+ Dropout2d)
        -> optional extra conv stages (cfg.extra_conv_channels, + Dropout2d)
        -> global average pool -> Dropout -> FC -> logits
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.num_ac_coeffs = cfg.num_ac_coeffs
        self.use_chroma = cfg.use_chroma

        self.lum_conv = nn.Sequential(
            nn.Conv2d(1, cfg.lum_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.lum_channels),
            nn.ReLU(inplace=True),
        )
        self.stride2_conv = nn.Sequential(
            nn.Conv2d(cfg.lum_channels, cfg.stride2_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.stride2_channels),
            nn.ReLU(inplace=True),
        )

        post_concat_in = cfg.stride2_channels + cfg.num_ac_coeffs
        if cfg.use_chroma:
            post_concat_in += 2 * cfg.num_chroma_coeffs
        self.post_concat_conv = nn.Sequential(
            nn.Conv2d(post_concat_in, cfg.post_concat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.post_concat_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(cfg.dropout),
        )

        extra_layers = []
        in_channels = cfg.post_concat_channels
        for out_channels in cfg.extra_conv_channels:
            extra_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(cfg.dropout),
            ]
            in_channels = out_channels
        self.extra_convs = nn.Sequential(*extra_layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(in_channels, cfg.num_classes)

    def forward_trunk(self, x: torch.Tensor, x_chroma: torch.Tensor = None) -> torch.Tensor:
        """Everything up to (and including) the pooled, dropout-applied
        feature vector -- shared by DctCnnClassifier and any subclass that
        swaps out the head (e.g. DctCnnMlpClassifier). x: (N, num_coeffs,
        y_rows, y_cols); x_chroma: (N, 2*num_coeffs, c_rows, c_cols)."""
        dc = x[:, 0:1]
        h = self.lum_conv(dc)
        h = self.stride2_conv(h)

        parts = [h]
        if self.num_ac_coeffs > 0:
            ac = x[:, 1:]
            parts.append(F.adaptive_avg_pool2d(ac, output_size=h.shape[-2:]))
        if self.use_chroma:
            parts.append(F.adaptive_avg_pool2d(x_chroma, output_size=h.shape[-2:]))
        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        h = self.post_concat_conv(h)
        h = self.extra_convs(h)
        h = self.pool(h).flatten(1)
        return self.head_dropout(h)

    def forward(self, x: torch.Tensor, x_chroma: torch.Tensor = None) -> torch.Tensor:
        return self.fc(self.forward_trunk(x, x_chroma))


# --------------------------------------------------------------------- #
# Quantization-aware training. Same shape as the float model above, but
# every conv is a QatConv2d (fake-quantized weight, EMA-tracked int8
# activation) and there's no Dropout -- QAT fine-tuning runs a handful of
# epochs at a low LR starting from the float model's already-regularized
# weights, so it doesn't need its own regularization. BatchNorm stays a
# real nn.BatchNorm2d during training (not fake-quantized, not folded) --
# same "train unfolded, fold only at export" choice as QatConv2d's
# docstring explains for its weight-folding.
# --------------------------------------------------------------------- #

class DctCnnClassifierQat(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.num_ac_coeffs = cfg.num_ac_coeffs
        self.use_chroma = cfg.use_chroma

        self.lum_conv = QatConv2d(1, cfg.lum_channels, kernel_size=3, padding=1)
        self.stride2_conv = QatConv2d(cfg.lum_channels, cfg.stride2_channels, kernel_size=3, stride=2, padding=1)

        post_concat_in = cfg.stride2_channels + cfg.num_ac_coeffs
        if cfg.use_chroma:
            post_concat_in += 2 * cfg.num_chroma_coeffs
        self.post_concat_conv = QatConv2d(post_concat_in, cfg.post_concat_channels, kernel_size=3, padding=1)

        extra_qat = []
        in_channels = cfg.post_concat_channels
        for out_channels in cfg.extra_conv_channels:
            extra_qat.append(QatConv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        self.extra_convs = nn.ModuleList(extra_qat)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = QatLinear(in_channels, cfg.num_classes)

    def forward_trunk(self, x: torch.Tensor, x_chroma: torch.Tensor = None) -> torch.Tensor:
        """Everything up to the pooled feature vector, before the output
        layer -- shared with any subclass that inserts extra head layers
        (e.g. DctCnnMlpClassifierQat) before the final QatLinear."""
        dc = x[:, 0:1]
        h = self.lum_conv(dc, quantize_activation=True)
        h = self.stride2_conv(h, quantize_activation=True)

        parts = [h]
        if self.num_ac_coeffs > 0:
            ac = x[:, 1:]
            parts.append(F.adaptive_avg_pool2d(ac, output_size=h.shape[-2:]))
        if self.use_chroma:
            parts.append(F.adaptive_avg_pool2d(x_chroma, output_size=h.shape[-2:]))
        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        h = self.post_concat_conv(h, quantize_activation=True)
        for conv in self.extra_convs:
            h = conv(h, quantize_activation=True)

        return self.pool(h).flatten(1)

    def forward(self, x: torch.Tensor, x_chroma: torch.Tensor = None) -> torch.Tensor:
        h = self.forward_trunk(x, x_chroma)
        return self.output(h, quantize_activation=False)


def copy_conv_bn_block(qat_block: QatConv2d, float_block: nn.Sequential) -> None:
    """Copies one Conv2d+BatchNorm2d block's weights from a float
    `nn.Sequential(Conv2d, BatchNorm2d, ReLU, ...)` into the matching
    `QatConv2d`. Shared by copy_conv_stack_from_float and any variant with
    extra conv blocks of its own to copy, e.g. init_cnn_ac_qat_from_float's
    ac_proj/chroma_proj."""
    conv, bn = float_block[0], float_block[1]
    qat_block.conv.weight.data.copy_(conv.weight.data)
    qat_block.conv.bias.data.copy_(conv.bias.data)
    qat_block.bn.load_state_dict(bn.state_dict())


def copy_conv_stack_from_float(qat_model: DctCnnClassifierQat, float_model: DctCnnClassifier) -> None:
    """Copies lum_conv/stride2_conv/post_concat_conv/extra_convs' weights
    (conv + BatchNorm) from a trained float model into a QAT model with the
    same trunk. Shared by init_cnn_qat_from_float and (via subclassing) any
    variant with a different head, e.g. init_cnn_mlp_qat_from_float."""
    copy_conv_bn_block(qat_model.lum_conv, float_model.lum_conv)
    copy_conv_bn_block(qat_model.stride2_conv, float_model.stride2_conv)
    copy_conv_bn_block(qat_model.post_concat_conv, float_model.post_concat_conv)

    float_convs = [m for m in float_model.extra_convs if isinstance(m, nn.Conv2d)]
    float_bns = [m for m in float_model.extra_convs if isinstance(m, nn.BatchNorm2d)]
    for qat_block, conv, bn in zip(qat_model.extra_convs, float_convs, float_bns):
        qat_block.conv.weight.data.copy_(conv.weight.data)
        qat_block.conv.bias.data.copy_(conv.bias.data)
        qat_block.bn.load_state_dict(bn.state_dict())


def init_cnn_qat_from_float(qat_model: DctCnnClassifierQat, float_model: DctCnnClassifier) -> None:
    copy_conv_stack_from_float(qat_model, float_model)
    qat_model.output.linear.weight.data.copy_(float_model.fc.weight.data)
    qat_model.output.linear.bias.data.copy_(float_model.fc.bias.data)


@torch.no_grad()
def calibrate_cnn_output_scale(qat_model: DctCnnClassifierQat, loader: DataLoader, percentile: float = 99.9) -> float:
    """The output layer produces logits, not a hidden activation, so it
    never fake-quantizes its own output during QAT and still needs one
    calibration pass. Unlike the MLP's calibrate_output_scale (which
    hand-derives the quantized forward pass layer by layer), this just
    runs the QAT model's own eval-mode forward -- it already applies every
    upstream fake-quantization step at its frozen EMA scales, so there's
    nothing to re-derive. Assumes qat_model and the loader's batches are
    both on CPU.

    Uses a `percentile` of the observed |logit| distribution, not the raw
    max -- a single outlier-confident training example can otherwise set a
    scale far coarser than what typical (harder, lower-margin) examples
    need, collapsing most test logits to 0 after quantization. See the
    MLP's calibrate_output_scale docstring for the full writeup (this hit
    ~8% int8-vs-QAT agreement on Caltech-256's 256-way task before this fix)."""
    qat_model.eval()
    abs_logit_batches = []
    for xb, xcb, _ in loader:
        logits = qat_model(xb, xcb)
        abs_logit_batches.append(logits.abs().flatten())
    abs_logits = torch.cat(abs_logit_batches).numpy()
    threshold = np.percentile(abs_logits, percentile)
    return max(float(threshold), 1e-8) / 127.0


def fold_batchnorm_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple:
    """Algebraically fold BN's frozen running stats into the preceding
    conv's weight/bias, producing an equivalent bias-only conv -- the form
    actually deployable on hardware without a separate BN op. Returns
    (w_folded, b_folded) as float64 NumPy arrays."""
    gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn.bias.detach().cpu().numpy().astype(np.float64)
    running_mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    running_var = bn.running_var.detach().cpu().numpy().astype(np.float64)
    eps = bn.eps

    s = gamma / np.sqrt(running_var + eps)
    t = beta - gamma * running_mean / np.sqrt(running_var + eps)

    w = conv.weight.detach().cpu().numpy().astype(np.float64)
    b = conv.bias.detach().cpu().numpy().astype(np.float64)

    w_folded = w * s[:, None, None, None]
    b_folded = b * s + t
    return w_folded, b_folded


# --------------------------------------------------------------------- #
# Bit-exact int8 NumPy reference.
# --------------------------------------------------------------------- #

def _quantize_conv_layer(w_folded: np.ndarray, b_folded: np.ndarray, input_scale, act_scale: float, layer_name: str) -> tuple:
    """input_scale: scalar, or array broadcastable over the weight's
    in_channels axis (axis=1). Shared by Int8CnnReference._build_conv_stack
    and any subclass reusing it (e.g. Int8CnnMlpReference)."""
    if np.ndim(input_scale) == 0:
        w_with_input = w_folded * input_scale
    else:
        w_with_input = w_folded * np.asarray(input_scale)[None, :, None, None]
    weight_scale = np.maximum(np.abs(w_with_input).reshape(w_with_input.shape[0], -1).max(axis=1), 1e-8) / 127.0
    w_q = np.clip(np.round(w_with_input / weight_scale[:, None, None, None]), -127, 127).astype(np.int8)
    b_q = np.round(b_folded / weight_scale).astype(np.int32)
    mult, shift = quantize_multipliers(weight_scale / act_scale)
    check_conv_accumulator_overflow(b_q, w_q, layer_name)
    return w_q, b_q, mult, shift


class Int8CnnReference:
    """Bit-exact int8 NumPy reference for `DctCnnClassifierQat`, matching
    the intended firmware integer arithmetic (int32 accumulation, per-
    output-channel fixed-point requantization, no floating point), the
    conv analogue of `models/mlp.py`'s `Int8MlpReference`.

    Every conv layer folds its trained BatchNorm's running stats into its
    weight/bias (`fold_batchnorm_into_conv`) before quantizing, and folds
    its input's activation scale(s) in exactly the way the MLP's dense1
    folds its per-feature input scale -- generalized here to three cases:
    a single scalar scale (lum_conv's DC-plane input, stride2_conv's/each
    extra conv's single-scale input from the previous layer's uniform
    activation), and a per-input-channel vector (post_concat_conv's input,
    which mixes stride2_conv's uniform-scale channels with the
    individually-calibrated AC-luma and chroma channels -- built once in
    `__init__` as `combined_input_scale`).

    Built once from a trained `DctCnnClassifierQat` (must be on CPU) plus
    the three input calibration scales used to quantize its raw inputs
    (`dc_scale`: scalar; `ac_scales`: [num_ac_coeffs]; `chroma_scales`:
    [2*num_chroma_coeffs], only used if `cfg.use_chroma`) and a calibration loader
    (also CPU) for the output scale.
    """

    def __init__(
        self,
        qat_model: DctCnnClassifierQat,
        dc_scale: float,
        ac_scales: np.ndarray,
        chroma_scales: np.ndarray,
        cfg: Config,
        calibration_loader: DataLoader,
    ):
        output_scale = calibrate_cnn_output_scale(qat_model, calibration_loader, percentile=cfg.calibration_percentile)
        self._build_conv_stack(qat_model, dc_scale, ac_scales, chroma_scales, cfg)

        # Output layer: input is the last conv stage's pooled, uniform-scale
        # activation; reuses the MLP's plain (non-conv) folding logic since
        # this is just a dense layer once flattened.
        w_out_real = qat_model.output.linear.weight.detach().cpu().numpy().astype(np.float64)
        b_out_real = qat_model.output.linear.bias.detach().cpu().numpy().astype(np.float64)
        w_out_folded = w_out_real * self.final_act_scale
        weight_scale_out = np.maximum(np.abs(w_out_folded).max(axis=1), 1e-8) / 127.0
        self.w_out_q = np.clip(np.round(w_out_folded / weight_scale_out[:, None]), -127, 127).astype(np.int8)
        self.b_out_q = np.round(b_out_real / weight_scale_out).astype(np.int32)
        self.mult_out, self.shift_out = quantize_multipliers(weight_scale_out / output_scale)
        check_accumulator_overflow(self.b_out_q, self.w_out_q, w_out_real.shape[1], "output")
        self.output_scale = output_scale

    def _build_conv_stack(
        self,
        qat_model: DctCnnClassifierQat,
        dc_scale: float,
        ac_scales: np.ndarray,
        chroma_scales: np.ndarray,
        cfg: Config,
    ) -> None:
        """Quantizes lum_conv/stride2_conv/post_concat_conv/extra_convs --
        everything up to and including the pooled feature vector's
        activation scale (`self.final_act_scale`). Split out from __init__
        so a subclass with a different head (e.g. Int8CnnMlpReference, an
        MLP instead of a single dense layer) can reuse this unchanged and
        only implement what comes after the pooled features."""
        self.num_ac_coeffs = cfg.num_ac_coeffs
        self.use_chroma = cfg.use_chroma

        # Layer 1: lum_conv -- single input channel (the DC plane), scalar input scale.
        act_scale_lum = float(qat_model.lum_conv.act_scale_ema.item() / 127.0)
        w1_folded, b1_folded = fold_batchnorm_into_conv(qat_model.lum_conv.conv, qat_model.lum_conv.bn)
        self.w1_q, self.b1_q, self.mult1, self.shift1 = _quantize_conv_layer(w1_folded, b1_folded, dc_scale, act_scale_lum, "lum_conv")

        # Layer 2: stride2_conv -- input is lum_conv's uniform-scale activation.
        act_scale_stride2 = float(qat_model.stride2_conv.act_scale_ema.item() / 127.0)
        w2_folded, b2_folded = fold_batchnorm_into_conv(qat_model.stride2_conv.conv, qat_model.stride2_conv.bn)
        self.w2_q, self.b2_q, self.mult2, self.shift2 = _quantize_conv_layer(w2_folded, b2_folded, act_scale_lum, act_scale_stride2, "stride2_conv")

        # Layer 3: post_concat_conv -- input channels mix stride2_conv's
        # uniform scale with the individually-calibrated AC/chroma scales.
        scale_parts = [np.full(qat_model.stride2_conv.conv.out_channels, act_scale_stride2)]
        if cfg.num_ac_coeffs > 0:
            scale_parts.append(np.asarray(ac_scales, dtype=np.float64))
        if cfg.use_chroma:
            scale_parts.append(np.asarray(chroma_scales, dtype=np.float64))
        combined_input_scale = np.concatenate(scale_parts)
        self.combined_input_scale = combined_input_scale

        act_scale_postconcat = float(qat_model.post_concat_conv.act_scale_ema.item() / 127.0)
        w3_folded, b3_folded = fold_batchnorm_into_conv(qat_model.post_concat_conv.conv, qat_model.post_concat_conv.bn)
        self.w3_q, self.b3_q, self.mult3, self.shift3 = _quantize_conv_layer(w3_folded, b3_folded, combined_input_scale, act_scale_postconcat, "post_concat_conv")

        # Extra conv stages, each a plain single-scale-input fold like layer 2.
        self.extra_layers_q = []
        prev_act_scale = act_scale_postconcat
        for i, block in enumerate(qat_model.extra_convs):
            act_scale_i = float(block.act_scale_ema.item() / 127.0)
            w_folded, b_folded = fold_batchnorm_into_conv(block.conv, block.bn)
            w_q, b_q, mult, shift = _quantize_conv_layer(w_folded, b_folded, prev_act_scale, act_scale_i, f"extra_conv[{i}]")
            self.extra_layers_q.append((w_q, b_q, mult, shift))
            prev_act_scale = act_scale_i

        self.act_scale_lum = act_scale_lum
        self.act_scale_stride2 = act_scale_stride2
        self.act_scale_postconcat = act_scale_postconcat
        self.final_act_scale = prev_act_scale

    def _run_conv_stack(self, q_dc: np.ndarray, q_ac: np.ndarray, q_chroma: np.ndarray) -> np.ndarray:
        """Runs the quantized conv stack built by `_build_conv_stack` and
        returns the flattened, globally-pooled int8 feature vector
        [N, final_channels] -- the input to whatever dense head follows.

        q_dc: int8 [N, 1, y_rows, y_cols]
        q_ac: int8 [N, num_ac_coeffs, y_rows, y_cols] (ignored if num_ac_coeffs==0)
        q_chroma: int8 [N, 2*num_chroma_coeffs, c_rows, c_cols] (ignored if not use_chroma)
        """
        h = conv2d_int8_reference(q_dc, self.w1_q, self.b1_q, self.mult1, self.shift1, stride=1, padding=1, output_min=0, output_max=127)
        h = conv2d_int8_reference(h, self.w2_q, self.b2_q, self.mult2, self.shift2, stride=2, padding=1, output_min=0, output_max=127)

        target_hw = h.shape[-2:]
        parts = [h]
        if self.num_ac_coeffs > 0:
            parts.append(avg_pool_int_exact(q_ac, target_hw).astype(np.int8))
        if self.use_chroma:
            parts.append(avg_pool_int_exact(q_chroma, target_hw).astype(np.int8))
        h = np.concatenate(parts, axis=1)

        h = conv2d_int8_reference(h, self.w3_q, self.b3_q, self.mult3, self.shift3, stride=1, padding=1, output_min=0, output_max=127)
        for w_q, b_q, mult, shift in self.extra_layers_q:
            h = conv2d_int8_reference(h, w_q, b_q, mult, shift, stride=1, padding=1, output_min=0, output_max=127)

        pooled = avg_pool_int_exact(h, (1, 1))
        return pooled.reshape(pooled.shape[0], pooled.shape[1])

    def forward(self, q_dc: np.ndarray, q_ac: np.ndarray, q_chroma: np.ndarray) -> np.ndarray:
        """Returns int8 logits [N, num_classes]."""
        flat = self._run_conv_stack(q_dc, q_ac, q_chroma)
        return dense_int8_reference_batched(flat, self.w_out_q, self.b_out_q, self.mult_out, self.shift_out, output_min=-128, output_max=127)

    def predict(self, q_dc: np.ndarray, q_ac: np.ndarray, q_chroma: np.ndarray) -> np.ndarray:
        return self.forward(q_dc, q_ac, q_chroma).argmax(axis=1)

    def accuracy(self, q_dc: np.ndarray, q_ac: np.ndarray, q_chroma: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(q_dc, q_ac, q_chroma)
        return float((preds == y).mean())
