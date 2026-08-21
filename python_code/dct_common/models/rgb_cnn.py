"""RGB-pixel CNN -- a pixel-domain baseline for train_rgb_cnn.py, structurally
similar to models/cnn.py (a stack of Conv2d+BatchNorm2d+ReLU stages,
configurable channel widths, global average pool -> Dropout -> Linear head,
same float -> QAT -> bit-exact-int8 discipline) but NOT a subclass of it --
cnn.py's trunk shape (lum_conv + stride2_conv + concat AC/chroma +
post_concat_conv) exists specifically to fuse the DCT model's several
differently-shaped coefficient-plane inputs; RGB pixels are already one
plain [3, H, W] tensor with no such fusion to do, so the trunk here is a
single, variable-length stack of conv stages instead of DctCnnClassifier's
fixed 3-named-stages-plus-extras shape. What IS reused directly (not
duplicated): QatConv2d/QatLinear (qat.py), fold_batchnorm_into_conv and
the low-level int8 primitives (quantization.py) -- those are already
architecture-agnostic.

This model IS deployed, unlike every other model here: it is the
pixel-domain arm of the DCT-vs-RGB latency comparison. Export via
export_rgb_cnn_c_weights.py (ESP-NN accelerated, NHWC activations),
firmware in esp32_cam/esp32_rgb_cnn/. An earlier version of this
docstring called it "not a firmware candidate" on the grounds that it
would need an on-device JPEG-to-RGB decode; that is wrong on both counts.
The firmware captures RGB565 directly and never decodes a JPEG -- the
real pipeline cost is the reverse, a software JPEG encode needed to keep
streaming working once the capture is no longer already a JPEG.

The rest of this project's premise is unchanged: everything else here
reads JPEG DCT coefficients directly and never touches pixels. This
module is the deliberate exception that makes that premise measurable.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..qat import QatConv2d, QatLinear
from ..quantization import (
    avg_pool_int_exact,
    check_conv_accumulator_overflow,
    conv2d_int8_reference,
    dense_int8_reference_batched,
    quantize_dense_layer,
    quantize_multipliers,
)
from .cnn import fold_batchnorm_into_conv  # architecture-agnostic, reused not duplicated


def _conv_stage_strides(conv_channels: tuple) -> list:
    """First stage is stride 1 (no downsampling yet), every subsequent
    stage in conv_channels is stride 2 -- this is what actually reduces
    the RGB input's much larger native resolution (compared to the DCT
    models' already-8x-downsampled block grid) down to something a global
    average pool can meaningfully summarize."""
    return [1] + [2] * (len(conv_channels) - 1)


class RgbCnnClassifier(nn.Module):
    def __init__(self, conv_channels: tuple, extra_conv_channels: tuple, dropout: float, num_classes: int) -> None:
        super().__init__()
        strides = _conv_stage_strides(conv_channels)
        stages = []
        in_ch = 3
        for out_ch, stride in zip(conv_channels, strides):
            stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        for out_ch in extra_conv_channels:
            stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            ))
            in_ch = out_ch
        self.conv_stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_stages(x)
        h = self.pool(h).flatten(1)
        return self.head_dropout(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_trunk(x))


class RgbCnnClassifierQat(nn.Module):
    def __init__(self, conv_channels: tuple, extra_conv_channels: tuple, num_classes: int) -> None:
        super().__init__()
        strides = _conv_stage_strides(conv_channels)
        qat_stages = []
        in_ch = 3
        for out_ch, stride in zip(conv_channels, strides):
            qat_stages.append(QatConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
            in_ch = out_ch
        for out_ch in extra_conv_channels:
            qat_stages.append(QatConv2d(in_ch, out_ch, kernel_size=3, padding=1))
            in_ch = out_ch
        self.conv_stages = nn.ModuleList(qat_stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = QatLinear(in_ch, num_classes)

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for stage in self.conv_stages:
            h = stage(h, quantize_activation=True)
        return self.pool(h).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.forward_trunk(x), quantize_activation=False)


def init_rgb_cnn_qat_from_float(qat_model: RgbCnnClassifierQat, float_model: RgbCnnClassifier) -> None:
    float_convs = [m for m in float_model.conv_stages]  # nn.Sequential of nn.Sequential(Conv2d, BN, ReLU[, Dropout2d])
    for qat_block, float_block in zip(qat_model.conv_stages, float_convs):
        conv, bn = float_block[0], float_block[1]
        qat_block.conv.weight.data.copy_(conv.weight.data)
        qat_block.conv.bias.data.copy_(conv.bias.data)
        qat_block.bn.load_state_dict(bn.state_dict())
    qat_model.output.linear.weight.data.copy_(float_model.fc.weight.data)
    qat_model.output.linear.bias.data.copy_(float_model.fc.bias.data)


@torch.no_grad()
def calibrate_rgb_output_scale(qat_model: RgbCnnClassifierQat, loader: DataLoader, percentile: float = 99.9) -> float:
    """Same percentile-of-|logit| calibration as cnn.py's
    calibrate_cnn_output_scale, adapted for this model's single-input,
    (xb, yb)-batch loader shape (cnn.py's version expects (xb, xcb, yb)
    triples for the DCT models' Y+chroma inputs, so it can't be reused
    directly here)."""
    qat_model.eval()
    abs_logit_batches = []
    for xb, _ in loader:
        logits = qat_model(xb)
        abs_logit_batches.append(logits.abs().flatten())
    abs_logits = torch.cat(abs_logit_batches).numpy()
    threshold = np.percentile(abs_logits, percentile)
    return max(float(threshold), 1e-8) / 127.0


class Int8RgbCnnReference:
    """Bit-exact int8 NumPy reference for RgbCnnClassifierQat. Input
    quantization is a fixed, exact scale=1.0 (see rgb_features.py's
    docstring: centered pixel values are already exactly int8-
    representable, unlike DCT coefficients' heavy-tailed distribution --
    no percentile calibration needed or done for the input, only for the
    output layer, same as every other int8 reference in this project)."""

    INPUT_SCALE = 1.0

    def __init__(self, qat_model: RgbCnnClassifierQat, calibration_loader: DataLoader, percentile: float = 99.9):
        output_scale = calibrate_rgb_output_scale(qat_model, calibration_loader, percentile=percentile)

        self.conv_layers_q = []
        prev_act_scale = self.INPUT_SCALE
        for i, block in enumerate(qat_model.conv_stages):
            act_scale_i = float(block.act_scale_ema.item() / 127.0)
            w_folded, b_folded = fold_batchnorm_into_conv(block.conv, block.bn)
            w_with_input = w_folded * prev_act_scale
            weight_scale = np.maximum(np.abs(w_with_input).reshape(w_with_input.shape[0], -1).max(axis=1), 1e-8) / 127.0
            w_q = np.clip(np.round(w_with_input / weight_scale[:, None, None, None]), -127, 127).astype(np.int8)
            b_q = np.round(b_folded / weight_scale).astype(np.int32)
            mult, shift = quantize_multipliers(weight_scale / act_scale_i)
            check_conv_accumulator_overflow(b_q, w_q, f"conv[{i}]")
            self.conv_layers_q.append((w_q, b_q, mult, shift, block.conv.stride[0]))
            prev_act_scale = act_scale_i
        self.final_act_scale = prev_act_scale

        self.w_out_q, self.b_out_q, self.mult_out, self.shift_out = quantize_dense_layer(
            qat_model.output.linear, self.final_act_scale, output_scale, "output",
        )
        self.output_scale = output_scale

    def _run_conv_stack(self, q_rgb: np.ndarray) -> np.ndarray:
        h = q_rgb
        for w_q, b_q, mult, shift, stride in self.conv_layers_q:
            h = conv2d_int8_reference(h, w_q, b_q, mult, shift, stride=stride, padding=1, output_min=0, output_max=127)
        pooled = avg_pool_int_exact(h, (1, 1))
        return pooled.reshape(pooled.shape[0], pooled.shape[1])

    def forward(self, q_rgb: np.ndarray) -> np.ndarray:
        flat = self._run_conv_stack(q_rgb)
        return dense_int8_reference_batched(flat, self.w_out_q, self.b_out_q, self.mult_out, self.shift_out, output_min=-128, output_max=127)

    def predict(self, q_rgb: np.ndarray) -> np.ndarray:
        return self.forward(q_rgb).argmax(axis=1)

    def accuracy(self, q_rgb: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(q_rgb) == y).mean())
