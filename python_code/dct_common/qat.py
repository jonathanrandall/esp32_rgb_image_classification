"""Quantization-aware-training building blocks shared by both models.

Both `QatLinear` (dense) and `QatConv2d` (conv+BN+ReLU) follow the same
recipe: fake-quantize the weight per-output-channel on every forward pass
(straight-through estimator), and optionally fake-quantize the output
activation at an int8 scale tracked as an eval-frozen EMA of the observed
max-abs activation -- not recomputed fresh per batch, since no real
firmware can do that; the EMA value each layer converges to during
training is used directly at export time. See `models/mlp.py` for the
original design writeup (dense1's weight-folding history, why activation
quantization is EMA-tracked, etc.) -- this module is the code, that's the
rationale.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FakeQuantSTE(torch.autograd.Function):
    """Round-and-clamp to signed int8 range with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def fake_quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(x, scale)


class QatLinear(nn.Module):
    """Dense layer with per-output-channel weight fake quantization and
    optional int8 fake quantization of its output activation.

    `input_scale`, if given, is a per-input-feature scale vector this
    layer's raw input was dequantized with (x = q * input_scale) -- e.g.
    the MLP's flat DCT feature vector, where every coefficient position
    has its own scale, unlike every other QatLinear use in this codebase
    (CNN output heads), which only ever see a single-scale pooled
    activation and pass input_scale=None. When set, the weight is folded
    with input_scale *before* its per-output-channel quantization scale is
    computed and it's fake-quantized -- exactly mirroring
    `Int8MlpReference`'s real int8 export math in models/mlp.py, then
    unfolded back so it can still be used with the dequantized x. Without
    this, training fake-quantized the *raw* weight (scale from the raw
    weight's own max-abs) while export quantized the *folded* weight
    (scale from the folded weight's max-abs, which can differ by several
    times whenever input_scale varies across features) -- a real
    train/export mismatch the straight-through estimator was never able
    to adapt the weights to, which showed up as poor int8-vs-QAT
    agreement concentrated in low-margin (near decision boundary)
    predictions."""

    def __init__(self, in_features: int, out_features: int, activation_ema_momentum: float = 0.1, input_scale=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation_ema_momentum = activation_ema_momentum
        self.register_buffer("act_scale_ema", torch.tensor(1.0))
        self.register_buffer("act_scale_initialized", torch.tensor(False))
        if input_scale is not None:
            self.register_buffer("input_scale", torch.as_tensor(input_scale, dtype=torch.float32))
        else:
            self.input_scale = None

    def forward(self, x: torch.Tensor, quantize_activation: bool) -> torch.Tensor:
        weight = self.linear.weight
        if self.input_scale is not None:
            folded = weight * self.input_scale[None, :]
            per_channel_max = folded.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            weight_scale = per_channel_max / 127.0
            q_weight = fake_quantize(folded, weight_scale) / self.input_scale[None, :]
        else:
            per_channel_max = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            weight_scale = per_channel_max / 127.0
            q_weight = fake_quantize(weight, weight_scale)

        out = F.linear(x, q_weight, self.linear.bias)

        if quantize_activation:
            if self.training:
                batch_max = out.detach().abs().amax().clamp_min(1e-8)
                if not bool(self.act_scale_initialized):
                    self.act_scale_ema.copy_(batch_max)
                    self.act_scale_initialized.fill_(True)
                else:
                    self.act_scale_ema.mul_(1.0 - self.activation_ema_momentum)
                    self.act_scale_ema.add_(self.activation_ema_momentum * batch_max)
            act_scale = (self.act_scale_ema / 127.0).clamp_min(1e-10)
            out = fake_quantize(out, act_scale)
        return out


class QatConv2d(nn.Module):
    """Conv2d + BatchNorm2d + ReLU with per-output-channel weight fake
    quantization and optional int8 fake quantization of its output
    activation -- the conv analogue of `QatLinear`.

    BatchNorm is kept as a real, separately-trained `nn.BatchNorm2d` during
    QAT (not fake-quantized itself, not folded into the conv weight here)
    -- exactly the same "train unfolded, fold only at export" choice the
    MLP's dense1 layer made for its input-scale folding: folding during
    training adds gradient-scale noise (dividing per-channel quantities
    with a wide spread) without measurably improving int8-vs-QAT
    agreement, so the model trains against the simpler unfolded op and the
    export step (see `DctCnnClassifierQat`'s int8 export in `models/cnn.py`)
    algebraically folds BN's running stats into the conv weight afterward.
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int = 3, stride: int = 1, padding: int = 1,
        activation_ema_momentum: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_ema_momentum = activation_ema_momentum
        self.register_buffer("act_scale_ema", torch.tensor(1.0))
        self.register_buffer("act_scale_initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor, quantize_activation: bool) -> torch.Tensor:
        weight = self.conv.weight
        per_channel_max = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        weight_scale = per_channel_max / 127.0
        q_weight = fake_quantize(weight, weight_scale)

        out = F.conv2d(x, q_weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        out = self.bn(out)
        out = torch.relu(out)

        if quantize_activation:
            if self.training:
                batch_max = out.detach().abs().amax().clamp_min(1e-8)
                if not bool(self.act_scale_initialized):
                    self.act_scale_ema.copy_(batch_max)
                    self.act_scale_initialized.fill_(True)
                else:
                    self.act_scale_ema.mul_(1.0 - self.activation_ema_momentum)
                    self.act_scale_ema.add_(self.activation_ema_momentum * batch_max)
            act_scale = (self.act_scale_ema / 127.0).clamp_min(1e-10)
            out = fake_quantize(out, act_scale)
        return out
