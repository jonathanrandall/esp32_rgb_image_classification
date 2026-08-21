"""Input calibration/quantization (MLP feature layout) and the generic
Q31-mantissa fixed-point requantization used by the bit-exact int8 MLP
reference (gemmlowp/TFLite/ESP-NN convention: int32 accumulation, then a
per-output-channel (multiplier, shift) requantize -- never floating-point
division at inference time).
"""

import math

import numpy as np

from .config import Config
from .features import component_slices


# --------------------------------------------------------------------- #
# Input calibration/quantization -- assumes the flat MLP feature layout.
# --------------------------------------------------------------------- #

def calibrate_input_scales(X: np.ndarray, cfg: Config, percentile: float | None = None) -> dict:
    """Per-(component, zig-zag position) scale, calibrated off `percentile`
    of |value| in X (default cfg.calibration_percentile)."""
    percentile = cfg.calibration_percentile if percentile is None else percentile
    slices = component_slices(cfg)
    block_counts = {"Y": cfg.y_rows * cfg.y_cols, "Cb": cfg.c_rows * cfg.c_cols, "Cr": cfg.c_rows * cfg.c_cols}
    comp_num_coeffs = {"Y": cfg.num_coeffs, "Cb": cfg.num_chroma_coeffs, "Cr": cfg.num_chroma_coeffs}

    scales = {}
    for comp_name, sl in slices.items():
        n_coeffs = comp_num_coeffs[comp_name]
        comp_values = X[:, sl].reshape(X.shape[0], block_counts[comp_name], n_coeffs)
        for k in range(n_coeffs):
            vals = comp_values[:, :, k].reshape(-1).astype(np.float64)
            threshold = max(np.percentile(np.abs(vals), percentile), 1e-6)
            scales[(comp_name, k)] = threshold / 127.0
    return scales


def scales_to_vector(scales: dict, cfg: Config) -> np.ndarray:
    """Flatten {(component, k): scale} into length num_coeffs +
    2*num_chroma_coeffs (chroma only present when cfg.use_chroma), ordered
    [Y0..Yk, Cb0..Cbj, Cr0..Crj] -- matches component_slices/
    extract_flat_features."""
    components = ("Y", "Cb", "Cr") if cfg.use_chroma else ("Y",)
    comp_num_coeffs = {"Y": cfg.num_coeffs, "Cb": cfg.num_chroma_coeffs, "Cr": cfg.num_chroma_coeffs}
    return np.array(
        [scales[(comp, k)] for comp in components for k in range(comp_num_coeffs[comp])],
        dtype=np.float64,
    )


def broadcast_scale_per_feature(scale_vec_3n: np.ndarray, cfg: Config) -> np.ndarray:
    """Expand the per-(component, coefficient) scales to length
    input_size, repeating each coefficient's scale across every spatial
    block. Component set and per-component coefficient count match
    scales_to_vector's (Cb/Cr only present when cfg.use_chroma)."""
    block_counts = {"Y": cfg.y_rows * cfg.y_cols, "Cb": cfg.c_rows * cfg.c_cols, "Cr": cfg.c_rows * cfg.c_cols}
    comp_num_coeffs = {"Y": cfg.num_coeffs, "Cb": cfg.num_chroma_coeffs, "Cr": cfg.num_chroma_coeffs}
    components = ("Y", "Cb", "Cr") if cfg.use_chroma else ("Y",)
    parts, idx = [], 0
    for comp_name in components:
        n = comp_num_coeffs[comp_name]
        comp_scales = scale_vec_3n[idx: idx + n]
        idx += n
        parts.append(np.tile(comp_scales, block_counts[comp_name]))
    return np.concatenate(parts)


def report_clipping_rates(X: np.ndarray, cfg: Config) -> None:
    for pct in (99.0, 99.5, 99.9, 100.0):
        scales = calibrate_input_scales(X, cfg, percentile=pct)
        scale_vec = broadcast_scale_per_feature(scales_to_vector(scales, cfg), cfg)
        q = X.astype(np.float64) / scale_vec[None, :]
        clip_rate = 100.0 * np.mean(np.abs(q) > 127)
        print(f"  percentile={pct:5.1f}  clip_rate={clip_rate:.4f}%")


def quantize_inputs(X: np.ndarray, scale_vec: np.ndarray) -> np.ndarray:
    """q = clamp(round(x / scale), -127, 127), per-feature scale."""
    q = np.round(X.astype(np.float64) / scale_vec[None, :])
    return np.clip(q, -127, 127).astype(np.int8)


# --------------------------------------------------------------------- #
# Per-channel calibration for spatial "plane" data (CNN layout: X shape
# [N, C, H, W] or [N, C]) -- same percentile-of-|value| idea as
# calibrate_input_scales above, generalized to arbitrary channel count
# instead of the flat MLP's fixed (component, coefficient) structure.
# --------------------------------------------------------------------- #

def calibrate_channel_scales(X: np.ndarray, percentile: float) -> np.ndarray:
    """One scale per channel (axis 1), calibrated off `percentile` of
    |value| across all other axes. Returns shape [C], dtype float64."""
    num_channels = X.shape[1]
    reduce_axes = tuple(a for a in range(X.ndim) if a != 1)
    scales = np.zeros(num_channels, dtype=np.float64)
    for c in range(num_channels):
        vals = np.take(X, c, axis=1).reshape(-1).astype(np.float64)
        threshold = max(np.percentile(np.abs(vals), percentile), 1e-6)
        scales[c] = threshold / 127.0
    return scales


def quantize_planes(X: np.ndarray, channel_scales: np.ndarray) -> np.ndarray:
    """q = clamp(round(x / scale), -127, 127), one scale per channel (axis 1).
    X shape [N, C, ...], channel_scales shape [C]."""
    shape = (1, -1) + (1,) * (X.ndim - 2)
    q = np.round(X.astype(np.float64) / channel_scales.reshape(shape))
    return np.clip(q, -127, 127).astype(np.int8)


# --------------------------------------------------------------------- #
# Generic Q31-mantissa fixed-point requantization.
# --------------------------------------------------------------------- #

def quantize_multiplier(real_multiplier: float) -> tuple:
    """Decompose a positive real multiplier into (int32 M, shift) such that
    real_multiplier == (M / 2**31) * 2**shift, M in [2**30, 2**31) -- the
    Q31-mantissa fixed-point convention used by gemmlowp/TFLite/ESP-NN."""
    if real_multiplier == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(real_multiplier)  # real = mantissa * 2**exponent, mantissa in [0.5, 1)
    q_mantissa = round(mantissa * (1 << 31))
    if q_mantissa == (1 << 31):
        q_mantissa //= 2
        exponent += 1
    assert (1 << 30) <= q_mantissa < (1 << 31), (real_multiplier, q_mantissa)
    return int(q_mantissa), int(exponent)


def quantize_multipliers(real_multipliers: np.ndarray) -> tuple:
    mults, shifts = zip(*(quantize_multiplier(float(m)) for m in real_multipliers))
    return np.array(mults, dtype=np.int64), np.array(shifts, dtype=np.int64)


def quantize_dense_layer(linear_module, input_scale: float, output_scale: float, layer_name: str) -> tuple:
    """Fold a scalar input_scale into a plain nn.Linear's weight and
    quantize to int8 weight / int32 bias / (mult, shift) -- the Q31-
    mantissa convention used throughout. `input_scale` must be a single
    scalar (a uniform-scale pooled activation, which is what every dense
    output head in this project reads); the MLP's dense1 layer needs a
    different, per-feature version of this folding instead (QatLinear's
    own `input_scale` handling in qat.py), since its raw input isn't
    uniform-scale. Shared by Int8CnnReference/Int8CnnPeopleReference's
    (models/cnn.py, models/cnn_people.py) output-layer quantization and
    models/rgb_cnn.py's -- this exact block used to be written out by
    hand in each of those; new code should call this instead of
    re-deriving it a fourth time."""
    w_real = linear_module.weight.detach().cpu().numpy().astype(np.float64)
    b_real = linear_module.bias.detach().cpu().numpy().astype(np.float64)
    w_folded = w_real * input_scale
    weight_scale = np.maximum(np.abs(w_folded).max(axis=1), 1e-8) / 127.0
    w_q = np.clip(np.round(w_folded / weight_scale[:, None]), -127, 127).astype(np.int8)
    b_q = np.round(b_real / weight_scale).astype(np.int32)
    mult, shift = quantize_multipliers(weight_scale / output_scale)
    check_accumulator_overflow(b_q, w_q, w_real.shape[1], layer_name)
    return w_q, b_q, mult, shift


def ragged_object_array(items: list) -> np.ndarray:
    """Build a 1D dtype=object array holding `items` verbatim (one array
    per slot, never concatenated/broadcast) -- for saving a variable-length
    list of per-layer weight/bias/mult/shift arrays (extra_conv stages,
    train_cnn_mlp.py's MLP head layers) into one .npz entry when there can
    be zero, one, or several. Plain `np.array(items, dtype=object)`
    usually works for this, but breaks with a genuinely confusing error
    ("could not broadcast input array from shape X into shape Y") whenever
    two items' shapes happen to share a leading dimension (e.g. two conv
    weight arrays shaped (64,128,3,3) and (64,64,3,3) -- both start with
    64) -- NumPy's shape-inference tries to build a regular ndarray first,
    gets partway through assuming a uniform shape, and only then discovers
    it can't continue. Building an empty object array up front and
    assigning into it element-by-element sidesteps that inference entirely
    and always does the ragged thing intended, regardless of shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def requantize_per_channel(acc: np.ndarray, multipliers: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """acc: int64 [..., out_channels]; multipliers/shifts: int64 [out_channels].
    out[..., j] = round_to_nearest(acc[..., j] * multiplier[j] / 2**31 * 2**shift[j])."""
    acc64 = acc.astype(np.int64)
    out = np.empty_like(acc64)
    for j in range(acc.shape[-1]):
        total_shift = 31 - int(shifts[j])
        product = acc64[..., j] * np.int64(multipliers[j])
        if total_shift > 0:
            rounding = np.int64(1) << (total_shift - 1)
            out[..., j] = (product + rounding) >> total_shift
        else:
            out[..., j] = product << (-total_shift)
    return out


def dense_int8_reference(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    multipliers: np.ndarray,
    shifts: np.ndarray,
    output_min: int = -128,
    output_max: int = 127,
) -> np.ndarray:
    acc = bias.astype(np.int64) + weights.astype(np.int64) @ x.astype(np.int64)
    out = requantize_per_channel(acc[None, :], multipliers, shifts)[0]
    return np.clip(out, output_min, output_max).astype(np.int8)


def dense_int8_reference_batched(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    multipliers: np.ndarray,
    shifts: np.ndarray,
    output_min: int = -128,
    output_max: int = 127,
) -> np.ndarray:
    """Same arithmetic as `dense_int8_reference`, batched: x is [N, in_features],
    returns [N, out_features]. Used for the CNN's final FC layer, where
    looping per-example (as the MLP reference does) would mean N separate
    tiny matvecs on top of N conv passes -- batching keeps this cheap."""
    acc = bias.astype(np.int64)[None, :] + x.astype(np.int64) @ weights.astype(np.int64).T
    out = requantize_per_channel(acc, multipliers, shifts)
    return np.clip(out, output_min, output_max).astype(np.int8)


def check_accumulator_overflow(bias: np.ndarray, weights: np.ndarray, input_length: int, layer_name: str) -> None:
    max_abs_bias = int(np.max(np.abs(bias.astype(np.int64))))
    worst_case = max_abs_bias + input_length * 127 * 127
    fits = worst_case < 2**31
    print(f"  {layer_name}: worst-case |accumulator| <= {worst_case:,}  (int32 max={2**31 - 1:,})  fits={fits}")
    assert fits, f"{layer_name} accumulator can overflow int32"


def check_conv_accumulator_overflow(bias: np.ndarray, weights: np.ndarray, layer_name: str) -> None:
    """Conv analogue of check_accumulator_overflow -- input_length is the
    per-output-pixel receptive field size (in_channels * kh * kw), read
    directly off the weight tensor shape [out_ch, in_ch, kh, kw]."""
    _, in_ch, kh, kw = weights.shape
    check_accumulator_overflow(bias, weights.reshape(weights.shape[0], -1), in_ch * kh * kw, layer_name)


# --------------------------------------------------------------------- #
# Spatial-plane int8 reference ops for the CNN: batched conv and adaptive
# average pooling (see avg_pool_int_exact's docstring for the pooling
# window rule).
# --------------------------------------------------------------------- #

def conv2d_int8_reference(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    multipliers: np.ndarray,
    shifts: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    output_min: int = -128,
    output_max: int = 127,
) -> np.ndarray:
    """Batched bit-exact int8 2D convolution: int32 accumulation via im2col,
    then the same per-output-channel (multiplier, shift) requantization as
    `dense_int8_reference`.

    x: int32/int8 [N, C_in, H, W]
    weights: int8 [C_out, C_in, KH, KW]
    bias: int32 [C_out]
    multipliers/shifts: int64 [C_out] (from quantize_multipliers)
    Returns int8 [N, C_out, H_out, W_out].
    """
    x64 = x.astype(np.int64)
    n, c_in, h, w = x64.shape
    c_out, c_in_w, kh, kw = weights.shape
    assert c_in == c_in_w, (c_in, c_in_w)

    if padding > 0:
        x64 = np.pad(x64, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_pad, w_pad = x64.shape[2], x64.shape[3]
    h_out = (h_pad - kh) // stride + 1
    w_out = (w_pad - kw) // stride + 1

    # im2col: gather every (kh, kw) shifted view, then flatten to a matmul.
    cols = np.empty((n, c_in, kh, kw, h_out, w_out), dtype=np.int64)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x64[:, :, i: i + h_out * stride: stride, j: j + w_out * stride: stride]
    cols = cols.reshape(n, c_in * kh * kw, h_out * w_out)
    w_flat = weights.astype(np.int64).reshape(c_out, c_in * kh * kw)

    acc = np.einsum("ok,nkp->nop", w_flat, cols) + bias.astype(np.int64)[None, :, None]
    acc = acc.reshape(n, c_out, h_out, w_out)

    acc_channels_last = np.moveaxis(acc, 1, -1)
    out_channels_last = requantize_per_channel(acc_channels_last, multipliers, shifts)
    out = np.moveaxis(out_channels_last, -1, 1)
    return np.clip(out, output_min, output_max).astype(np.int8)


def _round_div(numerator: np.ndarray, denom: int) -> np.ndarray:
    """Round-to-nearest (half away from zero) integer division, safe for
    negative numerators (plain `//` floors towards -inf, which would bias
    pooled AC/chroma coefficients, which can be negative)."""
    sign = np.sign(numerator)
    return sign * ((np.abs(numerator) + denom // 2) // denom)


def avg_pool_int_exact(x: np.ndarray, output_size: tuple) -> np.ndarray:
    """Integer average pooling from [N, C, H, W] to [N, C, oh, ow], using
    the same per-output-index window rule as
    `torch.nn.functional.adaptive_avg_pool2d`: output index i pools input
    range [floor(i*H/oh), ceil((i+1)*H/oh)). When H is an exact multiple
    of oh (true for this project's original 4:2:0/even-y_rows configs)
    this degenerates to non-overlapping bins of identical size kh=H/oh --
    but at odd y_rows (e.g. 160x120 under 4:2:2, y_rows=15) the ratio
    isn't exact and adjacent windows genuinely overlap by one row/column,
    same as the float/QAT model's adaptive_avg_pool2d does. Matching that
    window rule (not just a coarser same-size-bin approximation) is what
    keeps this "bit-exact" reference actually consistent with what the
    trained float/QAT model computed.

    Averaging preserves scale (mean of same-scale quantities is still at
    that scale), so this is safe to call directly on already-quantized
    int8/int32 values without re-deriving a scale afterward."""
    n, c, h, w = x.shape
    oh, ow = output_size
    x64 = x.astype(np.int64)
    out = np.empty((n, c, oh, ow), dtype=np.int64)
    for oy in range(oh):
        y0, y1 = (oy * h) // oh, -(-(oy + 1) * h // oh)
        for ox in range(ow):
            x0, x1 = (ox * w) // ow, -(-(ox + 1) * w // ow)
            window = x64[:, :, y0:y1, x0:x1]
            out[:, :, oy, ox] = _round_div(window.sum(axis=(2, 3)), (y1 - y0) * (x1 - x0))
    return out
