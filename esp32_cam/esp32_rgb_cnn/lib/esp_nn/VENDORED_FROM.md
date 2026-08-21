# Vendored from espressif/esp-nn

Source: https://github.com/espressif/esp-nn
Commit: `10b6c0fc884a3b05f94a752f91d00ebadfe5d8d0` (2026-07-29)

Per `esp32_nn_instructions.md`: vendored directly into `lib/` (PlatformIO
auto-builds anything under `lib/`) instead of switching
`platformio.ini` to `framework = espidf, arduino` to pull this in via the
component manager -- not worth restructuring an already-working,
verified `arduino`-framework project just to get a dependency manager.

**Scope: esp32s3 int8 conv2d only**, not the whole library. This project's
board is exactly an ESP32-S3, so the library's multi-chip dispatch
(esp32, esp32p4, generic) is dead weight; depthwise conv, pooling,
activation, softmax, fully-connected, and int16 variants aren't needed
for the current standalone `extra_conv0` prototype
(`esp32_nn_instructions.md` Step 2) and weren't vendored either -- add
them later, from the same commit, if the prototype proves out and scope
grows.

## What's here and why

- `include/esp_nn.h`, `esp_nn_defs.h` -- top-level dispatch header +
  shared struct definitions (`data_dims_t`, `conv_params_t`,
  `quant_data_t`).
- `include/esp_nn_ansi_headers.h`, `esp_nn_ansi_c.h` -- portable C
  reference declarations (`esp_nn_conv_s8_ansi` etc.) and the
  `esp_nn_conv_s8 -> esp_nn_conv_s8_ansi` macro dispatch used when
  `CONFIG_NN_OPTIMIZED` isn't defined. Kept specifically so the
  correctness-comparison harness can build and run on host `gcc` too
  (the esp32s3-optimized path is real xtensa assembly, host-unbuildable)
  before trusting anything on real hardware.
- `include/esp_nn_esp32s3.h` -- esp32s3-optimized declarations + the
  `esp_nn_conv_s8 -> esp_nn_conv_s8_esp32s3` macro dispatch used when
  `CONFIG_NN_OPTIMIZED` + `CONFIG_IDF_TARGET_ESP32S3` are both defined
  (set via `platformio.ini` `build_flags`).
- `src/common/common_functions.h` -- shared inline helpers
  (`esp_nn_multiply_by_quantized_mult` etc., portable C -- used by the
  ansi reference path).
- `src/common/esp_nn_common_functions_esp32s3.S`,
  `esp_nn_dot_s8_esp32s3.S`, `esp_nn_multiply_by_quantized_mult_esp32s3.S`
  -- xtensa assembly primitives the esp32s3 conv kernels below call into
  (im2col dot-product, s8->s16 widening, the esp32s3 asm requantize path)
  -- included because they're referenced unconditionally at link time by
  `esp_nn_conv_esp32s3.c`'s dispatcher / `esp_nn_conv_s8_mult8_1x1_esp32s3.S`,
  even though this project's `extra_conv0` shape (3x3, in_ch=out_ch=64)
  doesn't hit their runtime code paths.
- `src/convolution/esp_nn_conv_ansi.c` -- portable C reference conv2d
  (`esp_nn_conv_s8_ansi`), used both as the `CONFIG_NN_OPTIMIZED=0`
  fallback and as the grouped-conv fallback the esp32s3 dispatcher itself
  calls when `in_channels != filter_channels`.
- `src/convolution/esp_nn_conv_esp32s3.c` -- the esp32s3 dispatcher
  (`esp_nn_conv_s8_esp32s3`): picks between a 1x1-specific path, an
  im2col path for small `filter_width * in_channels`, and the general
  aligned-assembly path -- also *contains* (not separate files)
  `esp_nn_conv_s8_im2col_s3` and `esp_nn_aligned_s8_pad_asymmetric`.
- `src/convolution/esp_nn_conv_s8_1x1_esp32s3.c`,
  `esp_nn_conv_s8_mult8_1x1_esp32s3.S` -- the two 1x1-conv code paths the
  dispatcher can call. Not this prototype's actual runtime path
  (`extra_conv0` is 3x3), vendored anyway because the dispatcher
  references them unconditionally at compile time -- the linker needs
  them resolved regardless of which branch executes at runtime.
- `src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.S`
  -- **this is the actual kernel `extra_conv0` will exercise**: the
  general-purpose padded/aligned assembly conv path, selected by the
  dispatcher for any conv that isn't 1x1 and isn't small enough for the
  im2col path.

## Deliberately NOT vendored

- `esp_nn_conv_s8_3x3_opt_esp32s3.c` (and its declared-but-unused externs
  `esp_nn_conv_s8_3x3_can_use`/`esp_nn_conv_s8_3x3_opt`) -- the *only*
  call site in `esp_nn_conv_esp32s3.c` is wrapped in `#if 0`, with a
  source comment: "TODO: fix inline asm priming + performance regression
  before enabling." Dead code at this commit; not linked, not needed.
  Worth rechecking in a future esp-nn version in case it gets re-enabled
  upstream -- it's nominally the more specific/faster path for exactly
  this project's 3x3 kernels.
- Depthwise conv (all `esp_nn_depthwise_conv_*` files, and the
  `esp_nn_multiply_by_quantized_mult_ver1_esp32s3` symbol they alone use)
  -- this model has no depthwise-separable layers.
- `esp_nn_esp32p4.h`, `esp_nn_generic_opt.h`, anything `*_esp32p4.*` --
  other chip targets.
- Pooling, activation functions, softmax, logistic, fully-connected,
  `esp_nn_mean_*` -- other operators, out of scope for the `extra_conv0`
  conv2d-only prototype.

## Build flags required

`platformio.ini` needs both of these for `esp_nn.h` to select the
esp32s3-optimized path instead of silently falling back to `esp_nn_conv_s8_ansi`:

```
build_flags =
    -DCONFIG_NN_OPTIMIZED=1
    -DCONFIG_IDF_TARGET_ESP32S3=1
```
