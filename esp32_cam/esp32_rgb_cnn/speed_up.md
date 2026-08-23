# The 611-Millisecond Fix

ESP32-S3 · RGB pixel CNN · performance log

Inference on the RGB CNN went from 6,828 ms to 466 ms — 14.7×. Slightly less
than half of that came from turning ESP-NN on. Slightly more than half came
from deleting a memory-layout bug that had nothing to do with the neural
network at all.

## Summary

Same weights, same architecture, all three builds. 14.7× end to end.

| build                      | inference   | fps  | vs previous |
|-----------------------------|------------:|-----:|------------:|
| portable C, no acceleration | 6828.30 ms  | 0.15 | —           |
| ESP-NN, CHW layout          | 1083.87 ms  | 0.90 | 6.3×        |
| **ESP-NN, NHWC layout**     | **465.75 ms** | **2.00** | 2.33× |

## 1. Make sure ESP-NN is actually running

The first working build measured 6,828 ms and the obvious question was
whether the vendor-accelerated int8 kernels were even in the binary. They
weren't — the first exported model header carried no ESP-NN code path at
all, so the acceleration flags inherited from an earlier project were
configuring nothing. That was fixed by exporting a header with a real
ESP-NN branch, which alone brought inference to 1,083.87 ms (6.3×).

That claim was then checked against the linked binary rather than trusted
from build flags, twice over the course of the work: `nm` and `objdump`
confirmed direct calls into the esp32s3 assembly kernels
(`esp_nn_conv_s8_esp32s3`, its im2col and aligned-assembly sub-paths, the
quantized-multiply assembly routine), and confirmed that the one portable
fallback symbol present in the binary (`esp_nn_conv_s8_ansi`) is called
only from a code branch this model's convolutions never take. Cheap to
check, and worth checking before quoting any number downstream of it.

## 2. Delete the CHW↔NHWC repack

1,083.87 ms for 74.65 million MACs is roughly 69 MMAC/s, poor for this
silicon's SIMD. Splitting the accelerated convolution wrapper into its
three actual parts — repack the input, run the kernel, repack the output —
found why: the model stored its activations channel-planar (CHW),
inherited from an earlier project where every activation was a few
kilobytes and the format didn't matter. ESP-NN requires channel-last
(NHWC). So every one of the four convolutions was transposing its input
into a 307 KB scratch buffer, running the kernel, and transposing the
output back — work that computes nothing the model needs, and that a
per-layer timer silently bills to the convolution.

| layer | repack in | kernel | repack out | total   | overhead |
|-------|----------:|-------:|-----------:|--------:|---------:|
| conv0 |      4.88 | 198.10 |      91.94 |  294.92 |    32.8% |
| conv1 |    149.86 | 130.77 |      86.83 |  367.45 |    64.4% |
| conv2 |    146.79 |  81.82 |      41.17 |  269.78 |    69.7% |
| conv3 |     71.74 |  54.85 |      17.67 |  144.25 |    62.0% |
| **total** | **611 ms repack** | | **466 ms kernel** | | **57%** |

The exporter was rewritten to emit activations in NHWC end to end, and the
repack buffers — 614 KB of PSRAM — were deleted entirely. The rebuilt
firmware landed at **465.75 ms**, matching the 466 ms the repack-excluded
kernel column had predicted, and its per-layer kernel rates (42.0, 168.9,
280.0, 418.0 MMAC/s) came back within measurement noise of that column
(41.9, 169.1, 270.3, 403.3). On-device logits were bit-identical to the CHW
build across every test vector, confirming the rewrite changed nothing
about what the model computes.

This cost is specific to models with large activations. The project this
was inherited from has activations a few kilobytes in size, where a format
mismatch is free. Here the largest activation is 307 KB, where the same
mismatch was more than half the runtime.

> **These numbers are a floor, not a ceiling.** ESP-NN's int8 convolution
> dispatches to one of several sub-kernels by shape. This model's layers
> split across two of them — conv0 (3 input channels) takes an im2col
> path, conv1–3 take a general aligned-assembly path — and neither is the
> library's dedicated 3×3 kernel, the natural fit for a model where every
> convolution is 3×3. That kernel exists in the vendor library but is
> compiled out at the version in use, with the authors' own comment citing
> an unresolved performance regression. So the 466 ms figure reflects what
> this library's general-purpose paths can do on this hardware, not what
> the silicon is capable of. Worth rechecking against future library
> releases.

## Where more speed could come from

Ranked by how confident I am each one pays off, most to least.

### 1. Reduce the model's input resolution — *high confidence*

MACs scale with pixel count, and every layer runs on the full 160×120
frame. At 96×96 (2.08× fewer pixels), carrying the measured per-layer
rates forward:

| layer | MACs @96² | rate (measured) | projected |
|-------|----------:|-----------------:|----------:|
| conv0 |   3.98 M | 42.0 MMAC/s  | ~95 ms |
| conv1 |  10.62 M | 168.9 MMAC/s | ~63 ms |
| conv2 |  10.62 M | 280.0 MMAC/s | ~38 ms |
| conv3 |  10.62 M | 418.0 MMAC/s | ~25 ms |
| **total (projected)** | **35.84 M** | — | **~224 ms, ~4.3 fps** |

Roughly double the current speed. Two things worth deciding before
committing to it: whether the capture path stays "native resolution in, no
resize" (by capturing at the sensor's square 96×96 mode directly) or
introduces an on-device resize that has to match however the training
data was downsampled — the first option is simpler and was the reason
160×120 was chosen originally. And this exact sensor mode has a known
blank-frame quirk on this hardware, already worked around by an existing
init pattern (start the camera larger, then switch down), so it shouldn't
need rediscovering. Smaller activations would also likely let one more
buffer (`conv1_out`) fit in fast internal memory where it currently
doesn't — a free side effect, testable once implemented. This is a model
change, not a firmware one, and accuracy at lower resolution is a training
decision.

### 2. Attack conv0 specifically — *moderate confidence, real effort*

conv0 is 42% of inference time for 11% of the model's MACs at any
resolution — a channel-width effect (only 3 input channels to vectorize
across), not a resolution effect, so it persists even after fix 1. Two
ways to address it, both experimental: reactivate the vendor library's
disabled 3×3 kernel path and re-verify correctness from scratch, or write
a small hand-tuned kernel for the fixed 3-channel case. Worth trying if
more speed is needed specifically; the vendor library's own unfinished
attempt at roughly this is a mild signal it isn't a quick win.

### 3. Drop the streaming JPEG encode for pure-inference numbers — *high confidence, small*

~18 ms/frame, about 4% of the current total. Already toggleable at runtime
without a reflash. Worth using when the number that matters is inference
alone rather than the full streaming pipeline.

### 4. Split work across both CPU cores — *speculative*

The chip has two cores and everything currently runs on one. Splitting
output channels across both for the larger layers is possible in
principle, but both cores share the same PSRAM bus that the larger
activations already live on, so a compute-bound layer could just become a
bandwidth-bound one instead. Would need a prototype to know either way —
not a change to make on the strength of the theory alone.

## Where this leaves things

For this architecture, these weights, on this vendor library: close to the
practical floor. The two fixes above were unambiguous bugs with clear
payoff — the kind worth finding regardless of what happens next. What
remains is either a genuine model decision (input resolution, which needs
a retrain) or work with uncertain or modest return (conv0's specific
kernel, dual-core). Reducing resolution is the only lever on this list
with both high confidence and a large expected effect.

---

Freenove ESP32-S3 WROOM (8 MB flash / 8 MB OPI PSRAM), 240 MHz,
arduino-esp32. Model: 4-conv int8 RGB CNN, 160×120 planar input, 17
classes, 74.65 M MACs. 14 August 2026.

---

## Re-measured 2026-08-23 — conv0 on the widened `32,32,64` model

Different model from the rest of this document: 5 classes, `--conv-channels
32,32,64`, and a **32x24** input (5x5 block mean of a 160x120 capture) rather
than 160x120 pixels. The conv0 finding survives the change, which is the point
— it is a channel-width effect, not a resolution one.

**First, a warning about the on-device benchmark output.** `run_inference_benchmark()`
prints hardcoded shapes and MAC counts:

```c
Serial.printf("  conv0 %7.2f ms  (3->16ch, 120x160, 8.29M MAC) %7.1f MMAC/s\n", ...
```

Those are the OLD model's numbers. The MMAC/s column is computed by dividing a
hardcoded MAC count by a measured time, so on any retrained model **the
milliseconds are real and the MMAC/s figures are fiction**. Reading them
directly gave "561 MMAC/s for conv0 against 6,161 for conv3", which is wrong by
more than an order of magnitude. Recompute from the manifest's actual shapes.

Correct figures for the shipped model:

| layer | in->out | out WxH | MACs | % MACs | ms | % time | MMAC/s |
|---|---|---|---:|---:|---:|---:|---:|
| conv0 | 3->32 | 32x24 | 663,552 | 15.8% | 14.77 | **48.4%** | **44.9** |
| conv1 | 32->32 | 16x12 | 1,769,472 | 42.1% | 7.40 | 24.2% | 239.1 |
| conv2 | 32->64 | 8x6 | 884,736 | 21.1% | 4.52 | 14.8% | 195.7 |
| conv3 | 64->32 | 8x6 | 884,736 | 21.1% | 3.59 | 11.8% | 246.4 |
| | | | **4,202,496** | | **30.52** | | |

**conv0 burns 48.4% of inference time to do 15.8% of the arithmetic**, running
at 44.9 MMAC/s against roughly 240 for every other layer — a **5.3x** efficiency
gap.

### Why

`in_channels = 3`. ESP-NN's esp32s3 conv dispatcher picks its path from the
layer shape, and a 3-channel input routes to the im2col path rather than the
aligned-assembly one the deeper layers get. Three channels is also simply too
few to vectorize across: the kernel's inner product is over `kh * in_channels`,
which is 9 here against 288 for conv1.

Note this is **not** the ESP-NN filter-cache cliff (`out_ch * in_ch` under
~3,600). conv0's product is 96, the smallest in the model. It is comfortably
inside the cache and still the slowest layer per MAC — a different mechanism
entirely, and worth not confusing with it.

### What follows

**Widening the first stage is unusually expensive here.** Going 16 -> 32
channels on conv0 doubled its MACs, but they are the *least efficient* MACs in
the model. It bought about 2 accuracy points, which may well be worth it — but
the cost is not proportional to the MAC count, and the same 2 points bought in
conv1 or conv2 would have cost roughly a fifth of the time.

**If inference needs to be faster, conv0 is where the headroom is**, not the
deeper layers. conv1-3 together already run near 240 MMAC/s and are unlikely to
improve much. Bringing conv0 to that rate would cut total inference from 30.5 ms
to about 18.5 ms — a 39% reduction from one layer.

Options, unchanged in character from section 2 above and still not quick wins:

1. **Reactivate the vendor library's disabled 3x3 kernel.**
   `esp_nn_conv_s8_3x3_opt_esp32s3.c` exists but its only call site is wrapped
   in `#if 0` with the comment "TODO: fix inline asm priming + performance
   regression before enabling". Nominally the faster path for exactly this
   shape. Would need correctness re-verified from scratch.
2. **Pad the input to more channels.** Tempting and probably wrong: padding 3 ->
   16 multiplies conv0's MACs by 5.3x, and its per-MAC rate would have to
   improve by more than that to break even. Worth a measurement before
   dismissing, but the arithmetic is not encouraging.
3. **Hand-write a 3-channel kernel.** The input is 3 channels and always will
   be; a specialised kernel does not need to be general. The vendor library's
   own unfinished attempt at roughly this is a mild signal it is not easy.

**Cheapest real lever, if accuracy allows it:** conv0's cost scales with its
*output* grid, which is the model input size. It runs stride 1, so it does full
work at full resolution before anything downsamples. Making conv0 stride 2 — or
feeding a smaller grid, e.g. 8x8 block mean instead of 5x5 — attacks it
directly, at a known accuracy cost measured in the reduction sweep.
