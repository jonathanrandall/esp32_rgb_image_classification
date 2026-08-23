# The streaming stalls and the resets are the same fault

Diagnosed 2026-08-23 on `esp32_rgb_cnn`. Applies to `esp32_classifier` and
`esp32_fps` too — same camera driver, same trigger.

For a long time these looked like two unrelated problems: the video would stall
for a second or more, and separately the board would occasionally reset. They
are one chain, and the reset is the stall escalating.

## The chain

1. **The Wi-Fi send stalls.** Frame ages up to **1.5 s** were measured while the
   device kept computing a frame every ~36 ms. The work is getting done; the
   bytes are not getting out.
2. **The stream loop stops returning frame buffers**, because it is blocked
   inside `httpd_resp_send_chunk()` and cannot reach `esp_camera_fb_return()`.
3. **The camera driver runs out of free buffers** and logs the overflow —
   `cam_hal.c:196`.
4. **That log call overflows `cam_task`'s stack** and the board panics.

Step 4 is the surprising one. `ESP_LOG` from `cam_task` drags in the whole
newlib `vprintf` machinery, which allocates a mutex and enters a critical
section. `cam_task` has nowhere near enough stack for it. **The act of reporting
the problem is what kills the board.**

## The evidence

`reset_reason` in `/status` read `panic`, not `brownout` — which ruled out the
power-supply theory that a capacitor would have addressed, and ruled out
disabling the brownout detector as pointless.

The backtrace only became visible after enabling USB serial. On the Freenove
board definition `ARDUINO_USB_CDC_ON_BOOT=0`, so `Serial` goes to UART0 and
`/dev/ttyACM0` returns nothing — the one piece of evidence that identifies a
crash was being thrown away. Adding these two flags to `platformio.ini` is what
made the diagnosis possible at all:

```ini
build_flags =
    -DARDUINO_USB_CDC_ON_BOOT=1
    -DARDUINO_USB_MODE=1
```

The panic itself:

```
Guru Meditation Error: Core 0 panic'ed (Unhandled debug exception).
Debug exception reason: Stack canary watchpoint triggered (cam_task)
```

Decoded with the matching ELF (`sha256` in the panic output must match the
build — it did):

```
xtensa-esp32s3-elf-addr2line -pfiaC -e .pio/build/freenove_esp32_s3_wroom/firmware.elf <addrs>
```

Read bottom-up:

```
cam_task                    cam_hal.c:196
  __wrap_esp_log_write      esp_diagnostics_log_hook.c:469
  esp_log_writev            log.c:200
  vprintf                   vprintf.c:34
  _vfprintf_r               vfprintf.c:1781
  __sfvwrite_r              fvwrite.c:251
  _fflush_r / __sflush_r    fflush.c
  __swrite                  stdio.c:94
  esp_vfs_write             vfs.c:431
  console_write             vfs_console.c:73
  uart_write                vfs_uart.c:208
  _lock_acquire_recursive   locks.c:167
  lock_init_generic         locks.c:73
  xQueueCreateMutex         queue.c:564
  ...spinlock, stack gone
```

## The fix

Stop the driver logging from that task. Set **before** `esp_camera_init()` so it
also covers logging during init:

```c
esp_log_level_set("cam_hal", ESP_LOG_NONE);
esp_log_level_set("s3 ll_cam", ESP_LOG_NONE);
esp_log_level_set("camera", ESP_LOG_NONE);
```

Applied to `esp32_rgb_cnn`, `esp32_fps` and `esp32_classifier`.

**It fixes the stalls too, not just the crashes.** This was not predicted and
is worth understanding, because the mechanism is a feedback loop:

1. `vprintf` writes to UART0 at 115200 baud, and **blocks** while the TX FIFO
   drains — whether or not anything is attached, which on this board it is not.
   A line like `E (12345) cam_hal: FB-OVF` is ~30 characters, about **2.6 ms**.
2. It fires **per dropped frame**. During a stall the sensor keeps producing at
   ~25 fps while the consumer is blocked in `httpd_resp_send_chunk()`, so every
   one of those frames finds no free buffer and logs.
3. While `cam_task` sits in that blocking write it is **not servicing the camera
   DMA**, so more frames are missed, which produces more log lines, which blocks
   it longer.

A brief link hiccup that should have cost a handful of dropped frames instead
amplified itself into a long visible stall. Silencing the tag breaks the loop:
the overflow still happens, and handling it is now what it always should have
been — drop the frame, move on, microseconds.

**What it still does not do is improve the radio.** The link is unchanged (see
below). What is gone is the amplifier that turned small hiccups into large ones.
Expect "much better, but still sensitive to RSSI", not "fixed".

## Corroboration

Lowering JPEG quality reduced **both** the stalls and the resets. That falls
straight out of the chain: fewer bytes per frame → shorter sends → buffers
returned in time → no overflow → no log → no crash. It was the observation that
first suggested the two symptoms were connected rather than coincidental.

## The underlying link problem, which is NOT fixed

Measured on the house network, same machine:

| target | avg RTT | max RTT | loss |
|---|---:|---:|---:|
| the board | 368 ms | 1,722 ms | 2.5% |
| the gateway | 5.7 ms | 15 ms | 0% |

So the PC↔router link is healthy and the router↔ESP32 link is not. Stalls track
RSSI: the same board swings between 22 fps and 1 fps as RSSI moves a few dB,
because throughput against RSSI is steeply nonlinear near the cliff.

Levers, in order of expected effect:

- **SoftAP** (`#define SOFTAP` at the top of `main.cpp`) — removes the router
  hop, the shared channel, and every other device competing for airtime. Cost:
  the board becomes its own network with no internet, so the viewer must join
  it. A laptop with ethernet sidesteps that.
- **External antenna** — only if the module is a `WROOM-1U` with a u.FL
  connector. On boards carrying both a trace antenna and a connector, a 0 Ω
  resistor selects between them; plugging in an antenna without moving it can
  make things worse.
- **Lower JPEG quality** — already demonstrated to help, and a legitimate lever
  to keep.
- **Router channel** — RSSI measures signal, not interference. If RSSI is good
  and throughput is not, that is congestion, and only a channel change helps.

## Diagnostic notes

**The on-page readout is the cheapest instrument.** In the FPS line:

```
FPS: 18.4   [frame age 55ms | rssi -58dBm | reloads 2 | resets 0]   last reset: panic
```

- `frame age` — ms since a frame left the device. Climbing past 8 s trips the
  watchdog reload.
- `rssi` vs `frame age` together separate a link stall from a device stall.
- `resets` counts `signal.frames` going *down*, which can only happen across a
  reset.
- `last reset:` appears only when the reason was not `poweron`/`external`, so it
  is quiet in the ordinary case and loud when it matters.

**`reset_reason` is worth reading before spending money.** It said `panic`,
which took a capacitor, a power supply and the brownout detector off the table
in one step.

**Do not diagnose the stream with your own `curl` to `:81`.** The board serves
one client; your connection takes the slot and reproduces the symptom you are
investigating.
