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

**As of 2026-08-23 only `esp32_fps` carries these flags.** `esp32_rgb_cnn` and
`esp32_classifier` do not, so a panic on either of those is again invisible over
USB. Add them before investigating any crash on those firmwares — otherwise the
first hour goes into rediscovering that serial is silent.

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

### Why setting the level is enough

The fix only works if the level test happens *before* the expensive part, so it
is worth knowing where it sits. `ESP_LOGx` expands (see `esp_log.h:402` in the
installed framework) to:

```
ESP_LOGE(tag, ...) -> ESP_LOG_LEVEL_LOCAL -> compile-time LOG_LOCAL_LEVEL check
                   -> ESP_LOG_LEVEL       -> esp_log_write(...)
                                          -> esp_log_writev(...)
                                             |
                                             +-- RUNTIME per-tag level check HERE
                                             |
                                             +-- vprintf(...)  <- the expensive part
```

The per-tag runtime filter lives inside `esp_log_writev`, upstream of the
`vprintf` call. The backtrace corroborates this directly: `esp_log_writev`
appears on the stack *calling* `vprintf`, so the filter it consults is
necessarily earlier in the same function.

With the tag set to `ESP_LOG_NONE` that check fails and `esp_log_writev` returns
immediately. No format-state frame, no VFS write, no recursive lock, no lazily
created mutex, no blocking UART transmit. The whole chain below the filter is
simply never entered — which is why one line fixes both the stack overflow and
the multi-millisecond stall.

`esp_log_level_set()` registers a tag whether or not it has been used yet, so
ordering relative to first use does not matter. Setting it before
`esp_camera_init()` is only to close the window during init itself, when
`cam_task` is created.

### What is given up

The `FB-OVF` warning is gone, and it was a real signal: it meant the consumer
was not keeping up. That is an acceptable trade here for two reasons.

First, the condition it reported is **recoverable** — the driver drops a frame
and continues. Losing a frame is invisible; losing the board is not.

Second, the same condition is **already visible somewhere better**. When the
consumer falls behind, `frame age` climbs and `fps` collapses in the status
readout, both of which are on screen and neither of which can crash the board.
The log was the worst available instrument for a condition that is already
instrumented properly.

If the raw driver warnings are ever needed again for a specific investigation,
raise the level temporarily *and* be aware that doing so re-arms the crash:

```c
esp_log_level_set("cam_hal", ESP_LOG_WARN);   // re-arms the cam_task stack overflow
```

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

So the PC↔router link is healthy and the router↔ESP32 link is not.

(The band the PC was on was not recorded for that table. Given the sections
below it should have been, and the comparison is worth redoing on both.)

### Signal strength is NOT the cause — the −28 dBm control

For a while the working theory was link margin: stalls appeared to track RSSI,
the same board swinging between 22 fps and 1 fps as RSSI moved a few dB, which
fits throughput being steeply nonlinear near the sensitivity cliff.

**That theory is disproved.** Tested 2026-08-24 with the board sitting *next to
the router* at **RSSI ≈ −28 dBm** — effectively the top of the range, nothing
left to gain. The stream still slowed badly whenever the viewing computer was on
2.4 GHz. Moving the board barely mattered.

Consequences, both of which reverse earlier advice in this file:

- **An external antenna is not worth buying.** It buys link margin, and link
  margin is already maxed out. This was recommended twice before the control was
  run; the control says it would have changed nothing.
- **The earlier RSSI correlation was not causal.** RSSI was co-varying with
  something else — most likely which band the viewer happened to be on, or what
  else was using the channel at the time. RSSI is still worth watching as a
  *symptom*, but it is not the lever.

What survives is a variable that has nothing to do with signal: **the band the
viewer is connected to.**

### The viewer's band matters, because the board is 2.4 GHz only

Observed 2026-08-24: the stream is noticeably better with the viewing computer
on the router's **5 GHz** SSID and noticeably worse on **2.4 GHz** — same board,
same position, same firmware.

The ESP32-S3 has no 5 GHz radio, so its own link is fixed. What is *not* fixed is
how many times each frame has to cross the 2.4 GHz medium:

| viewer on | 2.4 GHz transmissions per frame | contention |
|---|---|---|
| 5 GHz | 1 — board → router | the router → PC leg is on a different radio |
| 2.4 GHz | 2 — board → router → PC | both legs contend with each other |

Putting the viewer on 2.4 GHz doubles the airtime each frame costs *and* makes
the two halves of its own path compete, which is why the loss is much worse than
a factor of two.

The **802.11 performance anomaly** compounds it: CSMA grants equal transmit
*opportunities*, not equal airtime. The ESP32 is a single-stream HT20 station
that rate-drops hard as RSSI falls, so every frame it sends holds the channel far
longer than a modern laptop's. On a shared 2.4 GHz channel the slow station drags
the fast ones down and is dragged down in return.

### Why SoftAP was slower, not faster

Tried 2026-08-24. On paper it should win — it removes the router hop, so one
transmission per frame instead of two. It lost anyway:

- it **forces the viewer onto 2.4 GHz**, giving up the escape route that turned
  out to be the biggest lever available;
- the board now runs AP duties (beacons, association) on the same single radio it
  is streaming from;
- `SOFTAP_CHANNEL` is a fixed compile-time guess (6) against a router that picks
  its channel adaptively;
- a client on an internet-less network power-saves aggressively, and a sleeping
  client forces the AP to buffer — the board cannot override that, only its own
  power save, which station mode already disables;
- `/status` reports `rssi 0` in AP mode, because `WiFi.RSSI()` describes a
  station association the board no longer has. The one number that predicts
  stalls goes blank exactly in the mode being evaluated. Fixable via
  `esp_wifi_ap_get_sta_list()`, which carries per-station RSSI; not done.

Expected ordering, worth confirming with ping rather than by eye:
**viewer on 5 GHz > SoftAP ≳ viewer on 2.4 GHz.**

SoftAP is still the right answer for a 2.4-GHz-only *viewer* — the planned Cheap
Yellow Display, which has no 5 GHz option either. There it is genuinely one hop
instead of two.

### Levers, in measured order

- **Put the viewing computer on 5 GHz, or on ethernet.** Largest effect found so
  far, costs nothing, needs no firmware change. Ethernet is strictly better
  again: zero 2.4 GHz transmissions on the return leg.
- **Lower JPEG quality** — demonstrated to help, both directly (fewer bytes) and
  via the chain above (shorter sends → buffers returned in time).
- **2.4 GHz router channel** — RSSI measures signal, not interference. RSSI is
  already maxed out and throughput still is not, which is the textbook signature
  of congestion. Untested so far.
- **SoftAP** (`#define SOFTAP` at the top of `main.cpp`) — measured slower than a
  5 GHz viewer; see above. Keep it for the CYD case.
- ~~**External antenna**~~ — **withdrawn.** Disproved by the −28 dBm control
  above. (If it is ever revisited for a different reason: only applies to a
  `WROOM-1U` with a u.FL connector, and on boards carrying both a trace antenna
  and a connector a 0 Ω resistor selects between them, so plugging one in without
  moving that resistor makes things worse.)

### Still open: airtime, or the PC's own 2.4 GHz radio?

The −28 dBm control narrows the cause to two candidates, which make opposite
predictions:

| | mechanism | predicts |
|---|---|---|
| **H1 airtime** | double hop on one channel, plus the ESP32 being a slow station at *any* RSSI (1×1 HT20, conservative rate control) | *any* 2.4 GHz viewer is slow |
| **H2 the PC's radio** | laptop Wi-Fi/Bluetooth combo chips share one 2.4 GHz antenna and time-slice; an active BT device craters 2.4 GHz latency and leaves 5 GHz untouched | *other* 2.4 GHz viewers are fine |

H2 fits the evidence just as well as H1 and is independent of RSSI, which is why
the −28 dBm result does not distinguish them.

**The discriminating test:** stream to a phone on 2.4 GHz with the laptop off the
network entirely. Smooth ⇒ H2, and the fix is local and free. Stuttering ⇒ H1,
and 5 GHz or ethernet for the viewer is the answer.

Supporting tests on the PC, on 2.4 GHz:

```bash
rfkill block bluetooth                        # coexistence test
iw dev <iface> get power_save
sudo iw dev <iface> set power_save off
ping -c 40 -i 0.3 <board-ip>                  # with NO stream running
```

That last one separates "the medium is impaired" from "the medium is saturated".
Bad *idle* ping on 2.4 GHz means no firmware change can help. Clean idle ping
that collapses under stream load means the levers are bitrate-side — JPEG
quality, frame size, or raw RGB565 instead of an encode.

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
