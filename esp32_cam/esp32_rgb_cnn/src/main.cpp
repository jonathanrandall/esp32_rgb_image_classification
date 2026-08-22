#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_camera.h"
#include "img_converters.h"  // frame2jpg() -- see the streaming note in stream_handler()
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"  // heap_caps_malloc -- model buffers, see alloc_model_buffers()
#include "lwip/sockets.h"   // setsockopt/TCP_NODELAY on the stream socket, see stream_handler()
#include "esp_wifi.h"       // esp_wifi_get_ps() -- verify power save is really off, see connect_wifi()
#include <math.h>

#include "camera_pins.h"
#include "secrets.h"

// Per-stage profiling hooks. These MUST be defined before the header is
// included: model_weights.h brackets every stage of model_forward() with
// MODEL_PROFILE_BEGIN/END, which default to no-ops, so a build that forgets
// this silently loses the per-layer breakdown rather than failing.
//
// The per-layer split is the point of this firmware, not a nicety. The
// hypothesis under test is that conv0 costs far more than its ~11% share of
// the model's 74.7M MACs -- a claim about memory bandwidth rather than
// compute, which a single aggregate inference time cannot answer either way.
//
// These globals are written by whichever task is inside model_forward(), so
// they are only safe to read under the same mutex that serializes it -- see
// model_forward_locked(), which is the only thing that should call
// model_forward() directly.
static uint32_t g_prof_t0;
static uint32_t g_stage_us[7];
#define MODEL_PROFILE_BEGIN(stage) (g_prof_t0 = (uint32_t)esp_timer_get_time())
#define MODEL_PROFILE_END(stage) (g_stage_us[stage] = (uint32_t)esp_timer_get_time() - g_prof_t0)

// The header exposes conv primitives and weight tables that this firmware
// doesn't call directly (it uses the generated model_forward()), and which of
// them are live depends on whether the ESP-NN branch is compiled -- so
// -Wunused-function fires on the ones the inactive path would have used.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "model_weights.h"
#pragma GCC diagnostic pop

static_assert(MODEL_NUM_STAGES == 7, "g_stage_us must match MODEL_NUM_STAGES");

// True when this binary actually contains the ESP-NN kernels, so every
// reported number carries its own provenance instead of relying on someone
// remembering which build produced it.
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
#define BUILD_HAS_ESP_NN 1
#else
#define BUILD_HAS_ESP_NN 0
#endif

#include "rgb_test_vectors.h"

// WHICH NUMBER COMPARES TO WHICH -- the single easiest thing to get wrong
// when reading anything this firmware reports.
//
// The DCT firmware has two published figures on this same board, and they are
// 14-19x apart because of ESP-NN alone:
//     ESP-NN:   21.9 ms inference,  39.8 ms end-to-end,  23.7 fps
//     portable: 306 ms inference,  ~428 ms end-to-end,   2.3 fps
// Comparing across that boundary measures ESP-NN-vs-no-ESP-NN and buries the
// DCT-vs-RGB architectural difference the whole exercise is about. So: match
// the build. BUILD_HAS_ESP_NN below records which one this binary is, and it
// is reported at boot and in /status so a number can never be separated from
// its provenance.
//
// The portable-vs-portable pairing carries one unresolved caveat: it assumes
// the DCT project's naive path and this header's portable conv2d_int8 are
// comparable-quality implementations. If one has notably tighter loops the
// pairing is contaminated. That is being checked on the PC side and is NOT
// yet confirmed -- do not present it as authoritative until it is.
// Accelerated-vs-accelerated does not share this problem: both sides then run
// the same vendor kernels.

// Device will be reachable at http://esp32cam.local/
#define MDNS_HOSTNAME "esp32cam"
#define STREAM_PORT 81

// esp_http_server runs each instance on a single worker task. The /stream
// handler below never returns while a client is connected, so it must live
// on its own httpd instance (a different port) or it would block the "/",
// "/status" and "/config" handlers from ever being serviced.
static httpd_handle_t camera_httpd = NULL;
static httpd_handle_t stream_httpd = NULL;

// Purely a bandwidth/visual setting in THIS firmware, unlike the DCT firmware
// it was copied from. There, JPEG quality was effectively a model
// hyperparameter: the classifier consumed DCT coefficients parsed straight out
// of the camera's JPEG, so compression directly degraded the model's input.
// Here the model consumes RGB565 pixels and the JPEG exists only to have
// something to show in a browser, so it can be tuned freely for bandwidth
// without touching classification at all.
//
// TWO OPPOSITE SCALES -- do not use one constant for both.
//   sensor->set_quality() / camera_config_t.jpeg_quality: 0-63, LOWER is better.
//   frame2jpg() -> fmt2jpg() -> jpge::params::m_quality:  1-100, HIGHER is better.
// This firmware only hardware-encodes on the DCT side, which is where the 0-63
// value came from; carrying it into frame2jpg() asks the software encoder for
// quality 12 out of 100, which is why the stream looked bad and frames were an
// implausibly small ~1.3KB for 160x120. Kept as two separately named constants
// so the confusion cannot recur.
#define SENSOR_JPEG_QUALITY 12   // 0-63, lower = better (sensor hardware encoder)
#define JPEG_ENCODE_QUALITY 80   // 1-100, higher = better (jpge software encoder)

// Runtime-settable via /config?jq=N so quality/bandwidth/encode-time can be
// swept without a reflash -- encode time is one of the measured pipeline
// stages, and it varies with this.
static volatile int g_jpeg_quality = JPEG_ENCODE_QUALITY;

// ---------------------------------------------------------------------------
// Model buffers
// ---------------------------------------------------------------------------
//
// model_init() (generated, in model_weights.h) allocates every activation
// buffer, plus the NHWC repack pair on ESP-NN builds. It must be called once
// at boot and its return checked -- model_forward() would otherwise call it
// lazily and silently return zeroed logits if allocation failed.
//
// Placement is not uniform and that matters for interpreting the timings:
// each buffer prefers internal SRAM and falls back to PSRAM, allocated in
// descending read-traffic-per-byte order so the hottest small buffers win the
// scarce internal SRAM. conv0_out (307KB) can never satisfy the reserve check
// and therefore always lands in PSRAM -- which is deliberate, since letting it
// compete would only evict smaller, hotter buffers. Where everything actually
// landed is read back via model_buffer_report() and published at boot and in
// /status, because an inference time is not interpretable without it.

// The camera's capture geometry, which is NOT the model's input geometry any
// more. train_rgb_cnn.py's --rgb-block-width/height average each block of
// pixels down to one input value: at the default 8x8 the sensor delivers
// 160x120 and the model consumes 20x15.
//
// An 8x8 block mean is exactly the DCT arm's DC coefficient, so this makes the
// RGB arm an equal-resolution control against the compressed-domain model --
// the comparison the paper's limitations call for. It is also cheap enough to
// belong on the device: one add per pixel and a divide per output value,
// against a resize filter that would not be implementable at sensible cost.
#define CAM_WIDTH  160
#define CAM_HEIGHT 120
#define CAM_PIXELS (CAM_WIDTH * CAM_HEIGHT)

#define BLOCK_W (CAM_WIDTH  / MODEL_RGB_WIDTH)
#define BLOCK_H (CAM_HEIGHT / MODEL_RGB_HEIGHT)

static_assert(CAM_WIDTH  % MODEL_RGB_WIDTH  == 0, "capture width must divide by the model's input width");
static_assert(CAM_HEIGHT % MODEL_RGB_HEIGHT == 0, "capture height must divide by the model's input height");

#define RGB_PIXELS (MODEL_RGB_WIDTH * MODEL_RGB_HEIGHT)
#define SZ_RGB888 (3 * RGB_PIXELS)                       // 57,600

// The camera-facing input buffer. Not one of the model's own buffers: the
// header takes planar RGB888 as an argument and leaves producing it to the
// caller. Allocated AFTER model_init() so it queues behind the model's
// buffers for internal SRAM -- it is written once and read once per frame
// (traffic/byte ~1), so it is the last thing that should win fast memory.
static uint8_t *g_rgb888 = NULL;
static bool g_rgb888_internal = false;

static bool g_bufs_ready = false;

// Set while the boot self-test and benchmark are running. The model mutex
// makes concurrent forward passes *correct*; this makes the boot measurements
// *meaningful*. The web server is started before the self-test (so the device
// stays reachable if something goes wrong), which means a browser can connect
// and start classifying frames while the benchmark runs -- two tasks each
// doing 74.7M MACs would then split the CPU and roughly double both timings,
// and the benchmark's whole purpose is to be an uncontended reference number.
// While this is set the stream keeps capturing, encoding and pushing status;
// it just skips classification until the self-test finishes.
static volatile bool g_selftest_running = false;

// Number of /stream clients currently connected. The boot benchmark claims to
// be an uncontended reference number, and it is only uncontended if nobody is
// streaming: even with classification suspended, the stream task still
// captures and software-JPEG-encodes every frame. Measured difference is
// large -- 465.75 ms with no client versus 789.84 ms with one -- so the
// figure is annotated with the client count rather than left to be quoted out
// of context.
static volatile int g_stream_clients = 0;

static bool alloc_model_buffers() {
  if (!model_init()) {
    Serial.println("FATAL: model_init() failed -- not enough memory for the activation buffers.");
    return false;
  }

  g_rgb888 = (uint8_t *)heap_caps_malloc(SZ_RGB888, MALLOC_CAP_SPIRAM);
  if (g_rgb888) {
    g_rgb888_internal = false;
  } else {
    g_rgb888 = (uint8_t *)heap_caps_malloc(SZ_RGB888, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    g_rgb888_internal = (g_rgb888 != NULL);
  }
  if (!g_rgb888) {
    Serial.println("FATAL: could not allocate the planar RGB888 input buffer.");
    return false;
  }

  Serial.printf("=== Model buffer placement (ESP-NN %s) ===\n",
                BUILD_HAS_ESP_NN ? "ENABLED" : "disabled");
  int n = 0;
  const model_buffer_info_t *info = model_buffer_report(&n);
  size_t internal_total = 0, psram_total = 0;
  for (int i = 0; i < n; i++) {
    Serial.printf("  %-11s %7d bytes  %s\n", info[i].name, info[i].bytes,
                  info[i].internal ? "INTERNAL SRAM" : "PSRAM");
    if (info[i].internal)
      internal_total += (size_t)info[i].bytes;
    else
      psram_total += (size_t)info[i].bytes;
  }
  Serial.printf("  %-11s %7d bytes  %s\n", "rgb888", SZ_RGB888,
                g_rgb888_internal ? "INTERNAL SRAM" : "PSRAM");
  if (g_rgb888_internal)
    internal_total += SZ_RGB888;
  else
    psram_total += SZ_RGB888;

  Serial.printf("  total: %u bytes internal SRAM, %u bytes PSRAM\n", (unsigned)internal_total,
                (unsigned)psram_total);
  Serial.printf("  free after alloc: %u internal, %u PSRAM\n\n",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  g_bufs_ready = true;
  return true;
}

// ---------------------------------------------------------------------------
// RGB565 -> planar RGB888
// ---------------------------------------------------------------------------
//
// BYTE ORDER IS NOT ASSUMED. The two candidate readings of each 2-byte pixel
// differ only in which byte carries the red bits, and a wrong choice swaps R
// and B. That failure is close to invisible: the model still returns
// confident, plausible-looking predictions on swapped input, it is just
// quietly less accurate. So rather than hardcoding a convention, both
// readings are implemented, the active one is runtime-switchable via
// /config?swap=0|1 (no reflash), and /status continuously reports the mean
// R/G/B the frame would produce under BOTH readings. Point the camera at
// something strongly and unambiguously red and take the reading whose "r"
// mean is the large one -- that is the correct setting, empirically
// determined on this specific driver and sensor rather than assumed.
//
// swap=false: pixel = src[2i] << 8 | src[2i+1]   (high byte first)
// swap=true:  pixel = src[2i+1] << 8 | src[2i]   (low byte first)
//
// MEASURED, not assumed: swap=false. The initial default here was true, on a
// handover note that this driver's RGB565 output is byte-swapped per pixel.
// The roughness discriminator above refuted that on hardware by a wide margin
// -- 4.25 vs 36.47 mean neighbour difference, an 8.6x gap, on an ordinary
// indoor scene. So esp32-camera hands back the HIGH byte first and
// v = src[2i] << 8 | src[2i+1] is correct.
//
// This was not a cosmetic bug. With swap=true the model was being fed pixels
// whose red channel was built from green's low bits and blue's high bits, and
// it went on producing confident, stable, entirely meaningless predictions --
// exactly the silent failure the two-hypothesis diagnostic exists to catch.
static volatile bool g_rgb565_swap = false;

// SEPARATE, INDEPENDENT QUESTION from byte order: is the 5-bit field at the
// top of the 16-bit pixel red or blue? A sensor emitting BGR565 would leave
// the byte order correct and still hand the model transposed planes.
//
// The roughness test above CANNOT see this. Transposing two planes permutes
// the per-channel roughness values without changing any of them, so the
// statistic is identical either way -- confirmed PC-side at 9.41 vs 9.41 over
// 425 images. Neither can the embedded real-image self-test vectors: they are
// fed as ready-made planar arrays and bypass the camera path entirely, which
// is precisely where such a swap would live.
//
// What does discriminate is a distributional property of natural scenes: they
// skew warm. Over the 425-image training set, mean(R) - mean(B) has median
// +14.5 and is positive for 80.2% of images. So a running average of that
// quantity over many live frames is the test -- a persistently NEGATIVE value
// (below about -6) is strong evidence the planes are transposed. It must be
// averaged, not judged per frame: any single frame can legitimately be
// blue-dominant.
static volatile bool g_rb_swap = false;

// Whether to software-JPEG-encode and send image frames. See the streaming
// discussion in stream_handler(); /config?jpeg=0 turns encoding off so the
// capture+convert+inference pipeline can be measured without the encode
// stage, while the status JSON keeps flowing on the same connection.
static volatile bool g_stream_jpeg = true;

static void rgb565_to_planar_rgb888(const uint8_t *src, uint8_t *dst, bool swap) {
  // Plane 0 is what the model calls R. If the sensor is really BGR565 the two
  // outer planes are written the other way round; see g_rb_swap.
  uint8_t *R = g_rb_swap ? dst + 2 * RGB_PIXELS : dst;
  uint8_t *G = dst + RGB_PIXELS;
  uint8_t *B = g_rb_swap ? dst : dst + 2 * RGB_PIXELS;

#if BLOCK_W == 1 && BLOCK_H == 1
  for (int i = 0; i < RGB_PIXELS; i++) {
    uint8_t b0 = src[2 * i], b1 = src[2 * i + 1];
    uint16_t v = swap ? (uint16_t)(((uint16_t)b1 << 8) | b0) : (uint16_t)(((uint16_t)b0 << 8) | b1);
    uint8_t r5 = (uint8_t)((v >> 11) & 0x1F);
    uint8_t g6 = (uint8_t)((v >> 5) & 0x3F);
    uint8_t b5 = (uint8_t)(v & 0x1F);
    // Bit replication, not a shift: (r5<<3) alone would cap white at 248 and
    // bias every channel dark. Replicating the high bits into the low ones
    // maps 31 -> 255 exactly.
    R[i] = (uint8_t)((r5 << 3) | (r5 >> 2));
    G[i] = (uint8_t)((g6 << 2) | (g6 >> 4));
    B[i] = (uint8_t)((b5 << 3) | (b5 >> 2));
  }
#else
  // Block mean, matching dct_common/rgb_features.py's extract_rgb_blocks():
  // expand each channel to 8 bits FIRST (bit replication), then average, then
  // round. Averaging the 5- and 6-bit values and expanding afterwards would
  // give different numbers, and the model was trained on the former.
  const int bw = BLOCK_W, bh = BLOCK_H;
  const uint32_t n = (uint32_t)(bw * bh);
  const uint32_t half = n / 2;                 // for round-half-up, as np.rint
  for (int oy = 0; oy < MODEL_RGB_HEIGHT; oy++) {
    for (int ox = 0; ox < MODEL_RGB_WIDTH; ox++) {
      uint32_t sr = 0, sg = 0, sb = 0;
      for (int y = 0; y < bh; y++) {
        const uint8_t *row = src + (size_t)((oy * bh + y) * CAM_WIDTH + ox * bw) * 2;
        for (int x = 0; x < bw; x++) {
          uint8_t b0 = row[2 * x], b1 = row[2 * x + 1];
          uint16_t v = swap ? (uint16_t)(((uint16_t)b1 << 8) | b0)
                            : (uint16_t)(((uint16_t)b0 << 8) | b1);
          uint32_t r5 = (v >> 11) & 0x1F, g6 = (v >> 5) & 0x3F, b5 = v & 0x1F;
          sr += (r5 << 3) | (r5 >> 2);
          sg += (g6 << 2) | (g6 >> 4);
          sb += (b5 << 3) | (b5 >> 2);
        }
      }
      const int i = oy * MODEL_RGB_WIDTH + ox;
      R[i] = (uint8_t)((sr + half) / n);
      G[i] = (uint8_t)((sg + half) / n);
      B[i] = (uint8_t)((sb + half) / n);
    }
  }
#endif
}

// Byte-order discriminator that needs no red target and no human judgement.
//
// Returns the mean absolute horizontal neighbour difference across all three
// channels. The reasoning: under the WRONG byte order the two bytes are
// misassigned, which promotes low-order bits into high-order positions --
// red picks up green's low 3 bits, green picks up blue's low 3 bits, and so
// on. Low-order bits of a natural image are close to noise, so the wrong
// reading produces an image with large pixel-to-pixel jumps, while the right
// reading produces a smooth one. Real scenes are spatially smooth; the
// difference is typically several-fold, not marginal.
//
// This is strictly better than the mean-R/G/B-under-a-red-target method it
// replaces, which needs a cooperating human and a strongly coloured object,
// and which is genuinely ambiguous on a grey scene -- as observed here, where
// a neutral room gave r=134.8 vs r=134.6 across the two orders and settled
// nothing. The only scene that defeats the smoothness test is one that is
// already pure noise.
static float rgb565_roughness(const uint8_t *src, bool swap) {
  uint32_t acc = 0;
  uint32_t n = 0;
  for (int y = 0; y < MODEL_RGB_HEIGHT; y++) {
    const uint8_t *row = src + (size_t)y * MODEL_RGB_WIDTH * 2;
    int pr = 0, pg = 0, pb = 0;
    for (int x = 0; x < MODEL_RGB_WIDTH; x++) {
      uint8_t b0 = row[2 * x], b1 = row[2 * x + 1];
      uint16_t v = swap ? (uint16_t)(((uint16_t)b1 << 8) | b0) : (uint16_t)(((uint16_t)b0 << 8) | b1);
      int r = (int)((((v >> 11) & 0x1F) << 3) | (((v >> 11) & 0x1F) >> 2));
      int g = (int)((((v >> 5) & 0x3F) << 2) | (((v >> 5) & 0x3F) >> 4));
      int b = (int)(((v & 0x1F) << 3) | ((v & 0x1F) >> 2));
      if (x > 0) {
        acc += (uint32_t)abs(r - pr) + (uint32_t)abs(g - pg) + (uint32_t)abs(b - pb);
        n += 3;
      }
      pr = r;
      pg = g;
      pb = b;
    }
  }
  return n ? (float)acc / (float)n : 0.0f;
}

// Mean R/G/B under a given byte order, straight from the RGB565 frame.
// Diagnostic only -- called once per second from the stream loop, never in
// the per-frame timed path.
static void rgb565_channel_means(const uint8_t *src, bool swap, float *mr, float *mg, float *mb) {
  uint32_t sr = 0, sg = 0, sb = 0;
  for (int i = 0; i < RGB_PIXELS; i++) {
    uint8_t b0 = src[2 * i], b1 = src[2 * i + 1];
    uint16_t v = swap ? (uint16_t)(((uint16_t)b1 << 8) | b0) : (uint16_t)(((uint16_t)b0 << 8) | b1);
    uint8_t r5 = (uint8_t)((v >> 11) & 0x1F);
    uint8_t g6 = (uint8_t)((v >> 5) & 0x3F);
    uint8_t b5 = (uint8_t)(v & 0x1F);
    sr += (uint32_t)((r5 << 3) | (r5 >> 2));
    sg += (uint32_t)((g6 << 2) | (g6 >> 4));
    sb += (uint32_t)((b5 << 3) | (b5 >> 2));
  }
  *mr = (float)sr / RGB_PIXELS;
  *mg = (float)sg / RGB_PIXELS;
  *mb = (float)sb / RGB_PIXELS;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

// Serializes access to the model's buffers, and to the g_stage_us profiling
// globals the header's MODEL_PROFILE hooks write into.
//
// This guard is required, and its absence was observed live rather than
// theorised: model_init() allocates the activations once and every forward
// pass reuses them, so what used to be per-call private stack storage is now
// process-wide shared state. The /stream worker task and the Arduino loop
// task (boot self-test and benchmark) both run inferences, and before this
// existed they were overwriting each other's conv outputs -- producing a
// self-test that could not be trusted and classifications that were quietly
// garbage.
static SemaphoreHandle_t g_model_mutex = NULL;

// The ONLY place model_forward() should be called from. Copies the per-stage
// profiling results out while still holding the lock, since g_stage_us is
// overwritten by the next caller.
static void model_forward_locked(const uint8_t *rgb_planar, int8_t logits[MODEL_NUM_CLASSES],
                                 uint32_t stage_us[MODEL_NUM_STAGES]) {
  if (g_model_mutex) xSemaphoreTake(g_model_mutex, portMAX_DELAY);
  model_forward((const uint8_t(*)[MODEL_RGB_HEIGHT][MODEL_RGB_WIDTH])rgb_planar, logits);
  if (stage_us) memcpy(stage_us, g_stage_us, sizeof(g_stage_us));
  if (g_model_mutex) xSemaphoreGive(g_model_mutex);
}

// ---------------------------------------------------------------------------
// Live measurement state (written by stream_handler once/sec, read by /status)
// ---------------------------------------------------------------------------

static volatile float g_fps = 0.0f;

static volatile int8_t g_last_logits[MODEL_NUM_CLASSES] = {0};
static volatile bool g_has_classified = false;

// Per-stage wall clock. Reported separately and never summed into a single
// aggregate fps, because the whole point of the DCT-vs-RGB comparison is
// which stage the time actually goes into.
static volatile float g_t_capture_ms = 0.0f;  // esp_camera_fb_get()
static volatile float g_t_convert_ms = 0.0f;  // RGB565 -> planar RGB888
static volatile float g_t_infer_ms = 0.0f;    // model_forward_locked()
static volatile float g_t_jpeg_ms = 0.0f;     // frame2jpg() software encode
static volatile float g_t_total_ms = 0.0f;

static volatile float g_t_stage_quantize_ms = 0.0f;
static volatile float g_t_stage_conv0_ms = 0.0f;
static volatile float g_t_stage_conv1_ms = 0.0f;
static volatile float g_t_stage_conv2_ms = 0.0f;
static volatile float g_t_stage_conv3_ms = 0.0f;
static volatile float g_t_stage_pool_ms = 0.0f;
static volatile float g_t_stage_output_ms = 0.0f;

static volatile uint32_t g_jpeg_bytes = 0;

// Byte-order diagnostic, refreshed once/sec from the live frame.
static volatile float g_mean_r_noswap = 0, g_mean_g_noswap = 0, g_mean_b_noswap = 0;
static volatile float g_mean_r_swap = 0, g_mean_g_swap = 0, g_mean_b_swap = 0;
static volatile float g_rough_noswap = 0, g_rough_swap = 0;

// Running mean of mean(R) - mean(B) over every classified frame since boot --
// the R/B plane-transposition test described above. Reference from the
// training set: median +14.5, positive in 80.2% of images.
static volatile double g_rb_delta_sum = 0.0;
static volatile uint32_t g_rb_frames = 0;

// Running mean of per-frame pixel standard deviation across all three planes
// -- a contrast measure. Training reference: median 64.2, 10th pct 45.4.
// This is the one degradation that moves the model's decision margin: a PC
// sweep found margin tracking contrast monotonically (7 -> 6 -> 4 -> 3 as
// contrast goes 1.0 -> 0.7 -> 0.5 -> 0.3), while sensor noise and RGB565 bit
// depth both leave the margin untouched. So if live std sits well below ~45,
// low-contrast capture explains the weak margins and the bias-class fallback
// in one variable, and the fix is sensor configuration rather than code.
static volatile double g_pixstd_sum = 0.0;

// Confidence calibration, so a weak-signal failure can be told apart from a
// genuine domain shift. PC reference on 200 real test images (5-class model,
// 2026-08-15): max logit median 25 (10th pct 4), top-2 margin median 24 (10th
// pct 3). Live frames sitting far below that mean the pooled features are
// collapsing -- i.e. the input pipeline is still wrong -- rather than the
// model simply disagreeing with the scene. This matters because the dense
// layer's bias makes `computer` the argmax whenever the feature vector
// carries no discriminative signal (output bias 866, against 627 for the
// next-highest, `people`), so a confident-looking `computer` is exactly what
// a broken input pipeline produces. All four LCG-noise self-test vectors land
// there for the same reason. The 17-class model this firmware previously ran
// had `plates` in that role -- same effect, different class; re-derive it
// from the output biases after any retrain rather than trusting this note.
static volatile int g_last_max_logit = 0;
static volatile int g_last_margin = 0;
static volatile double g_max_logit_sum = 0.0;
static volatile double g_margin_sum = 0.0;

// Boot-time results. Tracked per tier rather than as one aggregate, because
// the tiers carry very different evidential weight -- see rgb_test_vectors.h.
// A pass on the LCG noise vectors alone would prove almost nothing.
#define SELFTEST_TOTAL_VECTORS (PATTERN_NUM_VECTORS + SYNTH_NUM_VECTORS + REAL_NUM_VECTORS)
static volatile int g_selftest_class_ok = 0;
static volatile int g_selftest_max_diff = 0;
static volatile bool g_selftest_bit_exact_all = false;
static volatile int g_selftest_pattern_ok = 0;
static volatile int g_selftest_synth_ok = 0;
static volatile int g_selftest_real_ok = 0;
static volatile bool g_selftest_determinism_ok = false;
static volatile float g_bench_total_ms = 0.0f;
static volatile bool g_bench_contended = false;
static volatile float g_bench_quantize_ms = 0.0f;
static volatile float g_bench_conv0_ms = 0.0f;
static volatile float g_bench_conv1_ms = 0.0f;
static volatile float g_bench_conv2_ms = 0.0f;
static volatile float g_bench_conv3_ms = 0.0f;
static volatile float g_bench_pool_ms = 0.0f;
static volatile float g_bench_output_ms = 0.0f;

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-S3 RGB CNN (160x120)</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; margin:0; padding:16px; }
    canvas { border:2px solid #444; border-radius:8px; image-rendering:pixelated; }
    #fps { font-size:1.4em; margin:12px; }
    #timing { font-size:0.9em; color:#aaa; margin-bottom:12px; }
    #scores { max-width:320px; margin:0 auto 16px; text-align:left; }
    .score-row { display:flex; align-items:center; gap:8px; margin:4px 0; font-size:1em; }
    .score-name { width:80px; flex-shrink:0; }
    .score-bar-track { flex:1; background:#333; border-radius:4px; height:16px; overflow:hidden; }
    .score-bar-fill { height:100%; background:#7fd; }
    .score-pct { width:48px; flex-shrink:0; text-align:right; color:#aaa; font-size:0.9em; }
    .score-row:first-child .score-name,
    .score-row:first-child .score-pct { color:#7fd; font-weight:bold; }
    .stream-row { display:flex; gap:16px; justify-content:center; align-items:flex-start; flex-wrap:wrap; }
    .panel { text-align:left; background:#1a1a1a; border-radius:8px; padding:12px 16px; min-width:240px; }
    .panel h3 { margin:0 0 8px; font-size:0.95em; color:#f97; font-weight:normal; }
    .stage-row { display:flex; align-items:center; gap:8px; margin:4px 0; font-size:0.85em; }
    .stage-name { width:110px; flex-shrink:0; color:#ccc; }
    .stage-bar-track { flex:1; background:#333; border-radius:3px; height:10px; overflow:hidden; min-width:40px; }
    .stage-bar-fill { height:100%; background:#f97; }
    .stage-ms { width:62px; flex-shrink:0; text-align:right; color:#aaa; }
    table.diag { font-size:0.82em; border-collapse:collapse; }
    table.diag td { padding:2px 8px 2px 0; color:#bbb; }
    table.diag td.hd { color:#888; }
    .psram { color:#f97; }
    .sram { color:#7fd; }
    .warn { color:#fc6; font-size:0.85em; margin:8px auto; max-width:640px; text-align:left; line-height:1.4; }
    button { background:#333; color:#eee; border:1px solid #555; border-radius:4px; padding:4px 10px; margin:2px; cursor:pointer; font-size:0.85em; }
    button.on { background:#7fd; color:#111; border-color:#7fd; }
  </style>
</head>
<body>
  <h2>ESP32-S3 RGB pixel CNN (160x120, portable int8 kernels)</h2>
  <div class="warn" id="caveat"></div>
  <div id="fps">FPS: --</div>
  <div id="scores"></div>
  <div id="timing"></div>
  <div class="stream-row">
    <canvas id="stream"></canvas>
    <div class="panel">
      <h3>inference stages (live)</h3>
      <div id="stageRows"></div>
      <h3 style="margin-top:12px">pipeline</h3>
      <div id="pipeRows"></div>
    </div>
    <div class="panel">
      <h3>RGB565 byte order</h3>
      <div id="byteOrder"></div>
      <h3 style="margin-top:12px">buffer placement</h3>
      <table class="diag" id="bufTable"></table>
      <h3 style="margin-top:12px">boot benchmark</h3>
      <table class="diag" id="benchTable"></table>
    </div>
  </div>
  <script>
    const canvas = document.getElementById('stream');
    const ctx = canvas.getContext('2d');

    const stageLabels = {
      quantize: 'quantize',
      conv0: 'conv0 3→16',
      conv1: 'conv1 16→32',
      conv2: 'conv2 32→64',
      conv3: 'conv3 64→32',
      pool: 'global pool',
      output: 'dense'
    };

    function bars(container, rows, labelMap) {
      const maxMs = Math.max(0.01, ...rows.map(s => s.ms));
      container.innerHTML = '';
      for (const s of rows) {
        const row = document.createElement('div');
        row.className = 'stage-row';
        const pct = (s.ms / maxMs * 100).toFixed(0);
        row.innerHTML =
          '<span class="stage-name">' + ((labelMap && labelMap[s.name]) || s.name) + '</span>' +
          '<span class="stage-bar-track"><span class="stage-bar-fill" style="width:' + pct + '%"></span></span>' +
          '<span class="stage-ms">' + s.ms.toFixed(2) + 'ms</span>';
        container.appendChild(row);
      }
    }

    function updateStatus(j) {
      document.getElementById('fps').textContent =
        'FPS: ' + j.fps.toFixed(1) + (j.stream_jpeg ? '' : '  (JPEG encode disabled)');

      const scoresDiv = document.getElementById('scores');
      scoresDiv.innerHTML = '';
      for (const s of (j.scores || [])) {
        const row = document.createElement('div');
        row.className = 'score-row';
        row.innerHTML =
          '<span class="score-name">' + s.class + '</span>' +
          '<span class="score-bar-track"><span class="score-bar-fill" style="width:' + s.pct + '%"></span></span>' +
          '<span class="score-pct">' + s.pct.toFixed(1) + '%</span>';
        scoresDiv.appendChild(row);
      }

      document.getElementById('timing').textContent =
        'capture ' + j.t_capture_ms.toFixed(1) + 'ms | convert ' + j.t_convert_ms.toFixed(1) +
        'ms | inference ' + j.t_infer_ms.toFixed(1) + 'ms | jpeg ' + j.t_jpeg_ms.toFixed(1) +
        'ms | total ' + j.t_total_ms.toFixed(1) + 'ms';

      bars(document.getElementById('stageRows'), j.stages || [], stageLabels);
      bars(document.getElementById('pipeRows'), [
        { name: 'capture', ms: j.t_capture_ms },
        { name: 'convert', ms: j.t_convert_ms },
        { name: 'inference', ms: j.t_infer_ms },
        { name: 'jpeg encode', ms: j.t_jpeg_ms }
      ], null);

      const bo = j.byte_order || {};
      const a = bo.noswap || {}, b = bo.swap || {};
      document.getElementById('byteOrder').innerHTML =
        '<table class="diag">' +
        '<tr><td class="hd"></td><td class="hd">R</td><td class="hd">G</td><td class="hd">B</td><td class="hd">rough</td></tr>' +
        '<tr><td>swap=0</td><td>' + (a.r||0).toFixed(0) + '</td><td>' + (a.g||0).toFixed(0) + '</td><td>' + (a.b||0).toFixed(0) + '</td><td>' + (a.rough||0).toFixed(1) + '</td></tr>' +
        '<tr><td>swap=1</td><td>' + (b.r||0).toFixed(0) + '</td><td>' + (b.g||0).toFixed(0) + '</td><td>' + (b.b||0).toFixed(0) + '</td><td>' + (b.rough||0).toFixed(1) + '</td></tr>' +
        '</table>' +
        '<div style="margin-top:6px">active: <b>swap=' + (bo.active_swap ? 1 : 0) + '</b></div>' +
        '<div style="margin-top:4px">' +
        '<button id="sw0" onclick="setSwap(0)">swap=0</button>' +
        '<button id="sw1" onclick="setSwap(1)">swap=1</button>' +
        '<button id="jp" onclick="setJpeg(' + (j.stream_jpeg ? 0 : 1) + ')">jpeg ' + (j.stream_jpeg ? 'off' : 'on') + '</button>' +
        '</div>' +
        (function () {
          const ra = a.rough || 0, rb = b.rough || 0;
          if (!ra && !rb) return '';
          const rec = ra < rb ? 0 : 1;
          const ratio = (Math.max(ra, rb) / Math.max(0.01, Math.min(ra, rb))).toFixed(1);
          const agree = (rec === (bo.active_swap ? 1 : 0));
          return '<div style="margin-top:6px;color:' + (agree ? '#7fd' : '#fc6') +
                 ';font-size:0.8em;line-height:1.35"><b>Recommended: swap=' + rec + '</b> (' +
                 ratio + '× smoother)' + (agree ? ' — matches active.' : ' — DIFFERS from active.') +
                 '</div>';
        })() +
        '<div style="margin-top:6px;color:#888;font-size:0.78em;line-height:1.35">' +
        '“rough” is mean absolute horizontal neighbour difference. The wrong byte order ' +
        'promotes low-order bits into high-order positions, so it yields a visibly noisier ' +
        'image; the correct one is smooth. Lower wins. This works on any non-noise scene, ' +
        'unlike comparing R means, which a grey scene leaves ambiguous.</div>';
      document.getElementById('sw' + (bo.active_swap ? 1 : 0)).className = 'on';

      const bufT = document.getElementById('bufTable');
      bufT.innerHTML = '<tr><td class="hd">buffer</td><td class="hd">bytes</td><td class="hd">where</td></tr>';
      for (const b of (j.buffers || [])) {
        const cls = b.mem === 'psram' ? 'psram' : 'sram';
        bufT.innerHTML += '<tr><td>' + b.name + '</td><td>' + b.bytes.toLocaleString() +
                          '</td><td class="' + cls + '">' + b.mem.toUpperCase() + '</td></tr>';
      }

      const bm = j.bench || {};
      const bt = document.getElementById('benchTable');
      bt.innerHTML = '<tr><td class="hd">stage</td><td class="hd">ms</td></tr>';
      for (const k of ['quantize','conv0','conv1','conv2','conv3','pool','output','total']) {
        if (bm[k] === undefined) continue;
        bt.innerHTML += '<tr><td>' + (stageLabels[k] || k) + '</td><td>' + bm[k].toFixed(2) + '</td></tr>';
      }

      const st = j.self_test || {};
      const t = (g) => g ? g.ok + '/' + g.total : '?';
      let caveat = j.esp_nn
        ? '<b>ESP-NN enabled</b> (esp32s3 int8 kernels). Compare against the DCT ' +
          'firmware&rsquo;s ESP-NN figures — 21.9 ms inference / 23.7 fps — not its portable ones.'
        : '<b>No ESP-NN in this build.</b> These kernels are portable C. Compare against the ' +
          'DCT firmware&rsquo;s portable figures — 306 ms inference / 2.3 fps — not its ' +
          'headline 23.7 fps.';
      if (st.status === 'pass') {
        // On an ESP-NN build, class agreement is the bar and small logit drift
        // is expected; claiming "bit-exact" there would be wrong.
        caveat += st.bit_exact_all
          ? '<br>Numerics self-test <b>PASS</b> — bit-exact vs the PC int8 reference on all ' +
            (st.class_total || 0) + ' vectors.'
          : '<br>Numerics self-test <b>PASS</b> — predicted class matches the PC int8 reference ' +
            'on ' + (st.class_ok || 0) + '/' + (st.class_total || 0) + ' vectors (max logit drift ' +
            (st.max_logit_diff || 0) + '), bit-exact on pattern ' + t(st.pattern) + ', synth ' +
            t(st.synth) + ', real ' + t(st.real) + '. Small drift is expected from ESP-NN&rsquo;s ' +
            'two-step requantize.';
      } else {
        caveat += '<br><b>Numerics self-test FAIL</b> — class match ' + (st.class_ok || 0) + '/' +
                  (st.class_total || 0) + ', max logit drift ' + (st.max_logit_diff || 0) +
                  ', determinism ' + (st.determinism_ok ? 'ok' : 'FAILED') +
                  '. Timings below are measuring the wrong computation. If a browser was ' +
                  'connected while the board booted, reload after the self-test completes.';
      }
      document.getElementById('caveat').innerHTML = caveat;
    }

    async function setSwap(v) { try { await fetch('/config?swap=' + v); } catch (e) {} }
    async function setJpeg(v) { try { await fetch('/config?jpeg=' + v); } catch (e) {} }

    async function renderJpegPart(bytes) {
      try {
        const blob = new Blob([bytes], { type: 'image/jpeg' });
        const bitmap = await createImageBitmap(blob);
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          canvas.style.width = (bitmap.width * 2) + 'px';
          canvas.style.height = (bitmap.height * 2) + 'px';
        }
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
      } catch (e) {
        // A single malformed/partial frame isn't worth tearing down the
        // connection over -- just skip it.
      }
    }

    function indexOfBytes(hay, needle, from) {
      outer: for (let i = from; i <= hay.length - needle.length; i++) {
        for (let j = 0; j < needle.length; j++) {
          if (hay[i + j] !== needle[j]) continue outer;
        }
        return i;
      }
      return -1;
    }

    // Reads a multipart/x-mixed-replace stream from url, calling
    // onPart(contentType, bodyBytes) for each complete part as it arrives.
    // The firmware interleaves a JSON status part into the same connection
    // the JPEG frames go out on (once/sec), so this page reads ONE connection
    // instead of polling /status separately -- a separate poll was measured
    // to contend with /stream for the ESP32-S3's single WiFi radio.
    async function streamMultipart(url, onPart) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error('stream fetch failed: ' + resp.status);
      const contentType = resp.headers.get('Content-Type') || '';
      const boundaryMatch = /boundary=(.+)$/i.exec(contentType);
      if (!boundaryMatch) throw new Error('no boundary in stream Content-Type');
      const boundaryBytes = new TextEncoder().encode('--' + boundaryMatch[1].trim());
      const sepBytes = new TextEncoder().encode('\r\n\r\n');
      const decoder = new TextDecoder();

      const reader = resp.body.getReader();
      let buf = new Uint8Array(0);

      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        if (value && value.length) {
          const merged = new Uint8Array(buf.length + value.length);
          merged.set(buf, 0);
          merged.set(value, buf.length);
          buf = merged;
        }

        while (true) {
          const boundaryIdx = indexOfBytes(buf, boundaryBytes, 0);
          if (boundaryIdx === -1) break;
          const headerStart = boundaryIdx + boundaryBytes.length;
          const sepIdx = indexOfBytes(buf, sepBytes, headerStart);
          if (sepIdx === -1) break; // headers not fully arrived yet

          const headerText = decoder.decode(buf.slice(headerStart, sepIdx));
          const lengthMatch = /Content-Length:\s*(\d+)/i.exec(headerText);
          const typeMatch = /Content-Type:\s*([^\r\n]+)/i.exec(headerText);
          if (!lengthMatch) {
            buf = buf.slice(sepIdx + sepBytes.length);
            continue;
          }

          const bodyLen = parseInt(lengthMatch[1], 10);
          const bodyStart = sepIdx + sepBytes.length;
          const bodyEnd = bodyStart + bodyLen;
          if (buf.length < bodyEnd) break; // body not fully arrived yet

          onPart(typeMatch ? typeMatch[1].trim() : '', buf.slice(bodyStart, bodyEnd));
          buf = buf.slice(bodyEnd);
        }
      }
    }

    async function runStream() {
      const url = 'http://' + window.location.hostname + ':81/stream';
      while (true) {
        try {
          await streamMultipart(url, (type, body) => {
            if (type === 'image/jpeg') {
              renderJpegPart(body);
            } else if (type === 'application/json') {
              try {
                updateStatus(JSON.parse(new TextDecoder().decode(body)));
              } catch (e) { /* one bad status part isn't worth resetting the connection over */ }
            }
          });
        } catch (e) {
          document.getElementById('fps').textContent = 'FPS: (stream disconnected, retrying...)';
        }
        await new Promise(r => setTimeout(r, 1000));
      }
    }
    runStream();
  </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  // No caching: this page is reflashed constantly during measurement, and a
  // browser holding stale JS renders current data with an old layout, which
  // reads as a firmware bug rather than a cache artifact.
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// Current WiFi power-save mode as a short string for /status. Read live
// (not cached from boot) so a mode change after association is visible.
static const char *wifi_ps_name() {
  wifi_ps_type_t ps;
  if (esp_wifi_get_ps(&ps) != ESP_OK) return "unknown";
  return ps == WIFI_PS_NONE ? "none" : (ps == WIFI_PS_MIN_MODEM ? "min_modem" : "max_modem");
}

// The bar differs by build, deliberately.
//
// PORTABLE build: bit-exact on every vector. The device runs the same integer
// arithmetic as the PC reference, so any difference at all is a bug.
//
// ESP-NN build: bit-exactness is NOT required and is reported for information
// only. esp_nn_conv_s8 uses a two-step requantization that rounds differently
// from the portable single combined round-shift -- a real, understood,
// small divergence that the DCT project hit and documented before this one.
// The correctness bar is that the predicted CLASS still matches on every
// vector, plus determinism. A class mismatch, or drift beyond a couple of
// LSB, is what would actually indicate a problem.
//
// Determinism is required in both builds and is not a formality: model_init()
// hands out heap buffers that are never zeroed and are reused for every
// frame, so a read-before-write would produce stable garbage that could match
// expectations within a boot.
#define SELFTEST_MAX_ACCEPTABLE_DRIFT 2

static const char *self_test_status() {
  bool bit_exact_all = g_selftest_pattern_ok == PATTERN_NUM_VECTORS &&
                       g_selftest_synth_ok == SYNTH_NUM_VECTORS &&
                       g_selftest_real_ok == REAL_NUM_VECTORS;
  g_selftest_bit_exact_all = bit_exact_all;
  if (!g_selftest_determinism_ok) return "fail";
#if BUILD_HAS_ESP_NN
  bool ok = g_selftest_class_ok == SELFTEST_TOTAL_VECTORS &&
            g_selftest_max_diff <= SELFTEST_MAX_ACCEPTABLE_DRIFT;
  return ok ? "pass" : "fail";
#else
  return bit_exact_all ? "pass" : "fail";
#endif
}

// Builds the status JSON into caller-provided storage, returning its length.
// Split out so the /stream handler can push the exact same payload as an
// interleaved multipart part -- one builder keeps the two paths from
// drifting. Note: sizeof(buf) would be a pointer size in here -- use bufsize.
static int build_status_json(char *buf, size_t bufsize) {
  int off = snprintf(
      buf, bufsize,
      "{\"fps\":%.1f,\"t_capture_ms\":%.1f,\"t_convert_ms\":%.1f,\"t_infer_ms\":%.1f,"
      "\"t_jpeg_ms\":%.1f,\"t_total_ms\":%.1f,\"jpeg_bytes\":%u,\"stream_jpeg\":%s,\"jpeg_quality\":%d,"
      "\"esp_nn\":%s,\"bench_contended\":%s,"
      "\"rssi\":%d,\"free_heap\":%u,\"free_psram\":%u,\"wifi_ps\":\"%s\","
      "\"stages\":["
      "{\"name\":\"quantize\",\"ms\":%.2f},"
      "{\"name\":\"conv0\",\"ms\":%.2f},"
      "{\"name\":\"conv1\",\"ms\":%.2f},"
      "{\"name\":\"conv2\",\"ms\":%.2f},"
      "{\"name\":\"conv3\",\"ms\":%.2f},"
      "{\"name\":\"pool\",\"ms\":%.2f},"
      "{\"name\":\"output\",\"ms\":%.2f}"
      "],"
      "\"bench\":{\"quantize\":%.2f,\"conv0\":%.2f,\"conv1\":%.2f,\"conv2\":%.2f,"
      "\"conv3\":%.2f,\"pool\":%.2f,\"output\":%.2f,\"total\":%.2f},"
      "\"self_test\":{\"status\":\"%s\",\"determinism_ok\":%s,"
      "\"pattern\":{\"ok\":%d,\"total\":%d},\"synth\":{\"ok\":%d,\"total\":%d},"
      "\"real\":{\"ok\":%d,\"total\":%d},"
      "\"class_ok\":%d,\"class_total\":%d,\"max_logit_diff\":%d,\"bit_exact_all\":%s},"
      "\"signal\":{\"frames\":%u,\"rb_delta_mean\":%.2f,\"pixel_std_mean\":%.1f,\"max_logit\":%d,\"margin\":%d,"
      "\"max_logit_mean\":%.1f,\"margin_mean\":%.1f,\"rb_swap\":%s},"
      "\"byte_order\":{\"active_swap\":%s,"
      "\"noswap\":{\"r\":%.1f,\"g\":%.1f,\"b\":%.1f,\"rough\":%.2f},"
      "\"swap\":{\"r\":%.1f,\"g\":%.1f,\"b\":%.1f,\"rough\":%.2f}},"
      "\"buffers\":[",
      g_fps, g_t_capture_ms, g_t_convert_ms, g_t_infer_ms, g_t_jpeg_ms, g_t_total_ms,
      (unsigned)g_jpeg_bytes, g_stream_jpeg ? "true" : "false", g_jpeg_quality,
      BUILD_HAS_ESP_NN ? "true" : "false", g_bench_contended ? "true" : "false",
      // RSSI/free-heap as standing diagnostics: a weak link (worse than about
      // -75 dBm) or a shrinking heap would each produce stall-like streaming
      // symptoms that look identical from the browser.
      (int)WiFi.RSSI(), (unsigned)ESP.getFreeHeap(),
      (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), wifi_ps_name(),
      g_t_stage_quantize_ms, g_t_stage_conv0_ms, g_t_stage_conv1_ms, g_t_stage_conv2_ms,
      g_t_stage_conv3_ms, g_t_stage_pool_ms, g_t_stage_output_ms, g_bench_quantize_ms,
      g_bench_conv0_ms, g_bench_conv1_ms, g_bench_conv2_ms, g_bench_conv3_ms, g_bench_pool_ms,
      g_bench_output_ms, g_bench_total_ms, self_test_status(),
      g_selftest_determinism_ok ? "true" : "false", g_selftest_pattern_ok, PATTERN_NUM_VECTORS,
      g_selftest_synth_ok, SYNTH_NUM_VECTORS, g_selftest_real_ok, REAL_NUM_VECTORS,
      g_selftest_class_ok, SELFTEST_TOTAL_VECTORS, g_selftest_max_diff,
      g_selftest_bit_exact_all ? "true" : "false", (unsigned)g_rb_frames,
      g_rb_frames ? (float)(g_rb_delta_sum / g_rb_frames) : 0.0f,
      g_rb_frames ? (float)(g_pixstd_sum / g_rb_frames) : 0.0f, g_last_max_logit, g_last_margin,
      g_rb_frames ? (float)(g_max_logit_sum / g_rb_frames) : 0.0f,
      g_rb_frames ? (float)(g_margin_sum / g_rb_frames) : 0.0f, g_rb_swap ? "true" : "false",
      g_rgb565_swap ? "true" : "false", g_mean_r_noswap, g_mean_g_noswap, g_mean_b_noswap,
      g_rough_noswap, g_mean_r_swap, g_mean_g_swap, g_mean_b_swap, g_rough_swap);

  int nbuf = 0;
  const model_buffer_info_t *binfo = model_buffer_report(&nbuf);
  for (int i = 0; i < nbuf; i++) {
    off += snprintf(buf + off, bufsize - off, "%s{\"name\":\"%s\",\"bytes\":%d,\"mem\":\"%s\"}",
                    i == 0 ? "" : ",", binfo[i].name, binfo[i].bytes,
                    binfo[i].internal ? "sram" : "psram");
  }
  off += snprintf(buf + off, bufsize - off, "%s{\"name\":\"rgb888\",\"bytes\":%d,\"mem\":\"%s\"}",
                  nbuf == 0 ? "" : ",", SZ_RGB888, g_rgb888_internal ? "sram" : "psram");
  off += snprintf(buf + off, bufsize - off, "],\"scores\":[");

  if (g_has_classified) {
    int8_t logits[MODEL_NUM_CLASSES];
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) logits[i] = g_last_logits[i];

    // Softmax over the raw int8 logits -> a real probability distribution,
    // computed here (once per status push) rather than in the per-frame hot
    // path, since it's display formatting and not part of classification.
    int8_t max_logit = logits[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++)
      if (logits[i] > max_logit) max_logit = logits[i];

    float exps[MODEL_NUM_CLASSES];
    float sum = 0.0f;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
      exps[i] = expf((float)(logits[i] - max_logit));
      sum += exps[i];
    }

    int order[MODEL_NUM_CLASSES];
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) order[i] = i;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++)
      for (int j = i + 1; j < MODEL_NUM_CLASSES; j++)
        if (exps[order[j]] > exps[order[i]]) {
          int t = order[i];
          order[i] = order[j];
          order[j] = t;
        }

    // Top 6 only: the full 17-class distribution would push this JSON past
    // the buffer, and the tail is all sub-1% noise.
    int n_report = MODEL_NUM_CLASSES < 6 ? MODEL_NUM_CLASSES : 6;
    for (int i = 0; i < n_report; i++) {
      float pct = 100.0f * exps[order[i]] / sum;
      off += snprintf(buf + off, bufsize - off, "%s{\"class\":\"%s\",\"pct\":%.1f}",
                      i == 0 ? "" : ",", MODEL_CLASS_NAMES[order[i]], pct);
    }
  }
  off += snprintf(buf + off, bufsize - off, "]}");
  return off;
}

static esp_err_t status_handler(httpd_req_t *req) {
  char buf[3072];
  int off = build_status_json(buf, sizeof(buf));
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, buf, off);
}

// Runtime knobs so the two experiments that would otherwise each need a
// reflash can be run from the page: the RGB565 byte-order determination, and
// pipeline timing with the software JPEG encode removed.
static esp_err_t config_handler(httpd_req_t *req) {
  char query[64];
  char val[8];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "swap", val, sizeof(val)) == ESP_OK) {
      g_rgb565_swap = (val[0] == '1');
      Serial.printf("config: rgb565 byte swap = %d\n", g_rgb565_swap ? 1 : 0);
    }
    if (httpd_query_key_value(query, "jpeg", val, sizeof(val)) == ESP_OK) {
      g_stream_jpeg = (val[0] == '1');
      Serial.printf("config: stream jpeg encode = %d\n", g_stream_jpeg ? 1 : 0);
    }
    if (httpd_query_key_value(query, "rbswap", val, sizeof(val)) == ESP_OK) {
      g_rb_swap = (val[0] == '1');
      g_rb_delta_sum = 0.0;  // statistic is only meaningful within one setting
      g_pixstd_sum = 0.0;
      g_max_logit_sum = 0.0;
      g_margin_sum = 0.0;
      g_rb_frames = 0;
      Serial.printf("config: R/B plane swap = %d (stats reset)\n", g_rb_swap ? 1 : 0);
    }
    if (httpd_query_key_value(query, "jq", val, sizeof(val)) == ESP_OK) {
      int q = atoi(val);
      if (q >= 1 && q <= 100) {
        g_jpeg_quality = q;
        Serial.printf("config: jpeg encode quality = %d (1-100, higher better)\n", q);
      }
    }
  }
  httpd_resp_set_type(req, "application/json");
  char out[96];
  int n = snprintf(out, sizeof(out), "{\"swap\":%s,\"jpeg\":%s,\"jq\":%d}",
                   g_rgb565_swap ? "true" : "false", g_stream_jpeg ? "true" : "false",
                   g_jpeg_quality);
  return httpd_resp_send(req, out, n);
}

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
// Format strings (not const char* variables) so they can be concatenated with
// STREAM_BOUNDARY at compile time into a single snprintf.
#define STREAM_PART_FMT "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"
#define STREAM_JSON_PART_FMT "Content-Type: application/json\r\nContent-Length: %u\r\n\r\n"

// STREAMING COSTS EXTRA IN THIS FIRMWARE -- and that is a finding, not an
// implementation detail.
//
// The DCT pipeline got streaming for free: the camera produced one JPEG, the
// classifier parsed its coefficients, and the same bytes went to /stream. One
// capture served both consumers. Capturing RGB565 breaks that, because the
// OV2640 driver delivers exactly one pixel format at a time. Keeping /stream
// working therefore means software-JPEG-encoding every frame with frame2jpg()
// -- a whole pipeline stage that simply does not exist in the DCT firmware
// and that no pure inference benchmark would ever capture.
//
// That stage is kept and measured (t_jpeg_ms) rather than dropped, because
// dropping it would quietly delete the strongest part of the DCT
// representation's case. /config?jpeg=0 disables just the encode so the same
// board can report both the with-encode and without-encode pipeline, which is
// the honest way to present it: the RGB pipeline's cost depends on whether
// you need the image, and the DCT pipeline's does not.
// TODO(design): this handler does capture, conversion, inference AND JPEG
// encoding inline, and never returns while a client is connected. Since
// esp_http_server serves requests from a single task, that means: nothing runs
// with no viewer, only one viewer is ever possible, a reload wedges the stream
// until send_wait_timeout fires (~5s), and /status on port 80 contends for the
// camera and model mutex. Four symptoms, one cause.
//
// The fix is to move the pipeline into its own task and leave this handler
// shipping the latest published JPEG. Written up, with the measured evidence
// and the buffer-ownership traps, in markdown/stream_handler_refactor.md.
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[192];
  char status_buf[3072];
  char status_hdr[192];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  // Disable Nagle's algorithm on this socket. Symptom without it (measured on
  // the sibling firmware, same streaming code): frames delivered in fast
  // bursts (median 4ms apart) separated by long stalls (p99 2.3s), while the
  // device's own per-frame timing stayed healthy throughout. That
  // burst-then-stall shape with a tiny median is the classic Nagle +
  // delayed-ACK interaction.
  int stream_fd = httpd_req_to_sockfd(req);
  if (stream_fd >= 0) {
    int nodelay = 1;
    if (setsockopt(stream_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) != 0) {
      Serial.println("warning: failed to set TCP_NODELAY on stream socket");
    }
  }

  int64_t last_report = esp_timer_get_time();
  uint32_t frames_since_report = 0;
  int64_t last_classify_log = 0;
  int64_t last_status_push = 0;
  int64_t last_diag = 0;
  bool warned_format = false;
  g_stream_clients++;

  uint8_t *rgb888 = g_rgb888;

  while (true) {
    int64_t t_capture_start = esp_timer_get_time();
    fb = esp_camera_fb_get();
    int64_t t_capture_end = esp_timer_get_time();
    if (!fb) {
      res = ESP_FAIL;
      break;
    }

    // The frame must be exactly what the model expects. A silent mismatch
    // here (wrong format after a sensor reconfigure, short frame) would feed
    // garbage to the classifier and still produce confident-looking output.
    bool frame_ok = (fb->format == PIXFORMAT_RGB565) && (fb->len >= (size_t)(RGB_PIXELS * 2));
    if (!frame_ok && !warned_format) {
      Serial.printf("ERROR: unexpected frame: format=%d len=%u (want RGB565, >=%u bytes)\n",
                    (int)fb->format, (unsigned)fb->len, (unsigned)(RGB_PIXELS * 2));
      warned_format = true;
    }

    // ONE guard for conversion and inference, deliberately computed once.
    // These were previously two separate conditions and the conversion was
    // missing the !g_selftest_running term -- so a browser connected during
    // the boot self-test kept overwriting g_rgb888 with camera frames while
    // run_model_self_test() was filling that same buffer with its patterns
    // and classifying them. Observed live: the self-test reported class 8/9
    // with a max logit diff of 39 instead of 9/9 at diff 2, but only when a
    // client happened to be connected at boot, which made it look
    // intermittent rather than like a race.
    //
    // The model mutex does not cover this. It serializes the model's own
    // internal buffers; g_rgb888 is the caller-owned INPUT buffer, passed in
    // by pointer, and was never protected by anything.
    bool do_classify = frame_ok && g_bufs_ready && !g_selftest_running;

    int64_t t_convert_end = t_capture_end;
    if (do_classify) {
      rgb565_to_planar_rgb888(fb->buf, rgb888, g_rgb565_swap);
      t_convert_end = esp_timer_get_time();
    }

    int64_t now = esp_timer_get_time();

    // Byte-order diagnostic, once/sec and outside every timed stage.
    if (frame_ok && now - last_diag >= 1000000) {
      float r, g, b;
      rgb565_channel_means(fb->buf, false, &r, &g, &b);
      g_mean_r_noswap = r;
      g_mean_g_noswap = g;
      g_mean_b_noswap = b;
      rgb565_channel_means(fb->buf, true, &r, &g, &b);
      g_mean_r_swap = r;
      g_mean_g_swap = g;
      g_mean_b_swap = b;
      g_rough_noswap = rgb565_roughness(fb->buf, false);
      g_rough_swap = rgb565_roughness(fb->buf, true);
      last_diag = now;
    }

    // Encode before releasing the frame buffer, then release it, then run
    // inference off the already-converted planar copy. Holding fb across a
    // multi-hundred-millisecond inference would stall the camera driver's
    // DMA rotation for no reason -- nothing after this point reads fb.
    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    int64_t t_jpeg_start = esp_timer_get_time();
    if (g_stream_jpeg && frame_ok) {
      if (!frame2jpg(fb, (uint8_t)g_jpeg_quality, &jpg_buf, &jpg_len)) {
        jpg_buf = NULL;
        jpg_len = 0;
      }
    }
    int64_t t_jpeg_end = esp_timer_get_time();

    esp_camera_fb_return(fb);
    fb = NULL;

    if (do_classify) {
      int8_t logits[MODEL_NUM_CLASSES];
      uint32_t stage_us[MODEL_NUM_STAGES];
      int64_t t_infer_start = esp_timer_get_time();
      model_forward_locked(rgb888, logits, stage_us);
      int64_t t_infer_end = esp_timer_get_time();

      for (int i = 0; i < MODEL_NUM_CLASSES; i++) g_last_logits[i] = logits[i];
      g_has_classified = true;
      g_t_capture_ms = (t_capture_end - t_capture_start) / 1000.0f;
      g_t_convert_ms = (t_convert_end - t_capture_end) / 1000.0f;
      g_t_jpeg_ms = (t_jpeg_end - t_jpeg_start) / 1000.0f;
      g_t_infer_ms = (t_infer_end - t_infer_start) / 1000.0f;
      g_t_total_ms = (t_infer_end - t_capture_start) / 1000.0f;
      g_jpeg_bytes = (uint32_t)jpg_len;
      g_t_stage_quantize_ms = stage_us[MODEL_STAGE_QUANTIZE] / 1000.0f;
      g_t_stage_conv0_ms = stage_us[MODEL_STAGE_CONV0] / 1000.0f;
      g_t_stage_conv1_ms = stage_us[MODEL_STAGE_CONV1] / 1000.0f;
      g_t_stage_conv2_ms = stage_us[MODEL_STAGE_CONV2] / 1000.0f;
      g_t_stage_conv3_ms = stage_us[MODEL_STAGE_CONV3] / 1000.0f;
      g_t_stage_pool_ms = stage_us[MODEL_STAGE_POOL] / 1000.0f;
      g_t_stage_output_ms = stage_us[MODEL_STAGE_DENSE] / 1000.0f;

      // Diagnostics, deliberately after every timed stage so they cannot
      // contaminate a reported latency.
      {
        uint32_t sr = 0, sb = 0;
        for (int i = 0; i < RGB_PIXELS; i++) {
          sr += rgb888[i];
          sb += rgb888[2 * RGB_PIXELS + i];
        }
        g_rb_delta_sum += ((double)sr - (double)sb) / RGB_PIXELS;

        // Pixel std over all three planes. uint64 accumulator: 57,600 samples
        // of v^2 <= 65,025 peaks near 3.7e9, which would overflow uint32.
        uint64_t sum = 0, sumsq = 0;
        for (int i = 0; i < SZ_RGB888; i++) {
          uint32_t v = rgb888[i];
          sum += v;
          sumsq += (uint64_t)v * v;
        }
        double mean = (double)sum / SZ_RGB888;
        double var = (double)sumsq / SZ_RGB888 - mean * mean;
        g_pixstd_sum += var > 0.0 ? sqrt(var) : 0.0;

        int best = 0, second = -128;
        for (int i = 1; i < MODEL_NUM_CLASSES; i++)
          if (logits[i] > logits[best]) best = i;
        for (int i = 0; i < MODEL_NUM_CLASSES; i++)
          if (i != best && logits[i] > second) second = logits[i];
        g_last_max_logit = logits[best];
        g_last_margin = logits[best] - second;
        g_max_logit_sum += g_last_max_logit;
        g_margin_sum += g_last_margin;
        g_rb_frames++;
      }

      now = esp_timer_get_time();
      if (now - last_classify_log >= 1000000) {  // throttle to 1/sec
        int pred = model_argmax(logits, MODEL_NUM_CLASSES);
        Serial.printf(
            "class=%s maxlogit=%d margin=%d | R-B mean %+.1f over %u frames | "
            "capture=%.1f convert=%.1f infer=%.1f jpeg=%.1f total=%.1f ms\n",
            MODEL_CLASS_NAMES[pred], g_last_max_logit, g_last_margin,
            g_rb_frames ? (float)(g_rb_delta_sum / g_rb_frames) : 0.0f, (unsigned)g_rb_frames,
            g_t_capture_ms, g_t_convert_ms, g_t_infer_ms, g_t_jpeg_ms, g_t_total_ms);
        last_classify_log = now;
      }
    }

    // Boundary and part header go out as ONE write rather than two: fewer,
    // larger TCP segments per frame, which reduces the small-packet churn
    // that made the Nagle stall above so easy to hit.
    if (jpg_buf) {
      size_t hlen =
          snprintf(part_buf, sizeof(part_buf), "%s" STREAM_PART_FMT, STREAM_BOUNDARY, (unsigned)jpg_len);
      // snprintf returns the length it *would* have written even when
      // truncated -- clamp so a future header format change can't silently
      // read past part_buf.
      if (hlen >= sizeof(part_buf)) hlen = sizeof(part_buf) - 1;

      if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
      free(jpg_buf);
      jpg_buf = NULL;
    }

    if (res != ESP_OK) break;

    frames_since_report++;
    now = esp_timer_get_time();
    if (now - last_report >= 1000000) {
      g_fps = frames_since_report * 1000000.0f / (float)(now - last_report);
      frames_since_report = 0;
      last_report = now;
    }

    // Push status as an interleaved multipart part on this SAME connection
    // rather than having the page poll /status separately -- the separate
    // poll was measured to contend with /stream for the ESP32-S3's single
    // WiFi radio (round-trip 300ms isolated vs 700-1300ms during streaming,
    // with outright stream stalls).
    if (res == ESP_OK && now - last_status_push >= 1000000) {
      int status_len = build_status_json(status_buf, sizeof(status_buf));
      size_t status_hlen = snprintf(status_hdr, sizeof(status_hdr), "%s" STREAM_JSON_PART_FMT,
                                    STREAM_BOUNDARY, (unsigned)status_len);
      if (status_hlen >= sizeof(status_hdr)) status_hlen = sizeof(status_hdr) - 1;

      res = httpd_resp_send_chunk(req, status_hdr, status_hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, status_buf, status_len);
      last_status_push = now;
      if (res != ESP_OK) break;
    }

    // With the JPEG encode disabled there is no image write in this loop, so
    // nothing yields to lower-priority tasks except the once-a-second status
    // push. Yield explicitly so the idle task still runs and the watchdog
    // stays fed.
    if (!g_stream_jpeg) vTaskDelay(1);
  }
  g_stream_clients--;
  return res;
}

static void start_camera_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  // Same reasoning as the stream server below, though this one is far less
  // exposed: its handlers all return promptly, so a stale socket here only
  // occupies a slot rather than blocking every other request.
  config.lru_purge_enable = true;
  config.max_uri_handlers = 5;
  config.stack_size = 8192;  // default 4096 is tight; index/status/config are light

  httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL};
  httpd_uri_t status_uri = {
      .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL};
  httpd_uri_t config_uri = {
      .uri = "/config", .method = HTTP_GET, .handler = config_handler, .user_ctx = NULL};

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
    httpd_register_uri_handler(camera_httpd, &config_uri);
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = STREAM_PORT;
  stream_config.ctrl_port = 32769;
  stream_config.max_uri_handlers = 2;
  // Much smaller than the DCT firmware's 49152 because this model's ~619KB of
  // intermediates are on the heap, not the stack (see alloc_model_buffers()).
  // What is left on this stack is ~6.4KB of handler locals (two 3072-byte
  // JSON/status buffers plus two 192-byte headers) and whatever frame2jpg()'s
  // encoder uses. 20480 keeps a wide margin over that.
  stream_config.stack_size = 20480;

  // WHY THESE THREE: stream_handler() below never returns while a client is
  // connected -- it is an infinite loop that also does the capture,
  // classification and JPEG encode. esp_http_server services requests from a
  // single task, so that one handler monopolises this whole server.
  //
  // The failure that produced: after a reflash the browser's PREVIOUS /stream
  // connection is dead at the TCP level, but the ESP32 does not know it. The
  // handler keeps spinning on the stale socket and the reloaded page's new
  // request is never serviced -- port 80 still serves INDEX_HTML, so the page
  // loads as a skeleton with no video and no predictions, which reads as a
  // model or camera fault rather than a socket one. Observed 2026-08-22:
  // `curl :81/stream` connected, sent the request, and received 0 bytes in
  // 12s while /status simultaneously reported a healthy 19.4 fps.
  //
  // lru_purge_enable is the one that fixes it: a new connection evicts the
  // oldest instead of queueing behind it, so a page reload always wins. The
  // timeouts make a dead socket error out promptly rather than hang forever.
  // 5s specifically, not less -- the DCT firmware measured sends that block
  // 0.94-1.66s and still complete, so a 1s timeout kills healthy connections
  // (that was esp32_classifier's second stall cause; see markdown/fix_stall.md).
  stream_config.lru_purge_enable = true;
  stream_config.send_wait_timeout = 5;
  stream_config.recv_wait_timeout = 5;

  httpd_uri_t stream_uri = {
      .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

static bool init_camera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // RGB565 at 160x120 -- exactly the model's input resolution
  // (network_info.json: downsample_factor=1, rgb 160x120), so there is no
  // on-device resize and therefore no need to reproduce the PC pipeline's
  // LANCZOS resampling, which would have been an unanswerable question.
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;  // 160x120 -> 38,400 byte frames
  config.jpeg_quality = SENSOR_JPEG_QUALITY;  // unused for RGB565 capture; frame2jpg carries its own

  // The DCT firmware initialised at QVGA and shrank to QQVGA afterwards with
  // set_framesize(). That trick is NOT valid here. It worked only because
  // JPEG frames are self-delimited, so the driver tolerated the sensor
  // emitting fewer bytes than the buffer was sized for. An RGB565 frame is a
  // fixed 2*W*H with no end marker, so the frame buffer and DMA descriptors
  // must be sized for the real capture size at init. Hence: init directly at
  // the size we actually want, and assert on fb->len per frame in
  // stream_handler() rather than assuming.
  if (psramFound()) {
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_framesize(sensor, FRAMESIZE_QQVGA);

    sensor->set_whitebal(sensor, 1);  // auto white balance on
    sensor->set_awb_gain(sensor, 1);
    sensor->set_wb_mode(sensor, 0);        // 0 = auto
    sensor->set_exposure_ctrl(sensor, 1);  // auto exposure on
    sensor->set_gain_ctrl(sensor, 1);      // auto gain on

    // DSP-based AEC off, deliberately, and this one is inherited on purpose:
    // with it ON, the OV2640's anti-banding / frame-rate-reduction behaviour
    // pinned T_capture at a rigid ~101-102ms (~6x the 60Hz mains flicker
    // period) regardless of how fast inference was -- confirmed by ruling out
    // JPEG quality and xclk_freq_hz first. That would put a hard ~9.9fps
    // ceiling on this firmware and silently confound the entire latency
    // comparison. Trade-off: capture time becomes variable, and banding can
    // appear under flickering light. See capture_time_issue.md.
    sensor->set_aec2(sensor, 0);

    // NOT inherited: the DCT firmware also ran ae_level=-1 and saturation=+2.
    // Both were compensations for DCT-feature pathologies (a DC coefficient
    // sitting near its quantized maximum, and chroma coefficients collapsing
    // to near zero) and neither has any counterpart here. This model was
    // trained on ordinary photographs, so deliberately darkening exposure and
    // pushing saturation to maximum would be a gratuitous domain shift
    // between training and deployment. Left at neutral.
    sensor->set_ae_level(sensor, 0);
    sensor->set_saturation(sensor, 0);
    sensor->set_special_effect(sensor, 0);  // rule out an accidental grayscale/tint mode
  }

  return true;
}

static void connect_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to WiFi \"%s\"", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, IP address: ");
  Serial.println(WiFi.localIP());

  // Re-assert no-power-save AFTER association, and verify it took. In
  // modem-sleep the station only wakes on DTIM beacons, so inbound packets
  // sit buffered at the AP for hundreds of ms -- exactly the latency profile
  // that turns a smooth stream into "20fps then drops to zero". The
  // setSleep(false) above runs BEFORE WiFi.begin(); that ordering is not
  // reliably honoured across arduino-esp32 versions.
  WiFi.setSleep(false);
  wifi_ps_type_t ps = WIFI_PS_NONE;
  if (esp_wifi_get_ps(&ps) == ESP_OK) {
    Serial.printf("WiFi power save mode: %s\n",
                  ps == WIFI_PS_NONE      ? "NONE (good, no DTIM latency)"
                  : ps == WIFI_PS_MIN_MODEM ? "MIN_MODEM (!! adds DTIM-interval latency)"
                                            : "MAX_MODEM (!! adds large DTIM latency)");
  } else {
    Serial.println("WiFi power save mode: (esp_wifi_get_ps failed)");
  }
}

// Numerics check for the generated model_forward(), against the PC-side int8
// reference. See rgb_test_vectors.h for the three tiers and why they carry
// very different weight.
//
// Reports the max absolute logit difference alongside pass/fail, because the
// magnitude is diagnostic: a mirror that is off by 1-2 LSB on one class
// suggests a requantization detail, while one off by tens of LSB suggests a
// geometry error (row pitch, padding, stride phase, plane indexing).
static int compare_logits(const int8_t *got, const int8_t *want, int *max_diff_out) {
  int max_diff = 0;
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    int d = abs((int)got[c] - (int)want[c]);
    if (d > max_diff) max_diff = d;
  }
  *max_diff_out = max_diff;
  return max_diff == 0;
}

// Folds one vector's result into the cross-tier class-match and worst-drift
// totals. Kept separate from the bit-exact counts because on an ESP-NN build
// those two can legitimately diverge -- see self_test_status().
static void record_result(const int8_t *got, const int8_t *want, int max_diff) {
  if (max_diff > g_selftest_max_diff) g_selftest_max_diff = max_diff;
  if (model_argmax(got, MODEL_NUM_CLASSES) == model_argmax(want, MODEL_NUM_CLASSES))
    g_selftest_class_ok++;
}

static void print_logits(const int8_t *logits) {
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) Serial.printf(" %d", (int)logits[c]);
  Serial.println();
}

static void run_model_self_test() {
  Serial.printf("=== model_forward() numerics self-test (ESP-NN %s) ===\n",
                BUILD_HAS_ESP_NN ? "ENABLED" : "disabled");
  uint8_t *rgb = g_rgb888;
  int8_t logits[MODEL_NUM_CLASSES];
  int max_diff = 0;
  g_selftest_class_ok = 0;
  g_selftest_max_diff = 0;

  // Determinism first: the same input twice must give the same logits. This
  // catches an intermediate buffer being read before it is written, which is
  // THE characteristic failure mode when stack locals are hoisted into
  // long-lived reused heap buffers, and which a single-shot comparison
  // against expected values would miss entirely (the stale data would be
  // consistent within one boot).
  {
    int8_t first[MODEL_NUM_CLASSES], second[MODEL_NUM_CLASSES];
    synth_fill(SYNTH_SEEDS[0], rgb);
    model_forward_locked(rgb, first, NULL);
    memset(rgb, 0xA5, SZ_RGB888);  // scribble between runs so a stale read shows up
    synth_fill(SYNTH_SEEDS[0], rgb);
    model_forward_locked(rgb, second, NULL);
    g_selftest_determinism_ok = memcmp(first, second, MODEL_NUM_CLASSES) == 0;
    Serial.printf("[determinism] same input twice -> same logits: %s\n",
                  g_selftest_determinism_ok ? "PASS" : "FAIL");
  }

  // Tier 2 (primary): structured patterns. Strong spatial structure and
  // per-channel asymmetry, so a geometry error cannot average itself away.
  int ok = 0;
  for (int t = 0; t < PATTERN_NUM_VECTORS; t++) {
    pattern_fill(t, rgb);
    model_forward_locked(rgb, logits, NULL);
    int exact_p = compare_logits(logits, PATTERN_EXPECTED_LOGITS[t], &max_diff);
    record_result(logits, PATTERN_EXPECTED_LOGITS[t], max_diff);
    if (exact_p) {
      ok++;
      Serial.printf("[pattern] PASS %-7s bit-exact\n", PATTERN_NAMES[t]);
    } else {
      Serial.printf("[pattern] FAIL %-7s max logit diff %d\n  got: ", PATTERN_NAMES[t], max_diff);
      print_logits(logits);
      Serial.print("  want:");
      print_logits(PATTERN_EXPECTED_LOGITS[t]);
    }
  }
  g_selftest_pattern_ok = ok;

  // Tier 1: LCG noise. Cheap smoke test only -- all four seeds pool to nearly
  // the same feature vector, so passing these proves very little on its own.
  ok = 0;
  for (int t = 0; t < SYNTH_NUM_VECTORS; t++) {
    synth_fill(SYNTH_SEEDS[t], rgb);
    model_forward_locked(rgb, logits, NULL);
    int exact_s = compare_logits(logits, SYNTH_EXPECTED_LOGITS[t], &max_diff);
    record_result(logits, SYNTH_EXPECTED_LOGITS[t], max_diff);
    if (exact_s) {
      ok++;
      Serial.printf("[synth]   PASS seed %-5u bit-exact\n", (unsigned)SYNTH_SEEDS[t]);
    } else {
      Serial.printf("[synth]   FAIL seed %-5u max logit diff %d\n  got: ", (unsigned)SYNTH_SEEDS[t],
                    max_diff);
      print_logits(logits);
    }
  }
  g_selftest_synth_ok = ok;

  // Tier 3: real photographs, both correctly classified by the PC-side int8
  // model -- so a mismatch means this port is wrong, not that the model is.
  ok = 0;
  for (int t = 0; t < REAL_NUM_VECTORS; t++) {
    model_forward_locked(REAL_IMAGES[t], logits, NULL);
    int pred = model_argmax(logits, MODEL_NUM_CLASSES);
    bool exact = compare_logits(logits, REAL_EXPECTED_LOGITS[t], &max_diff);
    record_result(logits, REAL_EXPECTED_LOGITS[t], max_diff);
    if (exact) ok++;
    Serial.printf("[real]    %s %-9s pred=%s expected=%s%s\n", exact ? "PASS" : "FAIL",
                  REAL_CLASS_NAMES[t], MODEL_CLASS_NAMES[pred],
                  MODEL_CLASS_NAMES[REAL_EXPECTED_CLASS[t]], exact ? " (bit-exact)" : "");
    if (!exact) {
      Serial.printf("  max logit diff %d\n  got: ", max_diff);
      print_logits(logits);
      Serial.print("  want:");
      print_logits(REAL_EXPECTED_LOGITS[t]);
    }
  }
  g_selftest_real_ok = ok;

  Serial.printf(
      "self-test: %s -- bit-exact: pattern %d/%d, synth %d/%d, real %d/%d; "
      "class match %d/%d; max|logit diff| %d; determinism %s\n",
      self_test_status(), g_selftest_pattern_ok, PATTERN_NUM_VECTORS, g_selftest_synth_ok,
      SYNTH_NUM_VECTORS, g_selftest_real_ok, REAL_NUM_VECTORS, g_selftest_class_ok,
      SELFTEST_TOTAL_VECTORS, g_selftest_max_diff, g_selftest_determinism_ok ? "ok" : "FAILED");
  if (strcmp(self_test_status(), "pass") != 0) {
    Serial.println(
        "  !! model_forward() does NOT agree with the PC int8 reference.\n"
        "     Every timing number below is measuring the wrong computation.\n");
  } else if (g_selftest_bit_exact_all) {
    Serial.println("  Forward pass is bit-exact vs the PC int8 reference.\n");
  } else {
    Serial.println(
        "  Predicted class matches the PC int8 reference on every vector, with small\n"
        "  logit drift -- expected on an ESP-NN build (its two-step requantize rounds\n"
        "  differently from the portable single combined round-shift). Class agreement,\n"
        "  not bit-exactness, is the correctness bar here.\n");
  }
}

// Scene-independent inference benchmark. The live per-frame numbers are real
// but depend on what the camera is pointed at only through the data (int8
// conv cost is data-independent) and, much more importantly, on contention
// with WiFi/the stream. This runs the forward pass on a fixed synthetic input
// with no networking in flight, so a comparison across two flashed builds is
// controlled rather than confounded -- the same role run_decode_benchmark()
// played in the DCT firmware.
#define BENCH_ITERS 5
static void run_inference_benchmark() {
  uint8_t *rgb = g_rgb888;
  synth_fill(SYNTH_SEEDS[0], rgb);

  uint64_t acc[MODEL_NUM_STAGES] = {0};
  int64_t acc_total = 0;
  for (int i = 0; i < BENCH_ITERS; i++) {
    int8_t logits[MODEL_NUM_CLASSES];
    uint32_t stage_us[MODEL_NUM_STAGES];
    int64_t t0 = esp_timer_get_time();
    model_forward_locked(rgb, logits, stage_us);
    acc_total += esp_timer_get_time() - t0;
    for (int s = 0; s < MODEL_NUM_STAGES; s++) acc[s] += stage_us[s];
  }
  const float d = 1000.0f * BENCH_ITERS;
  g_bench_quantize_ms = acc[MODEL_STAGE_QUANTIZE] / d;
  g_bench_conv0_ms = acc[MODEL_STAGE_CONV0] / d;
  g_bench_conv1_ms = acc[MODEL_STAGE_CONV1] / d;
  g_bench_conv2_ms = acc[MODEL_STAGE_CONV2] / d;
  g_bench_conv3_ms = acc[MODEL_STAGE_CONV3] / d;
  g_bench_pool_ms = acc[MODEL_STAGE_POOL] / d;
  g_bench_output_ms = acc[MODEL_STAGE_DENSE] / d;
  g_bench_total_ms = acc_total / d;
  g_bench_contended = g_stream_clients > 0;

  Serial.printf("=== Inference benchmark (%d iters, fixed synthetic input) ===\n", BENCH_ITERS);
  if (g_stream_clients > 0) {
    Serial.printf("  !! %d stream client(s) connected -- NOT an uncontended measurement.\n"
                  "     The stream task still captures and JPEG-encodes each frame even with\n"
                  "     classification suspended. Disconnect and reboot for the reference number.\n",
                  g_stream_clients);
  }
  Serial.printf("  quantize %.2f ms\n", g_bench_quantize_ms);
  // With the NHWC header there is no CHW<->NHWC repack left inside
  // conv2d_int8, so each conv stage IS the kernel -- these MMAC/s figures are
  // now directly comparable to the repack-excluded kernel column measured on
  // the previous CHW build (conv0 41.9, conv1 169.1, conv2 270.3, conv3 403.3).
  Serial.printf("  conv0    %7.2f ms  (3->16ch,  120x160,  8.29M MAC) %7.1f MMAC/s\n",
                g_bench_conv0_ms, 8.2944f * 1000.0f / g_bench_conv0_ms);
  Serial.printf("  conv1    %7.2f ms  (16->32ch, 60x80,   22.12M MAC) %7.1f MMAC/s\n",
                g_bench_conv1_ms, 22.1184f * 1000.0f / g_bench_conv1_ms);
  Serial.printf("  conv2    %7.2f ms  (32->64ch, 30x40,   22.12M MAC) %7.1f MMAC/s\n",
                g_bench_conv2_ms, 22.1184f * 1000.0f / g_bench_conv2_ms);
  Serial.printf("  conv3    %7.2f ms  (64->32ch, 30x40,   22.12M MAC) %7.1f MMAC/s\n",
                g_bench_conv3_ms, 22.1184f * 1000.0f / g_bench_conv3_ms);
  Serial.printf("  pool     %.2f ms\n", g_bench_pool_ms);
  Serial.printf("  dense    %.2f ms\n", g_bench_output_ms);
  Serial.printf("  TOTAL    %.2f ms  -> %.2f inferences/sec (inference only)\n", g_bench_total_ms,
                1000.0f / g_bench_total_ms);
  // conv0 is only ~11% of the model's MACs. If it takes far more than ~11% of
  // the time, that is a memory-bandwidth/cache result (full 120x160 extent,
  // only 3 input channels, 27-term inner loop, output in PSRAM), not a
  // compute result -- which is exactly the number this comparison wants.
  if (g_bench_total_ms > 0.0f) {
    Serial.printf("  conv0 share: %.1f%% of inference time vs ~11%% of MACs\n\n",
                  100.0f * g_bench_conv0_ms / g_bench_total_ms);
  }
}


void setup() {
  Serial.begin(115200);
  delay(500);

  if (!init_camera()) {
    Serial.println("Camera init failed, halting.");
    while (true) delay(1000);
  }

  connect_wifi();

  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("mDNS responder started: http://%s.local/\n", MDNS_HOSTNAME);
  } else {
    Serial.println("Error starting mDNS responder");
  }

  // After WiFi (which is the large internal-SRAM consumer) so the placement
  // decision in alloc_model_buffers() sees the real steady-state free heap,
  // but before the servers start so no request can arrive with the buffers
  // still unallocated.
  if (!alloc_model_buffers()) {
    Serial.println("Model buffer allocation failed -- classification disabled.");
  }
  g_model_mutex = xSemaphoreCreateMutex();
  if (!g_model_mutex) Serial.println("WARNING: model mutex creation failed.");

  start_camera_server();
  Serial.println("Camera stream ready.");

  // Run after networking is up, not before: gating WiFi/the web server behind
  // ~30 lines of Serial output risked a boot hang with no host serial reader
  // attached (which happened once, forcing a reflash).
  if (g_bufs_ready) {
    g_selftest_running = true;
    run_model_self_test();
    run_inference_benchmark();
    g_selftest_running = false;
    Serial.println("Boot self-test complete -- live classification enabled.\n");
  }
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    connect_wifi();
  }
  delay(1000);
}
