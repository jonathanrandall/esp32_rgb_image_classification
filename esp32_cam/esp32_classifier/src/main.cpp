#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_log.h"      // esp_log_level_set() -- see init_camera()
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "lwip/sockets.h"   // setsockopt/TCP_NODELAY on the stream socket, see stream_handler()
#include "esp_wifi.h"       // esp_wifi_get_ps() -- verify power save is really off, see connect_wifi()
#include "esp_system.h"     // esp_reset_reason() -- why the last boot happened, see reset_reason_name()
#include <math.h>

#include "camera_pins.h"
#include "secrets.h"
#include "dct_features.h"
#include "model_weights.h"

// DCT_NUM_AC_COEFFS (hand-written, dct_features.h) sizes the ac_planes buffers
// that dct_extract_coeffs() fills. MODEL_NUM_AC_COEFFS (exporter-generated,
// model_weights.h) is how many planes model_forward() INDEXES. If the model's
// count is larger, model_forward() reads past the end of a stack array and
// classifies on whatever happened to be there.
//
// This bit us on 2026-08-20: several models were retrained at 3/5/7/10 AC
// coefficients, exported, verified bit-exact on the PC, flashed -- and
// classified badly on camera, while every offline metric stayed healthy,
// because only model_weights.h had been updated. The bug is silent by
// construction: the export is correct, the verification is correct, the build
// succeeds, and the self-test passes (it feeds model_forward() directly from
// TEST_AC_RAW, never through the decoder's undersized buffer).
//
// A retrain at a different --num-ac-coeffs now fails the build instead.
static_assert(DCT_NUM_AC_COEFFS == MODEL_NUM_AC_COEFFS,
              "dct_features.h's DCT_NUM_AC_COEFFS must equal model_weights.h's "
              "MODEL_NUM_AC_COEFFS -- the decoder fills that many AC planes and "
              "model_forward() reads that many. Update dct_features.h to match "
              "the freshly exported model_weights.h.");

// Same class of mismatch on the chroma side. The firmware decoder implements
// chroma DC only (see esp32_chroma_extraction_prompt.md); a model trained with
// --num-chroma-ac-coeffs > 0 needs real new decoder code, not just a re-export.
// The same silent-mismatch class as the coefficient count, on the grid itself:
// the decoder fills DCT_Y_ROWS x DCT_Y_COLS and model_forward() reads
// MODEL_Y_ROWS x MODEL_Y_COLS. A model trained at a different capture
// resolution would otherwise be fed a wrongly-shaped buffer with no warning.
static_assert(DCT_Y_ROWS == MODEL_Y_ROWS && DCT_Y_COLS == MODEL_Y_COLS,
              "dct_features.h's luma grid must match model_weights.h's. Update "
              "DCT_Y_ROWS/DCT_Y_COLS, and the sensor->set_framesize() call.");
static_assert(DCT_C_ROWS == MODEL_C_ROWS && DCT_C_COLS == MODEL_C_COLS,
              "dct_features.h's chroma grid must match model_weights.h's.");

static_assert(MODEL_NUM_CHROMA_COEFFS == 1,
              "This firmware's dct_extract_coeffs() extracts chroma DC only. "
              "Retrain with --num-chroma-ac-coeffs 0, or implement chroma AC "
              "extraction in dct_features.cpp first.");
#include "test_vectors.h"
// real_test_jpegs.h embeds three real camera JPEGs together with the
// dc_plane/ac_planes/chroma_planes the host-compiled decoder produced for them.
// Those expectations are baked at a specific grid and coefficient count
// (160x120, 3 AC as of 2026-09-04), so the header only compiles against a
// matching build. To move to another coefficient count or resolution:
// set DCT_NUM_AC_COEFFS in dct_features.h, then
//   g++ -O2 -I src -I scratchpad -o gen scratchpad/gen_real_test_header.cpp \
//       src/dct_features.cpp && ./gen > include/real_test_jpegs.h
// and update the count in the guard below. The generator reuses the same
// embedded JPEG bytes (scratchpad/jpeg_bytes_only.h), so regenerating changes
// only the expected planes -- the DC and chroma planes must come out
// byte-identical, which is the cheapest check that a regeneration went right.
#define REAL_TEST_JPEGS_MATCH (DCT_Y_ROWS == 15 && DCT_Y_COLS == 20 && DCT_NUM_AC_COEFFS == 3)
#if REAL_TEST_JPEGS_MATCH
#include "real_test_jpegs.h"
#endif

// ESP-NN integration (per full_esp32_integration.md) lives inside
// model_weights.h's own conv2d_int8() -- macro-guarded on
// CONFIG_NN_OPTIMIZED + CONFIG_IDF_TARGET_ESP32S3 (platformio.ini). All
// four conv layers go through esp_nn_conv_s8 unconditionally. The one
// real hardware gotcha found during integration: the esp32s3 assembly
// requires the scratch buffer POINTER to be 16-byte aligned, which the
// generated model_esp_nn_init() satisfies via heap_caps_aligned_alloc --
// see esp32_nn_testing.md for the full (mis)diagnosis-and-fix history.

// Device will be reachable at http://esp32cam.local/
#define MDNS_HOSTNAME "esp32cam_dct"
#define STREAM_PORT 81

// esp_http_server runs each instance on a single worker task. The /stream
// handler below never returns while a client is connected, so it must live
// on its own httpd instance (a different port) or it would block the "/"
// and "/status" handlers from ever being serviced.
static httpd_handle_t camera_httpd = NULL;
static httpd_handle_t stream_httpd = NULL;

// How many frames per second are actually captured and run through the CNN
// -- updated on every call to capture_and_classify_one_frame(), regardless
// of whether that frame ends up sent to a client. Read by status_handler as
// "classify_fps". Do NOT report this as the stream's frame rate: after the
// 20fps send cap in stream_handler() went in, this kept reading the old
// uncapped ~25-47fps, because capture/inference never slowed down -- only
// what gets sent did. See g_sent_fps below for what's actually delivered.
static volatile float g_fps = 0.0f;

// How many frames per second are actually written to a client's socket --
// updated by mark_frame_sent(), called after a successful send in both
// stream_handler()'s loop and frame_handler()'s one-shot sends (mirrors
// g_fps's endpoint-agnostic design above). This -- not g_fps -- is what
// /status reports as "fps" and what the page displays, since it's the only
// one of the two that matches what the browser is actually receiving.
static volatile float g_sent_fps = 0.0f;

// Timestamp of the most recent successful send, mirroring g_last_frame_us --
// same reasoning: without it, g_sent_fps would report its last nonzero value
// forever once sends stop (client disconnected, nobody streaming), instead
// of honestly reporting 0.
static volatile int64_t g_last_sent_us = 0;

// Timestamp of the most recent captured frame. g_fps alone can't express
// "stopped": it's only recomputed when a frame is produced, so if capture
// stops completely g_fps keeps reporting whatever it last measured, forever.
// That's what made a wedged stream look like a healthy one in /status. This
// makes staleness explicit, and the page uses it to decide the stream died
// and reload the iframe (see INDEX_HTML's watchdog).
static volatile int64_t g_last_frame_us = 0;

// ---- /sendlog: per-frame send telemetry --------------------------------
//
// g_fps and g_sent_fps are one-second averages, which is far too coarse to
// see a stall: by the time a rate drops, the interesting event is already
// over. This records one row per frame as it is written to the socket, so
// the shape of a stall can be read afterwards instead of inferred.
//
// The single number that matters here is send_us, measured around the two
// httpd_resp_send_chunk() calls. It separates the three failure modes that
// look identical from the browser:
//
//   send_us spikes to 100s of ms  -> socket is backing up, peer not draining
//   send_us fine, rows stop       -> capture/classify stalled, or handler exited
//   send_us fine, rows continue   -> frames leave the device; loss is past it
//
// Deliberately fixed-size and allocation-free: this runs in the streaming
// hot path, and an instrument that perturbs what it measures is worse than
// no instrument. 128 rows is ~5s at 27fps -- enough to cover a stall and
// its onset. Written only by the stream worker task, read best-effort by
// the port-80 server; a torn row during a read is acceptable and cannot
// corrupt anything, since readers only ever format the fields.
#define SENDLOG_SLOTS 128
struct send_rec {
  int64_t  t_us;      // esp_timer_get_time() immediately before the first write
  uint32_t send_us;   // duration of both httpd_resp_send_chunk() calls
  uint32_t bytes;     // fb->len, the JPEG payload size
  int32_t  res;       // esp_err_t from the send (0 == ESP_OK)
  uint16_t conn;      // which stream_handler() invocation produced this row
  int16_t  rssi;      // dBm, sampled at most once a second (see below)
};
static send_rec g_sendlog[SENDLOG_SLOTS];
static volatile uint32_t g_sendlog_next = 0;   // total rows ever written; index = n % SLOTS
static volatile uint32_t g_stream_conns = 0;   // stream_handler() entries since boot
static volatile uint16_t g_sendlog_conn = 0;   // invocation number of the live handler

// STREAM TAKEOVER. Bumped by GET /claim on port 80; the running stream handler
// watches it and returns when it changes, freeing port 81 for the new client.
//
// WHY THIS EXISTS. stream_handler() below never returns while a client is
// connected, and esp_http_server services a server's requests from ONE task.
// So the port-81 task is inside that handler and cannot accept -- a second
// connection is never serviced, and there is nothing the server can do about
// it. That is not a timeout problem: measured on the sibling RGB firmware
// 2026-08-22, Firefox held two ESTAB sockets on :81 (connection pooling keeps
// an abandoned stream socket open), the stale one owned the handler, and the
// visible tab got nothing until Firefox was quit entirely. Closing the tab did
// not help. Nothing about this firmware makes it immune -- same server, same
// single-task structure.
//
// lru_purge_enable does NOT solve this, despite being set in
// start_camera_server() and despite max_open_sockets being lowered to 4. It
// fires only when max_open_sockets is exhausted, and it runs on the accept
// path -- the same blocked task, which never reaches accept.
// send_wait_timeout does not either: writes to a live-but-ignored socket
// succeed, so nothing ever errors.
//
// The way out is that port 80 is a SEPARATE httpd instance with its own task,
// whose handlers all return promptly. It is always responsive, so it can carry
// the "I want the stream" signal that port 81 cannot hear.
static volatile uint32_t g_stream_claim = 0;

// An incumbent ignores claims for this long after it starts, so two clients
// that both claim cannot kick each other in a tight loop. The claim is a
// counter, not an edge, so it is not lost during the grace period -- only
// deferred.
#define STREAM_CLAIM_GRACE_US 1500000

static void sendlog_record(int64_t t_us, uint32_t send_us, uint32_t bytes, esp_err_t res) {
  // RSSI is sampled at most once a second rather than per frame: it is a
  // driver call, this is the hot path, and the link does not change on a
  // 37ms timescale anyway. Correlating a stall with a degrading link only
  // needs second-resolution.
  static int64_t rssi_sampled_us = 0;
  static int16_t rssi_cached = 0;
  if (rssi_sampled_us == 0 || t_us - rssi_sampled_us > 1000000) {
    rssi_cached = (int16_t)WiFi.RSSI();
    rssi_sampled_us = t_us;
  }

  uint32_t n = g_sendlog_next;
  send_rec *r = &g_sendlog[n % SENDLOG_SLOTS];
  r->t_us    = t_us;
  r->send_us = send_us;
  r->bytes   = bytes;
  r->res     = (int32_t)res;
  r->conn    = g_sendlog_conn;
  r->rssi    = rssi_cached;
  // Published last, so a reader that sees the incremented count is looking
  // at a fully written row rather than a half-filled one.
  g_sendlog_next = n + 1;
}

// Latest classification result + per-stage timing, updated by stream_handler
// once per second, read by status_handler. Raw int8 logits (not a
// probability distribution) -- status_handler applies softmax at request
// time so the hot per-frame loop doesn't pay for float exp() math it
// doesn't need for anything else.
static volatile int8_t g_last_logits[MODEL_NUM_CLASSES] = {0};
static volatile bool g_has_classified = false;
static volatile float g_t_capture_ms = 0.0f;
static volatile float g_t_parse_ms = 0.0f;
static volatile float g_t_infer_ms = 0.0f;
static volatile float g_t_total_ms = 0.0f;

// Scene-independent T_jpeg_parse benchmark, set once at boot by
// run_decode_benchmark() and reported by /status. The live per-frame
// t_parse_ms above is real but scene-dependent (Huffman code lengths vary
// with image content, so decode time varies with whatever the camera
// happens to be pointed at) -- this repeats dct_extract_coeffs() many
// times over the fixed embedded real_test_jpegs.h corpus instead, so a
// before/after comparison across two flashed builds (e.g.
// DCT_FAST_HUFFMAN=1 vs 0) is a controlled A/B, not confounded by the
// camera seeing a different scene at each measurement.
static volatile float g_bench_parse_us = 0.0f;

// Full-pipeline correctness report, set once at boot by
// run_model_forward_self_test() and reported by /status. model_forward()
// now runs through an ESP-NN-accelerated conv2d_int8() (all 4 conv
// layers) when CONFIG_NN_OPTIMIZED+CONFIG_IDF_TARGET_ESP32S3 are set --
// deliberately not bit-exact against TEST_EXPECTED_LOGITS by design (see
// full_esp32_integration.md), so this tracks both the expected small
// bit-level drift (informational) and the real correctness bar: does the
// predicted class still match TEST_EXPECTED_CLASS.
static volatile int g_model_bit_exact_count = 0;
static volatile int g_model_class_match_count = 0;
static volatile int g_model_max_logit_diff = 0;

// Pass counts from the other two boot-time self-tests (decoder correctness
// -- now includes chroma_planes -- and model_forward_timed()'s bit-exact
// mirror check), reported by /status alongside the above so on-device
// self-test results are verifiable over the network without a serial
// connection (touching /dev/ttyACM0 resets the board into USB download
// mode on this host -- see esp32_nn_testing.md/project notes).
static volatile int g_decoder_self_test_pass = 0;
static volatile int g_decoder_self_test_total = 0;
static volatile int g_stage_timing_self_test_pass = 0;
static volatile int g_stage_timing_self_test_total = 0;

// Per-conv-stage timing within model_forward(), updated by stream_handler
// once per second alongside the other timing globals, read by
// status_handler. Named after the actual deployed CNN's stage sequence
// (model_weights.h's model_forward() body) rather than the MLP-era
// T_dense1/T_dense2 naming in esp32_cnn_implementation_prompt.md/
// minimal_decoder.md's Benchmarking plan -- this model doesn't have two
// dense layers, it has this conv stack plus one final dense output layer.
struct StageTimingsUs {
  int64_t quant_dc;         // dequantized DC plane -> int8
  int64_t lum_conv;         // lum_conv (3x3) on the luma plane
  int64_t stride2_conv;     // stride2_conv (3x3, stride 2) + copy into concat_buf
  int64_t ac_quant_pool;    // AC planes -> int8, avg-pooled to stride2's grid, copied into concat_buf
  int64_t chroma_quant_pool;// chroma DC planes -> int8, avg-pooled to stride2's grid, copied into concat_buf
  int64_t post_concat_conv; // post_concat_conv (3x3) on the concatenated features
  int64_t extra_conv0;      // extra_conv_0 (3x3)
  int64_t final_pool;       // global average pool -> per-channel feature vector
  int64_t output;           // final dense layer -> raw int8 logits
};
static volatile float g_t_stage_quant_dc_ms = 0.0f;
static volatile float g_t_stage_lum_conv_ms = 0.0f;
static volatile float g_t_stage_stride2_conv_ms = 0.0f;
static volatile float g_t_stage_ac_quant_pool_ms = 0.0f;
static volatile float g_t_stage_chroma_quant_pool_ms = 0.0f;
static volatile float g_t_stage_post_concat_conv_ms = 0.0f;
static volatile float g_t_stage_extra_conv0_ms = 0.0f;
static volatile float g_t_stage_final_pool_ms = 0.0f;
static volatile float g_t_stage_output_ms = 0.0f;

// Instrumented mirror of model_weights.h's model_forward() -- same
// sequence of calls into the same generated primitives/weight arrays
// (conv2d_int8, avg_pool_int8, model_quantize_dc/ac, the LUM_*/STRIDE2_*/
// POST_CONCAT_*/EXTRA_CONV_0_*/OUTPUT_* arrays), just with
// esp_timer_get_time() calls between stages. Not a reimplementation of the
// math -- every actual computation still goes through the exact
// auto-generated building blocks, so this can't silently diverge from
// model_forward()'s *numerics*. It CAN silently diverge from
// model_forward()'s *shape* if a future retrain changes the architecture
// (e.g. extra_conv_channels growing past one layer would add an
// EXTRA_CONV_1_* stage model_forward() would pick up automatically but
// this copy wouldn't) -- that's why run_model_forward_self_test() checks
// this produces bit-identical logits to model_forward() on all 25 known
// vectors every boot, not just once during development. model_weights.h
// itself is explicitly "DO NOT EDIT BY HAND" (regenerated by
// export_cnn_c_weights.py), so instrumenting model_forward() in place
// isn't an option -- any hand edit there would be silently destroyed the
// next time a retrained model_weights.h gets copied in.
static void model_forward_timed(
    const int32_t dc_plane[MODEL_Y_ROWS][MODEL_Y_COLS],
    const int32_t ac_planes[MODEL_NUM_AC_COEFFS][MODEL_Y_ROWS][MODEL_Y_COLS],
    const int32_t chroma_planes[2 * MODEL_NUM_CHROMA_COEFFS][MODEL_C_ROWS][MODEL_C_COLS],
    int8_t logits[MODEL_NUM_CLASSES],
    StageTimingsUs &st)
{
  int64_t t0 = esp_timer_get_time(), t1;

  // The activation buffers below are `static`, not stack locals. At 160x120
  // they summed to ~24KB and fitted comfortably; at 320x240 they are ~75KB and
  // overflowed the httpd task's stack ("***ERROR*** A stack overflow in task
  // httpd"), which cannot simply be raised -- the stream task's stack and the
  // Arduino loop task's both come out of the same ~320KB of internal SRAM,
  // alongside WiFi. Moving them to .bss costs the same RAM but takes it out of
  // the per-task stack budget, where it does not fit, and puts it somewhere it
  // does.
  //
  // Safe because exactly one task ever calls this: the stream server has a
  // single worker, and the boot self-tests that used to call it on the Arduino
  // loop task are skipped at grids this size (SELF_TEST_FITS_ON_LOOP_TASK).
  // If a second caller is ever added, these must become per-task again.
  static int8_t q_dc[MODEL_Y_ROWS * MODEL_Y_COLS];
  for (int y = 0; y < MODEL_Y_ROWS; y++)
    for (int x = 0; x < MODEL_Y_COLS; x++)
      q_dc[y * MODEL_Y_COLS + x] = model_quantize_dc(dc_plane[y][x]);
  t1 = esp_timer_get_time(); st.quant_dc = t1 - t0; t0 = t1;

  static int8_t lum_out[MODEL_LUM_CHANNELS * MODEL_Y_ROWS * MODEL_Y_COLS];
  conv2d_int8(q_dc, 1, MODEL_Y_ROWS, MODEL_Y_COLS,
              LUM_WEIGHT, LUM_BIAS, LUM_MULT, LUM_SHIFT,
              MODEL_LUM_CHANNELS, 3, 3, 1, 1, 0, 127, lum_out);
  t1 = esp_timer_get_time(); st.lum_conv = t1 - t0; t0 = t1;

  static int8_t stride2_out[MODEL_STRIDE2_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];
  conv2d_int8(lum_out, MODEL_LUM_CHANNELS, MODEL_Y_ROWS, MODEL_Y_COLS,
              STRIDE2_WEIGHT, STRIDE2_BIAS, STRIDE2_MULT, STRIDE2_SHIFT,
              MODEL_STRIDE2_CHANNELS, 3, 3, 2, 1, 0, 127, stride2_out);

  static int8_t concat_buf[MODEL_POST_CONCAT_IN * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];
  memcpy(concat_buf, stride2_out, sizeof(stride2_out));
  int concat_offset = MODEL_STRIDE2_CHANNELS;
  t1 = esp_timer_get_time(); st.stride2_conv = t1 - t0; t0 = t1;

  // Guarded, not just loop-bounded: at MODEL_NUM_AC_COEFFS == 0 the exporter
  // emits no model_quantize_ac() at all (there are no AC scales to emit), so
  // the call below would fail to compile even though the loop would run zero
  // times. Zero-length q_ac/ac_pooled arrays are also not standard C++. The
  // stage timer still fires so /status keeps reporting the same nine stages
  // for every model, just with ac_quant_pool at ~0.
#if MODEL_NUM_AC_COEFFS > 0
  static int8_t q_ac[MODEL_NUM_AC_COEFFS * MODEL_Y_ROWS * MODEL_Y_COLS];
  for (int k = 0; k < MODEL_NUM_AC_COEFFS; k++)
    for (int y = 0; y < MODEL_Y_ROWS; y++)
      for (int x = 0; x < MODEL_Y_COLS; x++)
        q_ac[(k * MODEL_Y_ROWS + y) * MODEL_Y_COLS + x] = model_quantize_ac(ac_planes[k][y][x], k);
  static int8_t ac_pooled[MODEL_NUM_AC_COEFFS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];
  avg_pool_int8(q_ac, MODEL_NUM_AC_COEFFS, MODEL_Y_ROWS, MODEL_Y_COLS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS, ac_pooled);
  memcpy(concat_buf + concat_offset * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS, ac_pooled, sizeof(ac_pooled));
  concat_offset += MODEL_NUM_AC_COEFFS;
#else
  (void)ac_planes;
#endif
  t1 = esp_timer_get_time(); st.ac_quant_pool = t1 - t0; t0 = t1;

  // Guarded exactly like the AC branch above. A --no-chroma model has no
  // chroma branch in the trained graph, so the exporter emits no
  // model_quantize_chroma() and no CHROMA_* weights at all -- referencing
  // them would not compile. The decoder still fills chroma_planes (chroma DC
  // is one multiply per MCU inside a Huffman walk that has already happened),
  // so the live call site is unchanged; this mirror simply ignores them, and
  // the reported chroma_quant_pool stage time is a true zero rather than a
  // missing field.
#if MODEL_USE_CHROMA
  static int8_t q_chroma[2 * MODEL_NUM_CHROMA_COEFFS * MODEL_C_ROWS * MODEL_C_COLS];
  for (int k = 0; k < 2 * MODEL_NUM_CHROMA_COEFFS; k++)
    for (int y = 0; y < MODEL_C_ROWS; y++)
      for (int x = 0; x < MODEL_C_COLS; x++)
        q_chroma[(k * MODEL_C_ROWS + y) * MODEL_C_COLS + x] = model_quantize_chroma(chroma_planes[k][y][x], k);
  static int8_t chroma_pooled[2 * MODEL_NUM_CHROMA_COEFFS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];
  avg_pool_int8(q_chroma, 2 * MODEL_NUM_CHROMA_COEFFS, MODEL_C_ROWS, MODEL_C_COLS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS, chroma_pooled);
  memcpy(concat_buf + concat_offset * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS, chroma_pooled, sizeof(chroma_pooled));
  concat_offset += 2 * MODEL_NUM_CHROMA_COEFFS;
#else
  (void)chroma_planes;
#endif
  t1 = esp_timer_get_time(); st.chroma_quant_pool = t1 - t0; t0 = t1;

  static int8_t post_concat_out[MODEL_POST_CONCAT_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];
  conv2d_int8(concat_buf, MODEL_POST_CONCAT_IN, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS,
              POST_CONCAT_WEIGHT, POST_CONCAT_BIAS, POST_CONCAT_MULT, POST_CONCAT_SHIFT,
              MODEL_POST_CONCAT_CHANNELS, 3, 3, 1, 1, 0, 127, post_concat_out);
  t1 = esp_timer_get_time(); st.post_concat_conv = t1 - t0; t0 = t1;

  static int8_t extra_conv_0_out[MODEL_FINAL_CHANNELS * MODEL_STRIDE2_ROWS * MODEL_STRIDE2_COLS];
  conv2d_int8(post_concat_out, MODEL_POST_CONCAT_CHANNELS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS,
              EXTRA_CONV_0_WEIGHT, EXTRA_CONV_0_BIAS, EXTRA_CONV_0_MULT, EXTRA_CONV_0_SHIFT,
              MODEL_FINAL_CHANNELS, 3, 3, 1, 1, 0, 127, extra_conv_0_out);
  t1 = esp_timer_get_time(); st.extra_conv0 = t1 - t0; t0 = t1;

  int8_t pooled[MODEL_FINAL_CHANNELS];
  avg_pool_int8(extra_conv_0_out, MODEL_FINAL_CHANNELS, MODEL_STRIDE2_ROWS, MODEL_STRIDE2_COLS, 1, 1, pooled);
  t1 = esp_timer_get_time(); st.final_pool = t1 - t0; t0 = t1;

  for (int j = 0; j < MODEL_NUM_CLASSES; j++) {
    int32_t acc = OUTPUT_BIAS[j];
    for (int i = 0; i < MODEL_FINAL_CHANNELS; i++) {
      acc += (int32_t)pooled[i] * (int32_t)OUTPUT_WEIGHT[j][i];
    }
    int64_t product = (int64_t)acc * (int64_t)OUTPUT_MULT[j];
    int32_t q = model_round_shift(product, 31 - OUTPUT_SHIFT[j]);
    logits[j] = (int8_t)model_clip(q, -128, 127);
  }
  t1 = esp_timer_get_time(); st.output = t1 - t0;
}

// NOT purely a visual setting -- treat changes here as model changes: per
// capture_time_issue.md Sec 6, too much compression is what caused the
// original DC-saturation problem in the DCT features this model classifies
// on. Unlike esp32_mlp, this firmware has no standing DC-average check, so
// a regression here would show up only as degraded classification, not as
// an obvious log line. If accuracy looks off, suspect this first.
//
// 14 and 15 were both tried on 2026-08-11 for the bandwidth win (14 cut
// frame size ~42%: 2798 -> 1623 bytes, 74.5 -> 38.6 KB/s) and both were
// reverted -- the saving wasn't worth an unmeasured risk to input quality.
#define STREAM_JPEG_QUALITY 10 // lower number means higher quality, 0-63

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-S3 Cam Stream (JPEG)</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; margin:0; padding:16px; }
    /* Shown at native 160x120 -- an earlier version tried a 2x CSS
       transform:scale() trick (iframes don't stretch embedded content to
       fill the frame box the way <img>/<canvas> do, so scaling the
       element itself was the only lever available), but border-radius +
       overflow:hidden + a transformed child is a known troublesome
       combo in WebKit/Safari and it cropped the right edge of the image
       on iPhone. Dropped the scaling rather than fight that -- reliable
       and correctly-framed beats bigger. See the script below for why
       this is an <iframe> (pointed straight at /stream) instead of
       <canvas>+fetch(). */
    // #streamBox { display:inline-block; border:2px solid #444; border-radius:8px; overflow:hidden; }
    // #streamFrame { display:block; width:160px; height:120px; border:none; image-rendering:pixelated; }
    #streamBox {
      display:inline-block;
      border:2px solid #444;
      border-radius:8px;
      overflow:visible;
      width:320px;
      height:240px;
    }
    #streamFrame {
      display:block;
      width:160px;
      height:120px;
      border:none;
      transform:scale(2);
      transform-origin:top left;
      image-rendering:pixelated;
    }
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
    #stages { text-align:left; background:#1a1a1a; border-radius:8px; padding:12px 16px; min-width:220px; }
    #stages h3 { margin:0 0 8px; font-size:0.95em; color:#f97; font-weight:normal; }
    .stage-row { display:flex; align-items:center; gap:8px; margin:4px 0; font-size:0.85em; }
    .stage-name { width:120px; flex-shrink:0; color:#ccc; }
    .stage-bar-track { flex:1; background:#333; border-radius:3px; height:10px; overflow:hidden; min-width:40px; }
    .stage-bar-fill { height:100%; background:#f97; }
    .stage-ms { width:58px; flex-shrink:0; text-align:right; color:#aaa; }
  </style>
</head>
<body>
  <h2>ESP32-S3 Camera Stream (JPEG, 160x120)</h2>
  <div id="fps">FPS: --</div>
  <div id="scores"></div>
  <div id="timing"></div>
  <div class="stream-row">
    <div id="streamBox"><iframe id="streamFrame" src=""></iframe></div>
    <div id="stages">
      <h3>model_forward() stages</h3>
      <div id="stageRows"></div>
    </div>
  </div>
  <script>
    // 2026-08-14 rewrite -- this used to fetch() the multipart stream
    // manually (ReadableStream reader, split on the boundary, JPEG parts
    // to <canvas>, JSON parts to the UI -- one connection for both, no
    // separate /status poll). That broke on iPhone Safari: the page and
    // canvas loaded fine but no frame or error ever appeared. Root-caused
    // by directly testing on-device rather than guessing further:
    //   - Navigating Safari straight to http://<ip>:81/stream (bypassing
    //     fetch() entirely) streamed live, no problem -- so the firmware,
    //     the network path, and Safari's OWN multipart/x-mixed-replace
    //     support (for direct navigation/iframes, just never for <img>)
    //     were all fine.
    //   - The fetch()-based path specifically failed with
    //     "TypeError: Load failed" -- Safari's generic fetch() failure,
    //     most consistent with a cross-origin (port 80 page -> port 81
    //     stream) fetch()/CORS restriction, not a data or protocol issue.
    // Fix: stop using fetch() for the video entirely -- <iframe src=
    // ".../stream"> instead, since iframe navigation isn't subject to
    // the same restriction fetch() hit, and this is the exact mechanism
    // just confirmed working live on-device. Status/scores go back to a
    // separate poll (like before the fetch()-based interleaving
    // optimization), but same-origin this time (/status is on port 80,
    // same as this page, so it's not cross-origin and doesn't share
    // fetch()'s failure mode) -- see pollStatus() below. This does
    // reopen the two-connection-contention question the interleaving
    // optimization existed to close (see streaming_stall_fix_port.md
    // Sec 5) -- accepted as the tradeoff for working on iPhone at all.

    // Claim the stream slot before connecting. The device serves ONE viewer,
    // and an abandoned-but-open socket (Firefox connection pooling) can still
    // own it -- /claim on port 80 tells the incumbent to stand down, because
    // port 81 is inside the handler and cannot hear anything. Fire-and-forget:
    // if the board is still booting this fails harmlessly and the watchdog
    // retries.
    async function connectStream() {
      try { await fetch('/claim', {cache: 'no-store'}); } catch (e) { /* booting */ }
      document.getElementById('streamFrame').src =
        'http://' + window.location.hostname + ':81/stream?r=' + Date.now();
    }
    connectStream();

    const stageLabels = {
      quant_dc: 'quant DC',
      lum_conv: 'lum_conv',
      stride2_conv: 'stride2_conv',
      ac_quant_pool: 'AC quant+pool',
      chroma_quant_pool: 'chroma quant+pool',
      post_concat_conv: 'post_concat_conv',
      extra_conv0: 'extra_conv0',
      final_pool: 'final_pool',
      output: 'output (dense)'
    };

    function updateStatus(j) {
      document.getElementById('fps').textContent = 'FPS: ' + j.fps.toFixed(1);

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
        'capture ' + j.t_capture_ms.toFixed(1) + 'ms | jpeg_parse ' + j.t_parse_ms.toFixed(1) +
        'ms | inference ' + j.t_infer_ms.toFixed(1) + 'ms | total ' + j.t_total_ms.toFixed(1) + 'ms' +
        // classify_fps is how fast frames are captured+classified (uncapped);
        // the big FPS number above is how many are actually sent (20fps cap
        // in stream_handler()) -- see g_sent_fps in main.cpp for why these
        // two numbers are expected to differ.
        ' | classify ' + j.classify_fps.toFixed(1) + 'fps';

      const stages = j.stages || [];
      const maxMs = Math.max(0.01, ...stages.map(s => s.ms));
      const stageRows = document.getElementById('stageRows');
      stageRows.innerHTML = '';
      for (const s of stages) {
        const row = document.createElement('div');
        row.className = 'stage-row';
        const pct = (s.ms / maxMs * 100).toFixed(0);
        row.innerHTML =
          '<span class="stage-name">' + (stageLabels[s.name] || s.name) + '</span>' +
          '<span class="stage-bar-track"><span class="stage-bar-fill" style="width:' + pct + '%"></span></span>' +
          '<span class="stage-ms">' + s.ms.toFixed(2) + 'ms</span>';
        stageRows.appendChild(row);
      }
    }

    // Same-origin (port 80, this page's own origin) poll of /status --
    // see the big comment above for why this is separate from the video
    // now instead of interleaved into /stream. Errors here get the same
    // "show the real browser error" treatment as the old fetch()-based
    // video path did, for the same on-device-diagnosability reason.
    // Stream watchdog. The firmware now drops a client whose socket has been
    // blocked for a few seconds rather than letting it wedge the server's
    // single worker task for ~16s (see start_camera_server()'s
    // send_wait_timeout note). That trade only works if a dropped stream
    // comes back by itself -- an <iframe> whose multipart response ended
    // just sits there showing the last frame forever, with no event we can
    // hook to notice. So use the status poll we're already doing as the
    // liveness signal: last_frame_age_ms is measured on the device and
    // keeps climbing whenever capture has stopped, which (unlike fps) can't
    // be confused with "running slowly". Reload the iframe when it stalls.
    let reloads = 0;
    async function reloadStream(why) {
      reloads++;
      const f = document.getElementById('streamFrame');
      // Drop the old src first so the previous connection is actually torn
      // down -- it still occupies the device's single stream slot -- then
      // claim the slot so an incumbent that ISN'T us (a stale socket from a
      // closed tab, another viewer) also lets go. Without the claim the
      // reload just reconnects into a slot somebody else still owns and the
      // watchdog fires again a few seconds later, forever.
      f.src = '';
      try { await fetch('/claim', {cache: 'no-store'}); } catch (e) { /* stalled */ }
      // Cache-bust so the browser reconnects rather than reusing the dead
      // response.
      const base = 'http://' + window.location.hostname + ':81/stream';
      f.src = base + '?r=' + Date.now();
      console.log('stream watchdog: reloading (' + why + '), reload #' + reloads);
    }

    let noFramePolls = 0;
    async function pollStatus() {
      while (true) {
        try {
          const resp = await fetch('/status', {cache: 'no-store'});
          if (!resp.ok) throw new Error('status fetch failed: ' + resp.status);
          const j = await resp.json();
          updateStatus(j);
          // No frames for this long means the stream is not coming back on
          // its own. Must stay comfortably above the device's send timeout
          // (stream_config.send_wait_timeout, now 5s) or a single slow patch
          // triggers a reload loop -- the reload is far more expensive than
          // the stall it is trying to fix. Raised 4s -> 8s alongside that
          // change on 2026-08-19; the old comment here still claimed 3s long
          // after the timeout had become 1s, which is how the margin was lost.
          let needReload = '';
          // -1 means the device has captured NO frame since boot. That is
          // not a large age, it is the ABSENCE of one, so an `age > 8000`
          // test can never catch it -- which is exactly how a reset used to
          // leave this page stuck on a dead iframe until someone refreshed
          // by hand. Require a run of them rather than acting on the first:
          // a fresh boot, or a page that has only just pointed its iframe at
          // :81, legitimately reads -1 for a poll or two, and reloading then
          // would fight the connection it is trying to establish.
          if (j.last_frame_age_ms === -1) {
            if (++noFramePolls >= 3) needReload = 'no frame since device boot';
          } else {
            noFramePolls = 0;
          }
          if (typeof j.last_frame_age_ms === 'number' && j.last_frame_age_ms > 8000) {
            needReload = 'no frame for ' + j.last_frame_age_ms + 'ms';
          }
          if (needReload) {
            noFramePolls = 0;
            await reloadStream(needReload);
            await new Promise(r => setTimeout(r, 3000)); // let it re-establish
          }
        } catch (e) {
          document.getElementById('fps').textContent =
            'FPS: (status fetch failed: ' + (e && e.name ? e.name + ': ' : '') + (e && e.message ? e.message : String(e)) + ' -- retrying...)';
        }
        await new Promise(r => setTimeout(r, 1000));
      }
    }
    pollStatus();
  </script>
</body>
</html>
)rawliteral";

// GET /claim -- "I am about to open the stream; whoever has it should let go."
// Deliberately on port 80, whose handlers all return promptly, so it stays
// answerable while port 81 is stuck inside stream_handler(). See
// g_stream_claim above for why that is the only channel available.
static esp_err_t claim_handler(httpd_req_t *req) {
  g_stream_claim++;
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  char buf[64];
  int n = snprintf(buf, sizeof(buf), "{\"claim\":%u}", (unsigned)g_stream_claim);
  return httpd_resp_send(req, buf, n);
}

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// Current WiFi power-save mode as a short string for /status. Read live
// (not cached from boot) so a mode change after association is visible --
// see connect_wifi() for why this is worth reporting at all.
static const char *wifi_ps_name() {
  wifi_ps_type_t ps;
  if (esp_wifi_get_ps(&ps) != ESP_OK) return "unknown";
  return ps == WIFI_PS_NONE ? "none" : (ps == WIFI_PS_MIN_MODEM ? "min_modem" : "max_modem");
}

// Why the last boot happened. Added 2026-08-15 to triage "it runs fine then
// suddenly stops / crashes and the flash LED comes on" WITHOUT a serial
// monitor -- reading /dev/ttyACM0 on this host resets the board into USB
// download mode, so the panic message that would normally identify this is
// unavailable by construction. Reported over the network instead.
//   brownout  -> supply can't hold up under WiFi TX + camera current spikes
//                (the classic ESP32-CAM failure; suspect cable/port/PSU first)
//   panic     -> firmware crash (guru meditation)
//   task_wdt / int_wdt -> a task hogged the CPU without yielding
//   ext       -> reset asserted on the EN pin, i.e. something outside the
//                firmware reset it (auto-reset circuit -- e.g. a host process
//                opening the serial port and toggling DTR/RTS)
//   poweron   -> clean power-up, nothing abnormal
static const char *reset_reason_name() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:   return "poweron";
    case ESP_RST_EXT:       return "ext";
    case ESP_RST_SW:        return "sw";
    case ESP_RST_PANIC:     return "panic";
    case ESP_RST_INT_WDT:   return "int_wdt";
    case ESP_RST_TASK_WDT:  return "task_wdt";
    case ESP_RST_WDT:       return "wdt";
    case ESP_RST_DEEPSLEEP: return "deepsleep";
    case ESP_RST_BROWNOUT:  return "brownout";
    case ESP_RST_SDIO:      return "sdio";
    default:                return "unknown";
  }
}

// Builds the status JSON into caller-provided storage, returning its length.
// Split out of status_handler() so the /stream handler can push the exact
// same payload as an interleaved multipart part (see stream_handler()) --
// one builder keeps the two paths from drifting apart. /status itself is
// kept because curl and other tooling shouldn't have to parse multipart.
// Note: sizeof(buf) would be a pointer size in here -- use bufsize.
static int build_status_json(char *buf, size_t bufsize) {
  // How long since the last captured frame, and an fps that tells the truth
  // when capture has stopped. Without this, a wedged stream reports its last
  // healthy fps indefinitely -- the single most misleading thing in this
  // endpoint, because it makes a frozen stream look identical to a running
  // one. Anything past ~2s of silence is reported as 0 fps.
  //
  // "No frame yet" (g_last_frame_us == 0, i.e. nothing has been captured
  // since boot) is reported as -1, NOT 0. It used to report 0, which is
  // indistinguishable from "a frame arrived just now" -- so after a reset
  // the page's `age > 8000` watchdog could never fire, the iframe sat on a
  // dead connection, and the stream only came back on a manual refresh.
  // The page special-cases -1; see pollStatus(). Reported with %d for the
  // same reason -- as %u it would print 4294967295.
  int64_t frame_age_us = (g_last_frame_us == 0) ? 0 : (esp_timer_get_time() - g_last_frame_us);
  int frame_age_ms = (g_last_frame_us == 0) ? -1 : (int)(frame_age_us / 1000);
  float classify_fps_reported = (g_last_frame_us != 0 && frame_age_us > 2000000) ? 0.0f : g_fps;

  // Same staleness treatment for the SENT rate, based on g_last_sent_us
  // rather than g_last_frame_us -- this is "fps" below, i.e. what a client
  // is actually receiving. Deliberately not the same number as
  // classify_fps_reported: capture/inference run on every frame regardless
  // of the 20fps send cap in stream_handler(), so classify_fps_reported can
  // legitimately sit well above 20 while this stays capped. See
  // g_sent_fps's declaration.
  int64_t sent_age_us = (g_last_sent_us == 0) ? 0 : (esp_timer_get_time() - g_last_sent_us);
  float fps_reported = (g_last_sent_us != 0 && sent_age_us > 2000000) ? 0.0f : g_sent_fps;
  // Exposed alongside last_frame_age_ms because the two can diverge, and the
  // gap between them IS the diagnosis: last_frame_age_ms is capture age, so
  // it stays low whenever the loop is running, even if nothing is reaching
  // the client. A client that ACKs but never renders (throttled tab, wedged
  // iframe decode) leaves capture age healthy and send age healthy while the
  // picture is frozen -- but a send path that has actually stopped shows up
  // here and nowhere else. The page's watchdog currently only checks capture
  // age; see fix_stall.md section 3.2.
  //
  // -1 for "nothing sent yet", same convention and same reason as
  // frame_age_ms above.
  int sent_age_ms = (g_last_sent_us == 0) ? -1 : (int)(sent_age_us / 1000);

  int off = snprintf(buf, bufsize,
      "{\"fps\":%.1f,\"classify_fps\":%.1f,\"t_capture_ms\":%.1f,\"t_parse_ms\":%.1f,\"t_infer_ms\":%.1f,\"t_total_ms\":%.1f,"
      "\"bench_parse_us\":%.1f,\"rssi\":%d,\"free_heap\":%u,\"wifi_ps\":\"%s\","
      "\"uptime_s\":%u,\"reset_reason\":\"%s\",\"min_free_heap\":%u,"
      "\"last_frame_age_ms\":%d,\"last_sent_age_ms\":%d,"
      "\"stages\":["
        "{\"name\":\"quant_dc\",\"ms\":%.2f},"
        "{\"name\":\"lum_conv\",\"ms\":%.2f},"
        "{\"name\":\"stride2_conv\",\"ms\":%.2f},"
        "{\"name\":\"ac_quant_pool\",\"ms\":%.2f},"
        "{\"name\":\"chroma_quant_pool\",\"ms\":%.2f},"
        "{\"name\":\"post_concat_conv\",\"ms\":%.2f},"
        "{\"name\":\"extra_conv0\",\"ms\":%.2f},"
        "{\"name\":\"final_pool\",\"ms\":%.2f},"
        "{\"name\":\"output\",\"ms\":%.2f}"
      "],"
      "\"model_forward_self_test\":{"
        "\"bit_exact\":%d,\"class_match\":%d,\"num_vectors\":%d,\"max_abs_logit_diff\":%d"
      "},"
      "\"decoder_self_test\":{\"pass\":%d,\"total\":%d},"
      "\"stage_timing_self_test\":{\"pass\":%d,\"total\":%d},"
      "\"scores\":[",
      fps_reported, classify_fps_reported, g_t_capture_ms, g_t_parse_ms, g_t_infer_ms, g_t_total_ms, g_bench_parse_us,
      // RSSI/free-heap as standing diagnostics: a weak link (RSSI worse
      // than about -75 dBm) or a shrinking heap would each produce
      // stall-like streaming symptoms that look identical from the
      // browser, so keep them visible rather than re-deriving them by
      // hand next time.
      (int)WiFi.RSSI(), (unsigned)ESP.getFreeHeap(), wifi_ps_name(),
      // uptime_s resets to ~0 on every reboot, so polling /status makes a
      // reset visible even when it happens between polls (fps/timing values
      // alone can't distinguish "stalled" from "rebooted"). min_free_heap is
      // the low-water mark since boot -- catches a transient allocation
      // squeeze that ESP.getFreeHeap() would have already recovered from.
      (unsigned)(esp_timer_get_time() / 1000000), reset_reason_name(),
      (unsigned)ESP.getMinFreeHeap(), frame_age_ms, sent_age_ms,
      g_t_stage_quant_dc_ms, g_t_stage_lum_conv_ms, g_t_stage_stride2_conv_ms, g_t_stage_ac_quant_pool_ms,
      g_t_stage_chroma_quant_pool_ms,
      g_t_stage_post_concat_conv_ms, g_t_stage_extra_conv0_ms, g_t_stage_final_pool_ms, g_t_stage_output_ms,
      g_model_bit_exact_count, g_model_class_match_count, N_TEST_VECTORS, g_model_max_logit_diff,
      g_decoder_self_test_pass, g_decoder_self_test_total,
      g_stage_timing_self_test_pass, g_stage_timing_self_test_total);

  if (g_has_classified) {
    int8_t logits[MODEL_NUM_CLASSES];
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) logits[i] = g_last_logits[i];

    // Softmax over the raw int8 logits -> a real probability distribution
    // (sums to 100%), computed here (once per /status request) rather than
    // in the per-frame hot path, since it's just display formatting, not
    // part of the classification itself.
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
          int t = order[i]; order[i] = order[j]; order[j] = t;
        }

    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
      float pct = 100.0f * exps[order[i]] / sum;
      off += snprintf(buf + off, bufsize - off, "%s{\"class\":\"%s\",\"pct\":%.1f}",
                       i == 0 ? "" : ",", MODEL_CLASS_NAMES[order[i]], pct);
    }
  }
  off += snprintf(buf + off, bufsize - off, "]}");
  return off;
}

static esp_err_t status_handler(httpd_req_t *req) {
  char buf[1536];
  int off = build_status_json(buf, sizeof(buf));
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, buf, off);
}

// Dumps the send ring buffer as CSV. Registered on camera_httpd (port 80),
// NOT on stream_httpd -- deliberately. The stream server has a single worker
// task and stream_handler() never returns while a client is connected, so an
// endpoint there would be unreachable during exactly the situation it exists
// to diagnose. This is the same reason /status stayed answerable through the
// socket-exhaustion wedge that made the device look alive.
//
// dt_ms (gap since the previous row) is usually the most informative column:
// it is the actual send period, so the send cap, the classification period
// and any stall all show up in it directly.

// GET /control?var=<name>&val=<int> -- set one OV2640 sensor control at runtime.
//
// Added 2026-08-20 to make sensor tuning measurable instead of guessable. The
// open question it serves: the camera's DCT coefficients carry far less energy
// than the training data's, especially in the AC planes (see
// results/CAMERA_DOMAIN_SHIFT.md). Matching the *data* to the camera was tried
// and did not help. The other direction -- moving the camera toward the data --
// has never been tried, and the three most relevant controls (contrast,
// sharpness, denoise) are not set anywhere in this firmware, so they sit at
// driver defaults.
//
// Sweeping them by reflashing is minutes per data point; over HTTP it is
// seconds, and python_code/sweep_sensor_settings.py drives it: set a value,
// capture frames from /stream, compute the zig-zag energy/sparsity profile,
// compare against the training set.
//
// Deliberately does NOT persist anything. A reboot returns to whatever
// init_camera() sets, so an experiment cannot silently become the deployed
// configuration -- if a setting turns out to help, it gets written into
// init_camera() on purpose.
static esp_err_t control_handler(httpd_req_t *req) {
  char query[128];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "expected ?var=<name>&val=<int>");
    return ESP_FAIL;
  }
  char var[32] = {0}, val_s[16] = {0};
  if (httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(query, "val", val_s, sizeof(val_s)) != ESP_OK) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "expected ?var=<name>&val=<int>");
    return ESP_FAIL;
  }
  int val = atoi(val_s);

  sensor_t *s = esp_camera_sensor_get();
  if (!s) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no sensor");
    return ESP_FAIL;
  }

  int res = -1;
  if      (!strcmp(var, "contrast"))    res = s->set_contrast(s, val);
  else if (!strcmp(var, "brightness"))  res = s->set_brightness(s, val);
  else if (!strcmp(var, "saturation"))  res = s->set_saturation(s, val);
  else if (!strcmp(var, "sharpness"))   res = s->set_sharpness(s, val);
  else if (!strcmp(var, "denoise"))     res = s->set_denoise(s, val);
  else if (!strcmp(var, "quality"))     res = s->set_quality(s, val);
  else if (!strcmp(var, "ae_level"))    res = s->set_ae_level(s, val);
  else if (!strcmp(var, "aec2"))        res = s->set_aec2(s, val);
  else if (!strcmp(var, "agc_gain"))    res = s->set_agc_gain(s, val);
  else if (!strcmp(var, "gainceiling")) res = s->set_gainceiling(s, (gainceiling_t)val);
  else if (!strcmp(var, "raw_gma"))     res = s->set_raw_gma(s, val);
  else if (!strcmp(var, "lenc"))        res = s->set_lenc(s, val);
  else if (!strcmp(var, "bpc"))         res = s->set_bpc(s, val);
  else if (!strcmp(var, "wpc"))         res = s->set_wpc(s, val);
  else if (!strcmp(var, "dcw"))         res = s->set_dcw(s, val);
  else {
    httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "unknown var");
    return ESP_FAIL;
  }

  // A nonzero return means the driver rejected it -- several OV2640 controls
  // exist in the sensor_t struct but are not implemented for this part, and
  // report failure rather than silently doing nothing. Worth surfacing: a
  // control that cannot be set is a different result from one that does not
  // help.
  char out[96];
  int n = snprintf(out, sizeof(out), "{\"var\":\"%s\",\"val\":%d,\"result\":%d}\n", var, val, res);
  httpd_resp_set_type(req, "application/json");
  Serial.printf("control: %s = %d (driver returned %d)\n", var, val, res);
  return httpd_resp_send(req, out, n);
}

static esp_err_t sendlog_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/plain");
  char line[128];

  // Snapshot the count once: the stream task keeps writing while we format,
  // and a moving target would make start/total inconsistent with each other.
  uint32_t total = g_sendlog_next;
  uint32_t start = (total > SENDLOG_SLOTS) ? (total - SENDLOG_SLOTS) : 0;

  int n = snprintf(line, sizeof(line),
                   "# now_us=%lld conns=%u rows=%u slots=%d\n",
                   (long long)esp_timer_get_time(), (unsigned)g_stream_conns,
                   (unsigned)total, SENDLOG_SLOTS);
  esp_err_t res = httpd_resp_send_chunk(req, line, n);
  if (res == ESP_OK) {
    const char *hdr = "row,conn,t_us,dt_ms,send_ms,bytes,res,rssi\n";
    res = httpd_resp_send_chunk(req, hdr, strlen(hdr));
  }

  int64_t prev_us = 0;
  for (uint32_t i = start; i < total && res == ESP_OK; i++) {
    const send_rec *r = &g_sendlog[i % SENDLOG_SLOTS];
    float dt_ms = (prev_us == 0) ? 0.0f : (float)(r->t_us - prev_us) / 1000.0f;
    prev_us = r->t_us;
    n = snprintf(line, sizeof(line), "%u,%u,%lld,%.1f,%.2f,%u,%d,%d\n",
                 (unsigned)i, (unsigned)r->conn, (long long)r->t_us, dt_ms,
                 (float)r->send_us / 1000.0f, (unsigned)r->bytes,
                 (int)r->res, (int)r->rssi);
    res = httpd_resp_send_chunk(req, line, n);
  }

  if (res == ESP_OK) res = httpd_resp_send_chunk(req, NULL, 0);
  return res;
}

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
// Format strings (not const char* variables) so they can be concatenated
// with STREAM_BOUNDARY at compile time into a single snprintf, letting one
// write carry boundary+header -- see the send sites below.
#define STREAM_PART_FMT      "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"
#define STREAM_JSON_PART_FMT "Content-Type: application/json\r\nContent-Length: %u\r\n\r\n"

// Captures one frame and runs DCT-decode + inference on it, updating every
// g_t_*/g_last_logits/g_has_classified/g_fps global exactly the same way
// regardless of caller -- factored out of stream_handler()'s loop body so
// frame_handler() (one-shot, see below) can't drift out of sync with it.
// *out_fb is set to the captured frame (caller must esp_camera_fb_return()
// it) whenever this returns true; false only means capture itself failed
// (a DCT-decode failure still returns true with *out_fb set -- there's
// still a JPEG worth serving, just no fresh classification this frame,
// matching the original stream_handler()'s behavior).
// Call once for every frame actually written to a client's socket -- see
// g_sent_fps's declaration for why this is tracked separately from
// capture_and_classify_one_frame()'s g_fps.
static void mark_frame_sent() {
  int64_t now = esp_timer_get_time();
  g_last_sent_us = now;
  static int64_t sent_fps_last_report = 0;
  static uint32_t sent_frames_since_report = 0;
  if (sent_fps_last_report == 0) sent_fps_last_report = now;
  sent_frames_since_report++;
  if (now - sent_fps_last_report >= 1000000) {
    g_sent_fps = sent_frames_since_report * 1000000.0f / (float)(now - sent_fps_last_report);
    sent_frames_since_report = 0;
    sent_fps_last_report = now;
  }
}

static bool capture_and_classify_one_frame(camera_fb_t **out_fb, bool classify = true) {
  int64_t t_capture_start = esp_timer_get_time();
  camera_fb_t *fb = esp_camera_fb_get();
  int64_t t_capture_end = esp_timer_get_time();
  *out_fb = fb;
  if (!fb) return false;

  int64_t now = esp_timer_get_time();
  static int64_t last_classify_log = 0;

  // classify=false (reachable via /stream?noai=1) captures and sends frames
  // without running the decoder+CNN. Purely diagnostic: it isolates "is the
  // classification what causes the stalls?" on THIS board, at this
  // resolution, quality and link. The sibling esp32_fps project appears to
  // answer that question already, but it changes three things at once
  // (96x96 instead of 160x120, JPEG quality 32 instead of 12, and no
  // inference), so it can't attribute the difference to any one of them.
  // Normal streaming is unaffected -- classification stays on by default.
  if (!classify) {
    g_t_capture_ms = (t_capture_end - t_capture_start) / 1000.0f;
    g_t_parse_ms = 0.0f;
    g_t_infer_ms = 0.0f;
    g_t_total_ms = g_t_capture_ms;
  } else {

  // Classification runs on every captured frame, alongside streaming it
  // to the browser (not instead of), per
  // esp32_cnn_implementation_prompt.md step 6.
  int32_t dc_plane[DCT_Y_ROWS][DCT_Y_COLS];
  int32_t ac_planes[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS];
  int32_t chroma_planes[2][DCT_C_ROWS][DCT_C_COLS];
  const char *decode_err = nullptr;
  bool decode_ok = dct_extract_coeffs(fb->buf, fb->len, dc_plane, ac_planes, chroma_planes, &decode_err);
  int64_t t_parse_end = esp_timer_get_time();

  if (decode_ok) {
    int8_t logits[MODEL_NUM_CLASSES];
    StageTimingsUs stage_us;
    model_forward_timed(dc_plane, ac_planes, chroma_planes, logits, stage_us);
    int64_t t_infer_end = esp_timer_get_time();

    for (int i = 0; i < MODEL_NUM_CLASSES; i++) g_last_logits[i] = logits[i];
    g_has_classified = true;
    g_t_capture_ms = (t_capture_end - t_capture_start) / 1000.0f;
    g_t_parse_ms = (t_parse_end - t_capture_end) / 1000.0f;
    g_t_infer_ms = (t_infer_end - t_parse_end) / 1000.0f;
    g_t_total_ms = (t_infer_end - t_capture_start) / 1000.0f;
    g_t_stage_quant_dc_ms = stage_us.quant_dc / 1000.0f;
    g_t_stage_lum_conv_ms = stage_us.lum_conv / 1000.0f;
    g_t_stage_stride2_conv_ms = stage_us.stride2_conv / 1000.0f;
    g_t_stage_ac_quant_pool_ms = stage_us.ac_quant_pool / 1000.0f;
    g_t_stage_chroma_quant_pool_ms = stage_us.chroma_quant_pool / 1000.0f;
    g_t_stage_post_concat_conv_ms = stage_us.post_concat_conv / 1000.0f;
    g_t_stage_extra_conv0_ms = stage_us.extra_conv0 / 1000.0f;
    g_t_stage_final_pool_ms = stage_us.final_pool / 1000.0f;
    g_t_stage_output_ms = stage_us.output / 1000.0f;

    if (now - last_classify_log >= 1000000) {  // throttle to 1/sec
      int pred = model_argmax(logits, MODEL_NUM_CLASSES);
      Serial.printf("class=%s | T_capture=%.1fms T_jpeg_parse=%.1fms T_inference=%.1fms T_total=%.1fms\n",
                    MODEL_CLASS_NAMES[pred], g_t_capture_ms, g_t_parse_ms, g_t_infer_ms, g_t_total_ms);
      last_classify_log = now;
    }
  } else if (now - last_classify_log >= 1000000) {
    Serial.printf("dct_extract_coeffs failed: %s\n", decode_err);
    last_classify_log = now;
  }

  }  // end if (classify)

  // fps bookkeeping -- shared across both /stream's persistent loop and
  // /frame's one-shot requests, so g_fps reports how many frames/sec were
  // actually produced system-wide over the last full second, regardless of
  // which endpoint(s) are driving capture.
  g_last_frame_us = now;
  static int64_t fps_last_report = 0;
  static uint32_t fps_frames_since_report = 0;
  if (fps_last_report == 0) fps_last_report = now;
  fps_frames_since_report++;
  if (now - fps_last_report >= 1000000) {
    g_fps = fps_frames_since_report * 1000000.0f / (float)(now - fps_last_report);
    fps_frames_since_report = 0;
    fps_last_report = now;
  }

  return true;
}

// Handles ONE HTTP request = ONE captured/classified frame = exactly two
// multipart parts (image/jpeg, then application/json), then ends the
// response normally. Originally added so the page could poll this via
// fetch() instead of holding /stream's persistent connection open, as a
// fix for iPhone Safari sitting forever on its first `reader.read()`
// against /stream's never-ending, no-Content-Length response. That fetch()
// -based approach turned out to have its own Safari-specific failure
// (fetch() to this endpoint failed outright with "TypeError: Load failed",
// most likely CORS-related for the cross-origin port 80->81 request --
// see index_handler()'s big comment for the full trail) -- the page now
// uses an <iframe src="/stream"> instead, which sidesteps fetch()
// entirely. This endpoint is kept as a general-purpose one-shot
// image+status utility (e.g. for scripting/tooling), just no longer
// depended on by the page itself.
static esp_err_t frame_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  char part_buf[192];    // STREAM_BOUNDARY (36 bytes) + image part header; see stream_handler()'s original sizing note
  char status_buf[1536]; // same sizing as status_handler()/stream_handler()
  char status_hdr[192];

  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  if (!capture_and_classify_one_frame(&fb)) return ESP_FAIL;

  size_t hlen = snprintf(part_buf, sizeof(part_buf), "%s" STREAM_PART_FMT, STREAM_BOUNDARY, (unsigned)fb->len);
  if (hlen >= sizeof(part_buf)) hlen = sizeof(part_buf) - 1;
  res = httpd_resp_send_chunk(req, part_buf, hlen);
  if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  if (res == ESP_OK) mark_frame_sent();

  if (res == ESP_OK) {
    // Every /frame response carries a fresh status part -- unlike
    // /stream's 1/sec throttle (there, status shares an always-open
    // connection with many frames/sec so throttling made sense; here
    // each request already IS one frame, so there's nothing to throttle
    // against and the client benefits from up-to-date scores/timings on
    // every poll).
    int status_len = build_status_json(status_buf, sizeof(status_buf));
    size_t status_hlen = snprintf(status_hdr, sizeof(status_hdr), "%s" STREAM_JSON_PART_FMT,
                                  STREAM_BOUNDARY, (unsigned)status_len);
    if (status_hlen >= sizeof(status_hdr)) status_hlen = sizeof(status_hdr) - 1;
    res = httpd_resp_send_chunk(req, status_hdr, status_hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, status_buf, status_len);
  }
  if (res == ESP_OK) res = httpd_resp_send_chunk(req, NULL, 0);  // end this (bounded) response
  return res;
}

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  // Must hold STREAM_BOUNDARY (36 bytes) + the image part header (up to
  // 86 bytes). 192 leaves real headroom -- the snprintf clamp below would
  // otherwise silently truncate the boundary and corrupt the stream
  // framing rather than fail loudly.
  char part_buf[192];

  // Connection accounting for /sendlog. A stall that is really a
  // drop-and-reconnect cycle looks like a stall from the browser, but shows
  // up here as a rising connection count -- which is the single fastest way
  // to tell "one stream went quiet" from "the stream keeps being torn down
  // and re-established". See fix_stall.md sections 3.1 and 3.2.
  uint32_t conn = ++g_stream_conns;
  g_sendlog_conn = (uint16_t)conn;
  uint32_t frames_sent = 0;
  int64_t conn_start_us = esp_timer_get_time();

  // Snapshot the takeover counter at entry; the loop below stands down if it
  // changes. See g_stream_claim for why a stale socket can otherwise own this
  // handler indefinitely.
  const uint32_t my_claim = g_stream_claim;

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  // Disable Nagle's algorithm on this socket. Symptom without it (measured
  // on the sibling esp32_mlp firmware, same streaming code): frames
  // delivered in fast bursts (median 4ms apart) separated by long stalls
  // (p99 2.3s, max 3.9s), while the device's own per-frame timing stayed
  // healthy the whole time. That burst-then-stall shape with a tiny median
  // is the classic Nagle + delayed-ACK interaction: Nagle holds a small
  // write waiting for the ACK of the previous one, the peer's delayed-ACK
  // timer sits on that ACK for a few hundred ms, and everything queued
  // behind it releases at once. A frame here is ~2.3KB of JPEG plus a small
  // boundary+header write, which hits that pattern squarely.
  int stream_fd = httpd_req_to_sockfd(req);
  if (stream_fd >= 0) {
    int nodelay = 1;
    if (setsockopt(stream_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) != 0) {
      Serial.println("warning: failed to set TCP_NODELAY on stream socket");
    }

    // TCP keepalive, so a peer that vanished WITHOUT closing cleanly (WiFi
    // drop, device sleep, tab killed) is detected instead of leaving this
    // handler writing into a socket nobody is reading. That matters more
    // here than usual: this loop holds the server's only worker task, so a
    // handler that never notices its dead peer blocks every other client
    // too. Sends into a dead socket succeed while the send buffer has room,
    // so without this the loop can spin happily into the void.
    // 5s idle + 3 probes 2s apart => dead peer detected in ~11s, after
    // which the next send fails and the loop breaks and frees the task.
    int ka = 1, idle = 5, intvl = 2, cnt = 3;
    setsockopt(stream_fd, SOL_SOCKET,  SO_KEEPALIVE,  &ka,    sizeof(ka));
    setsockopt(stream_fd, IPPROTO_TCP, TCP_KEEPIDLE,  &idle,  sizeof(idle));
    setsockopt(stream_fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
    setsockopt(stream_fd, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,   sizeof(cnt));
  }

  // /stream?noai=1 -> stream without classifying (diagnostic A/B; see
  // capture_and_classify_one_frame()). Anything else classifies as usual.
  bool classify = true;
  {
    char q[32];
    if (httpd_req_get_url_query_str(req, q, sizeof(q)) == ESP_OK && strstr(q, "noai=1")) {
      classify = false;
      Serial.println("stream: classification DISABLED for this connection (noai=1)");
    }
  }

  // Send-rate cap: skip SENDING (not classifying) a frame if less than this
  // long has passed since the last one was sent. Every frame is still
  // captured and run through the CNN either way -- g_last_frame_us/g_fps
  // update on every capture in capture_and_classify_one_frame(), regardless
  // of whether this loop decides to put it on the wire.
  //
  // Why: classification currently paces the sender by accident (~21ms/frame
  // inference leaves the loop unable to outrun the link) -- see
  // inference_paces_the_stream memory. That pacing disappears the moment
  // inference gets faster (a smaller model, more ESP-NN), at which point the
  // producer can outrun the link exactly like the /stream?noai=1 test did:
  // send buffer fills, httpd_resp_send_chunk() blocks, client gets dropped
  // by the 1s send_wait_timeout. This makes the cap explicit and independent
  // of inference speed, so it isn't silently lost by a later optimization.
  // THE ALIASING RULE (measured 2026-08-19): this interval must stay BELOW the
  // minimum loop period, or the preview rate collapses to a fraction of the
  // classification rate. The test below runs *after* a frame is classified, so
  // the achievable send period can only be a whole multiple of the loop
  // period: the first admissible multiple is ceil(interval / loop_period).
  //
  // At 40000us against a ~38ms loop, ceil(40/38) = 2, so every second frame
  // was discarded and the preview sat at 13.9fps against a 26.8fps
  // classification rate -- measured directly, dt_ms pinned at 72.1ms (two loop
  // periods) with no spread. Halving the preview rate was never the intent;
  // it was an artifact of the interval landing just above the loop period.
  //
  // 20000us is below the fastest loop iteration observed (t_total min 21.4ms,
  // with capture at 0.5ms), so in practice the test always passes and no frame
  // is skipped. The cap survives only as a backstop for the case its original
  // comment below describes -- a much faster future model, where bounding the
  // offered load becomes real again rather than hypothetical. If inference
  // ever drops below ~20ms/frame, this value must drop with it or the aliasing
  // returns.
  //
  // Original rationale, still valid as the reason a cap exists at all:
  // classification currently paces the sender by accident (~21ms/frame
  // inference leaves the loop unable to outrun the link) -- see
  // inference_paces_the_stream memory. That pacing disappears the moment
  // inference gets faster (a smaller model, more ESP-NN), at which point the
  // producer can outrun the link exactly like the /stream?noai=1 test did:
  // send buffer fills, httpd_resp_send_chunk() blocks, client gets dropped
  // by the send_wait_timeout. NOTE: that failure was reproduced once, on
  // older firmware, and has never been re-tested -- see markdown/fix_stall.md
  // test D. The cap may turn out to be unnecessary entirely.
  static const int64_t MIN_SEND_INTERVAL_US = 20000;  // 50 fps -- see above
  int64_t last_send_us = 0;

  while (true) {
    // Stand down for a newer client. Checked before doing any work, so a
    // takeover costs at most one frame period. The grace window stops two
    // clients that both claim from kicking each other in a tight loop; the
    // claim is a counter rather than an edge, so a claim arriving during the
    // grace is deferred, never lost.
    if (g_stream_claim != my_claim &&
        esp_timer_get_time() - conn_start_us > STREAM_CLAIM_GRACE_US) {
      Serial.println("stream: yielding to a newer client (/claim)");
      break;
    }

    if (!capture_and_classify_one_frame(&fb, classify)) {
      res = ESP_FAIL;
      break;
    }

    int64_t frame_now = esp_timer_get_time();
    if (frame_now - last_send_us < MIN_SEND_INTERVAL_US) {
      esp_camera_fb_return(fb);
      fb = NULL;
      continue;  // classified above, just not sent -- keeps offered load capped
    }
    last_send_us = frame_now;

    // Boundary and part header go out as ONE write rather than two: fewer,
    // larger TCP segments per frame (2 writes instead of 3), which reduces
    // the small-packet churn that made the Nagle stall above so easy to
    // hit. The JPEG body stays a separate write -- it's already large and
    // copying ~2.3KB into a staging buffer just to merge it would cost
    // more than it saves.
    size_t hlen = snprintf(part_buf, sizeof(part_buf), "%s" STREAM_PART_FMT,
                           STREAM_BOUNDARY, (unsigned)fb->len);
    // snprintf returns the length it *would* have written even when
    // truncated — clamp so a future header format change can't silently
    // read past part_buf again the way this once did.
    if (hlen >= sizeof(part_buf)) hlen = sizeof(part_buf) - 1;

    // Timed around BOTH writes, because either can block: the boundary
    // header is small enough to slip into a full send buffer while the JPEG
    // body does not, so timing only the body would miss the case where the
    // socket is already backing up. See sendlog_record() for what this
    // measurement distinguishes.
    int64_t send_t0 = esp_timer_get_time();
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    int64_t send_t1 = esp_timer_get_time();
    sendlog_record(send_t0, (uint32_t)(send_t1 - send_t0), (uint32_t)fb->len, res);
    if (res == ESP_OK) { mark_frame_sent(); frames_sent++; }

    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK) break;
    // No interleaved JSON status part anymore -- this used to push one
    // once/sec (see git history) so the page's fetch()-based multipart
    // parser could read status off this same connection instead of
    // polling /status separately. Now that the page renders this stream
    // via a plain <iframe> (see index_handler()'s big comment for why:
    // fetch() failed outright on iPhone Safari) and reads status via a
    // separate same-origin /status poll instead, nothing consumes a JSON
    // part here anymore -- worse, the iframe's native
    // multipart/x-mixed-replace rendering doesn't know JSON isn't an
    // image, so it "replaced" the displayed frame with blank/non-image
    // content once/sec, visible as a black flash. Pure image/jpeg parts
    // only, now.
  }

  // Exit reason, on serial. res tells you WHY the stream ended (a send
  // failure vs a capture failure vs a clean client disconnect), and the
  // duration tells you whether it survived seconds or minutes -- a run of
  // short-lived connections is the signature of the reload cycle in
  // fix_stall.md section 3.1.
  Serial.printf("stream: connection #%u ended -- %u frames in %.1fs, res=0x%x\n",
                (unsigned)conn, (unsigned)frames_sent,
                (float)(esp_timer_get_time() - conn_start_us) / 1000000.0f,
                (int)res);
  return res;
}

static void start_camera_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_uri_handlers = 6;   // /, /status, /sendlog, /control, /claim
  config.stack_size = 8192;  // default 4096 is tight; index/status are light but keep a margin
  // Same rationale as the stream server below, milder here: these handlers
  // return promptly, but the page polls /status once a second forever, so a
  // client that goes away mid-request shouldn't be able to accumulate slots.
  config.lru_purge_enable = true;

  httpd_uri_t index_uri = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = index_handler,
    .user_ctx = NULL
  };
  httpd_uri_t status_uri = {
    .uri = "/status",
    .method = HTTP_GET,
    .handler = status_handler,
    .user_ctx = NULL
  };
  // On this server rather than the stream server, so it stays answerable
  // while /stream is stalled -- see sendlog_handler(). Fits inside the
  // existing max_uri_handlers above.
  httpd_uri_t sendlog_uri = {
    .uri = "/sendlog",
    .method = HTTP_GET,
    .handler = sendlog_handler,
    .user_ctx = NULL
  };

  httpd_uri_t control_uri = {
    .uri = "/control",
    .method = HTTP_GET,
    .handler = control_handler,
    .user_ctx = NULL
  };

  // Must live on THIS server (port 80), not the stream server -- its whole
  // purpose is to be answerable while port 81's single task is stuck inside
  // stream_handler(). See g_stream_claim.
  httpd_uri_t claim_uri = {
    .uri = "/claim",
    .method = HTTP_GET,
    .handler = claim_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
    httpd_register_uri_handler(camera_httpd, &sendlog_uri);
    httpd_register_uri_handler(camera_httpd, &control_uri);
    httpd_register_uri_handler(camera_httpd, &claim_uri);
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = STREAM_PORT;
  stream_config.ctrl_port = 32769;
  stream_config.max_uri_handlers = 3;  // /stream + /frame, +1 headroom
  // stream_handler()/frame_handler() both call dct_extract_coeffs() +
  // model_forward() per frame (via the shared capture_and_classify_one_frame()
  // helper). The AC-coefficient model's conv-layer intermediate buffers in
  // model_forward() alone sum to ~23.7KB (lum_out+stride2_out+concat_buf+
  // q_ac+ac_pooled+post_concat_out+extra_conv_0_out+q_dc+pooled at the
  // current 15x20 Y grid / 7 AC planes), on top of ~9.6KB for
  // dc_plane+ac_planes -- both far past the 4096-byte default that once
  // crash-looped this device for the plain DC-only model at ~9.6KB total.
  // Sized with real margin above the ~34KB worst-case sum rather than
  // cutting it close. frame_handler() is registered on this SAME server
  // (not camera_httpd, whose stack_size is only 8192 -- nowhere near
  // enough for this) specifically to inherit this sizing for free.
  // 49152 was sized for the 160x120 model. At 320x240 the activation buffers
  // model_forward() puts on the stack roughly quadruple: lum_out alone is
  // 16*30*40 = 19,200 bytes, and dc_plane + chroma_planes + concat_buf +
  // post_concat_out + extra_conv_0_out bring the total to ~80KB before ESP-NN's
  // scratch. A stack overflow here is a reboot or silent corruption, not a
  // clean error, so this is sized with real margin rather than trimmed.
  stream_config.stack_size = 49152;   // buffers are static now, see model_forward_timed()

  // THE "streams fine at 27fps, then drops to zero and stays stuck" fix
  // (2026-08-15). Reproduced deterministically: hold 6 /stream sockets open
  // without reading, and the 7th connection is refused outright -- a new
  // client then NEVER gets served, while /status on port 80 keeps answering
  // normally (so the device looks alive, which is what made this look like
  // an RF problem). Releasing the sockets recovers instantly.
  //
  // Why it happens in normal use: this server has ONE worker task and
  // stream_handler() never returns while a client is connected (see the
  // note at the top of this file), so it can't reap the other sessions.
  // Every abandoned connection -- iframe reload, tab switch, Safari retry,
  // a WiFi hiccup that kills the TCP connection without a clean FIN --
  // permanently consumes one of the default 7 slots. With lru_purge_enable
  // false (the default), a full table means new connections are REFUSED
  // rather than the stalest one being dropped, so the wedge is terminal.
  //
  // lru_purge_enable lets an arriving client evict the least-recently-used
  // session instead. Note the active streamer is itself the LRU here (the
  // LRU counter updates on receive, and a streaming client sends nothing
  // after its GET), so a reconnecting browser displaces the stuck session
  // -- which is exactly the recovery we want, and is what the upstream
  // esp32-camera example does too. max_open_sockets is cut to 4 so the
  // purge engages after a couple of stale connections rather than six.
  stream_config.lru_purge_enable = true;
  stream_config.max_open_sockets = 4;

  // Bound how long ONE slow reader can hold the worker task. Measured on
  // this board before this line existed: a client that connects and then
  // stops reading makes the whole stream server unavailable to everyone
  // else for ~16s (its send buffer fills, httpd_resp_send_chunk() blocks).
  // A client that dies outright is fine -- recovery there was 0.2s -- so
  // the problem case is specifically a SLOW reader, which is exactly what
  // the browser becomes when the WiFi link degrades. Dropping it after a
  // few seconds and letting it reconnect beats freezing every client.
  // Paired with the page's own stream watchdog (see INDEX_HTML), which
  // reloads the iframe when frames stop, so a drop self-heals instead of
  // leaving a frozen picture until someone hits refresh.
  // WAS 1s. Raised to 5s on 2026-08-19 after measurement showed the 1s value
  // was firing on healthy traffic, not on stalled readers.
  //
  // The reasoning for 1s assumed a healthy client absorbs a frame in ~37ms, so
  // 1s was "~27 frames of slack -- it only ever fires on a client that is
  // genuinely not keeping up". That assumption does not survive contact with a
  // lossy link. Measured on this network with /sendlog: sends that block for
  // 0.94s, 1.09s, 1.25s, 1.33s and 1.6s and then COMPLETE SUCCESSFULLY
  // (res=ESP_OK). Those are not failing clients; they are TCP waiting out a
  // retransmit on a link with a few percent packet loss. A 1s timeout sits
  // below this link's own jitter, so it was guaranteed to fire on traffic that
  // was about to succeed.
  //
  // The cost of each spurious fire is out of all proportion to the hiccup:
  // connection dropped -> handler exits -> capture stops (it lives inside the
  // stream loop) -> 4s for the page watchdog to notice -> 250ms teardown -> 3s
  // settle. A 1.3s network hiccup became a ~5s blackout. Measured: 36 straight
  // connections, every one ending res=0xb006 (ESP_ERR_HTTPD_RESP_SEND), none
  // lasting more than a minute.
  //
  // 5s is above the worst block observed (1.6s) with margin. The wedged-reader
  // case the 1s value was defending against is already covered twice over:
  // TCP keepalive detects a vanished peer in ~11s, and lru_purge_enable lets
  // an arriving client evict a stuck session. The trade accepted here is that
  // a genuinely wedged reader now holds the worker for up to ~10s (two
  // blocking sends per frame) instead of ~2s -- worth it to stop destroying
  // healthy connections. See markdown/fix_stall.md section 9.
  stream_config.send_wait_timeout = 5;
  stream_config.recv_wait_timeout = 5;
  // Don't let stale connections queue up several deep behind the one being
  // served: each queued socket costs another full timeout before anyone
  // else is reached, which is what turned a single bad client into a
  // 40-second outage for every client.
  stream_config.backlog_conn = 2;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };
  httpd_uri_t frame_uri = {
    .uri = "/frame",
    .method = HTTP_GET,
    .handler = frame_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &frame_uri);
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
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // Init at QVGA to size the frame buffer/DMA correctly, then drop to
  // 160x120 (FRAMESIZE_QQVGA) with an explicit sensor->set_framesize() call
  // below -- same "init large, then shrink" trick this project has used
  // since the 96x96 config, kept here even though QQVGA isn't known to hit
  // the OV2640's blank-frame quirk the way FRAMESIZE_96X96 did (the quirk
  // was only ever confirmed at 96x96) -- no reason to risk it on an
  // unverified assumption when the proven pattern costs nothing extra.
  // JPEG frames are self-delimited (found via their end marker), so the
  // driver tolerates the sensor's actual output size changing after init.
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = STREAM_JPEG_QUALITY;

  if (psramFound()) {
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  // SILENCE THE CAMERA DRIVER'S LOGGING. A crash fix, not tidiness.
  //
  // Decoded panic on the sibling RGB firmware, 2026-08-23: "Stack canary
  // watchpoint triggered (cam_task)", backtrace cam_hal.c:196 ->
  // esp_log_write -> vprintf -> newlib stdio locks -> xQueueCreateMutex.
  // cam_task blew its stack INSIDE the log call: the driver's
  // frame-buffer-overflow warning drags in the whole newlib vprintf machinery,
  // which needs far more stack than cam_task is given. The act of reporting the
  // problem is what kills the board.
  //
  // Applied here too because this firmware shares the same camera driver and
  // the same trigger: when the WiFi send stalls, the stream loop stops
  // returning frame buffers, the driver runs out, and it logs. Nothing about
  // this project's DCT path makes it immune -- it had simply not been unlucky
  // yet. See markdown/stream_stall_issue.md.
  esp_log_level_set("cam_hal", ESP_LOG_NONE);
  esp_log_level_set("s3 ll_cam", ESP_LOG_NONE);
  esp_log_level_set("camera", ESP_LOG_NONE);

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_framesize(sensor, FRAMESIZE_QQVGA);  // 160x120 -- must match DCT_Y_ROWS/COLS
    sensor->set_quality(sensor, STREAM_JPEG_QUALITY);

    // Live testing showed DC (brightness) sitting near its max quantized
    // value (123/127) while chroma (Cb/Cr) stayed almost entirely near
    // zero (as little as -8..0) in every real captured frame, regardless
    // of scene -- consistent with washed-out/low-saturation color rather
    // than a decode bug (see jpeg_capture.md). None of these were being
    // touched before; all default to sensor/driver defaults otherwise.
    sensor->set_whitebal(sensor, 1);        // auto white balance on
    sensor->set_awb_gain(sensor, 1);
    sensor->set_wb_mode(sensor, 0);         // 0 = auto
    sensor->set_exposure_ctrl(sensor, 1);   // auto exposure on
    // DSP-based AEC off, deliberately: with it on, OV2640's anti-banding/
    // frame-rate-reduction behavior pinned T_capture at a rigid ~101-102ms
    // (~6x the 60Hz mains flicker period) regardless of inference speed --
    // confirmed by ruling out JPEG quality and xclk_freq_hz first (neither
    // changed the number). Off, capture time becomes variable but usually
    // much smaller. Trade-off: aec2 was originally turned on to fix a
    // DC-saturation problem (input's DC feature sitting near max) -- worth
    // re-checking DC/chroma input stats if classification looks off after
    // a lighting change, or if banding artifacts show up under flickering
    // light sources. See capture_time_issue.md.
    sensor->set_aec2(sensor, 0);
    sensor->set_ae_level(sensor, -1);       // target slightly darker exposure (range -2..2) to pull DC back from saturation
    sensor->set_gain_ctrl(sensor, 1);       // auto gain on
    sensor->set_saturation(sensor, 2);      // max (range -2..2) -- amplify whatever chroma signal exists
    sensor->set_special_effect(sensor, 0);  // 0 = no effect (rule out an accidental grayscale/tint mode)
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

  // Re-assert no-power-save AFTER association, and verify it actually took.
  // In modem-sleep the station only wakes on DTIM beacons, so inbound
  // packets sit buffered at the AP for hundreds of ms -- exactly the
  // latency profile that turns a smooth stream into "20fps then drops to
  // zero". The setSleep(false) above runs BEFORE WiFi.begin(); that
  // ordering is not reliably honoured across arduino-esp32 versions
  // (association can reset the mode), hence re-asserting here where the
  // connection exists. esp_wifi_get_ps() is then read back so this is
  // verified rather than assumed -- also reported by /status as "wifi_ps".
  // Note this was NOT the cause of the stall on the sibling esp32_mlp
  // firmware (it read NONE all along); it's kept as a cheap standing
  // rule-out so this class of problem is triageable without a reflash.
  WiFi.setSleep(false);
  wifi_ps_type_t ps = WIFI_PS_NONE;
  if (esp_wifi_get_ps(&ps) == ESP_OK) {
    Serial.printf("WiFi power save mode: %s\n",
                  ps == WIFI_PS_NONE ? "NONE (good, no DTIM latency)" :
                  ps == WIFI_PS_MIN_MODEM ? "MIN_MODEM (!! adds DTIM-interval latency)" :
                  "MAX_MODEM (!! adds large DTIM latency)");
  } else {
    Serial.println("WiFi power save mode: (esp_wifi_get_ps failed)");
  }
}

// Decoder correctness proof, on the real target: decodes real 160x120
// JPEGs captured live from this exact camera (native 4:2:2, see
// jpeg_capture.md Sec7) embedded in real_test_jpegs.h, and checks the
// resulting dc_plane/ac_planes match what the identical decoder produced
// when compiled and run on the host (gcc). This proves the ESP32-S3
// target/compiler produces the same result as the host build, not just
// that the algorithm is right on paper. Non-fatal on failure -- logs
// loudly and lets the device keep running so it stays reachable for
// debugging why.
static void run_decoder_self_test() {
#if !REAL_TEST_JPEGS_MATCH
  // No embedded corpus for this grid -- see the note at real_test_jpegs.h's
  // include. Reported as 0/0 so /status shows "not run" rather than "passed".
  g_decoder_self_test_pass = 0;
  g_decoder_self_test_total = 0;
  Serial.printf("Decoder self-test: SKIPPED -- real_test_jpegs.h is for %dx%d/%dAC, "
                "this build is %dx%d/%dAC\n\n",
                15, 20, 2, DCT_Y_ROWS, DCT_Y_COLS, DCT_NUM_AC_COEFFS);
  return;
#else
  Serial.println("=== Decoder self-test: real camera-captured JPEGs vs host-build ground truth ===");
  int failures = 0;
  for (int i = 0; i < NUM_REAL_TEST_JPEGS; i++) {
    const RealTestJpeg &tj = REAL_TEST_JPEGS[i];
    int32_t dc_plane[DCT_Y_ROWS][DCT_Y_COLS];
    int32_t ac_planes[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS];
    int32_t chroma_planes[2][DCT_C_ROWS][DCT_C_COLS];
    const char *err = nullptr;

    bool ok = dct_extract_coeffs(tj.data, tj.len, dc_plane, ac_planes, chroma_planes, &err);
    if (!ok) {
      Serial.printf("FAIL vector %d: decode error: %s\n", i, err);
      failures++;
      continue;
    }

    bool match = memcmp(dc_plane, tj.dc_plane, sizeof(dc_plane)) == 0 &&
                 memcmp(ac_planes, tj.ac_planes, sizeof(ac_planes)) == 0 &&
                 memcmp(chroma_planes, tj.chroma_planes, sizeof(chroma_planes)) == 0;
    if (match) {
      Serial.printf("PASS vector %d: dc_plane+ac_planes+chroma_planes bit-exact\n", i);
    } else {
      Serial.printf("FAIL vector %d: plane mismatch\n", i);
      failures++;
    }
  }
  g_decoder_self_test_pass = NUM_REAL_TEST_JPEGS - failures;
  g_decoder_self_test_total = NUM_REAL_TEST_JPEGS;
  Serial.printf("Decoder self-test: %d/%d bit-exact\n\n", NUM_REAL_TEST_JPEGS - failures, NUM_REAL_TEST_JPEGS);
#endif
}

// model_forward() correctness proof, on the real target: all 25 known
// (raw planes -> expected logits) test vectors, same discipline
// verify_cnn_c_export.py already used on the PC side.
//
// Since full_esp32_integration.md, this is no longer expected to be
// bit-exact when CONFIG_NN_OPTIMIZED+CONFIG_IDF_TARGET_ESP32S3 are set:
// conv2d_int8() routes through ESP-NN's esp32s3-optimized esp_nn_conv_s8
// for all four conv layers, which uses a different (but validated, see
// full_esp32_integration.md) two-step requantization rounding than this
// project's single combined round-shift -- a real, understood, small
// divergence, not a bug. So this now tracks two separate things instead
// of one pass/fail: bit-exactness (informational -- expected to drop from
// 25/25 on an ESP-NN build, not a failure by itself) and class-match
// (the actual correctness bar -- does the accelerated path still predict
// the same class as the known-good reference). A class mismatch is what
// would actually indicate a problem worth chasing.
// A DC-only model (MODEL_NUM_AC_COEFFS == 0) has no AC planes, so
// verify_cnn_c_export.py emits no TEST_AC_RAW array at all in
// test_vectors.h -- referencing it would not compile. Both model_forward()
// and model_forward_timed() take the pointer but never dereference it in
// that configuration (their AC loops run zero times, and model_forward()
// even casts it to void), so NULL is safe and keeps the three call sites
// below identical across every coefficient count.
#if MODEL_NUM_AC_COEFFS > 0
#define SELF_TEST_AC_RAW(t) TEST_AC_RAW[t]
#else
#define SELF_TEST_AC_RAW(t) NULL
#endif

// The same thing on the chroma side, for a --no-chroma model. The exporter
// keeps model_forward()'s signature fixed across every configuration (it
// casts the unused arguments to void), so only the SELF-TEST call sites need
// this -- the live path in loop() always passes real decoded planes, which
// the decoder fills regardless because chroma DC costs one multiply per MCU
// during the Huffman walk that has already happened.
#if MODEL_USE_CHROMA
#define SELF_TEST_CHROMA_RAW(t) TEST_CHROMA_RAW[t]
#else
#define SELF_TEST_CHROMA_RAW(t) NULL
#endif

// ~80KB of activation buffers at the 320x240 grid, run on the Arduino loop
// task whose stack is deliberately left at 49KB (see
// getArduinoLoopTaskStackSize). The equivalent check still happens on the PC --
// verify_cnn_c_export.py proves the exported C bit-exact against the NumPy
// reference before anything is flashed -- so skipping it here costs the
// ESP-NN-vs-reference comparison, not correctness of the export.
#define SELF_TEST_FITS_ON_LOOP_TASK (MODEL_Y_ROWS * MODEL_Y_COLS <= 15 * 20)

static void run_model_forward_self_test() {
#if !SELF_TEST_FITS_ON_LOOP_TASK
  g_model_bit_exact_count = 0;
  g_model_class_match_count = 0;
  g_model_max_logit_diff = 0;
  Serial.printf("model_forward() self-test: SKIPPED -- %dx%d grid needs ~80KB of "
                "stack, more than the loop task has. PC-side verification still "
                "applies.\n\n", MODEL_Y_ROWS, MODEL_Y_COLS);
  return;
#else
  Serial.println("=== model_forward() self-test: all 25 known test vectors ===");
  int bit_exact = 0, class_matches = 0, max_diff = 0;
  for (int t = 0; t < N_TEST_VECTORS; t++) {
    int8_t logits[MODEL_NUM_CLASSES];
    model_forward(TEST_DC_RAW[t], SELF_TEST_AC_RAW(t), SELF_TEST_CHROMA_RAW(t), logits);

    bool bit_exact_match = memcmp(logits, TEST_EXPECTED_LOGITS[t], MODEL_NUM_CLASSES) == 0;
    int pred = model_argmax(logits, MODEL_NUM_CLASSES);
    bool class_match = pred == TEST_EXPECTED_CLASS[t];
    int vec_max_diff = 0;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
      int d = abs((int)logits[c] - (int)TEST_EXPECTED_LOGITS[t][c]);
      if (d > vec_max_diff) vec_max_diff = d;
    }
    if (vec_max_diff > max_diff) max_diff = vec_max_diff;

    if (bit_exact_match) bit_exact++;
    if (class_match) class_matches++;

    if (!class_match) {
      Serial.printf("FAIL vector %d: predicted class %d (expected %d) -- CLASS MISMATCH, max logit diff %d\n",
                    t, pred, TEST_EXPECTED_CLASS[t], vec_max_diff);
    } else if (!bit_exact_match) {
      Serial.printf("vector %d: class %d correct, logits differ (max diff %d)\n", t, pred, vec_max_diff);
    } else {
      Serial.printf("PASS vector %d: class %d, logits bit-exact\n", t, pred);
    }
  }
  g_model_bit_exact_count = bit_exact;
  g_model_class_match_count = class_matches;
  g_model_max_logit_diff = max_diff;
  Serial.printf("model_forward self-test: %d/%d bit-exact, %d/%d class matches (max logit diff %d)\n\n",
                bit_exact, N_TEST_VECTORS, class_matches, N_TEST_VECTORS, max_diff);
#endif
}

// model_forward_timed() correctness proof: since it's a hand-maintained
// mirror of model_forward()'s body (see the comment above its definition
// for why -- model_weights.h is auto-generated, can't be instrumented in
// place), this checks it produces bit-identical logits to the real
// model_forward() on all 25 known vectors, every boot -- not just a
// one-time check during development. If a future retrain changes the
// architecture (e.g. a second extra_conv layer) this is what would catch
// model_forward_timed() having silently fallen out of sync, rather than
// the stage panel on the webpage just quietly showing numbers for the
// wrong computation.
static void run_stage_timing_self_test() {
#if !SELF_TEST_FITS_ON_LOOP_TASK
  // Same ~80KB of stack as run_model_forward_self_test(), same loop task, same
  // reason. Missing this guard cost a boot loop: the self-test overflowed the
  // 49KB loop stack and panicked with StoreProhibited on every boot, so the
  // device came up, connected to WiFi, printed "Camera stream ready", and then
  // rebooted before anything could connect.
  g_stage_timing_self_test_pass = 0;
  g_stage_timing_self_test_total = 0;
  Serial.printf("stage-timing self-test: SKIPPED -- %dx%d grid needs ~80KB of stack\n\n",
                MODEL_Y_ROWS, MODEL_Y_COLS);
  return;
#endif
  Serial.println("=== model_forward_timed() self-test: matches model_forward() on all 25 known vectors ===");
  int failures = 0;
  for (int t = 0; t < N_TEST_VECTORS; t++) {
    int8_t logits_ref[MODEL_NUM_CLASSES];
    int8_t logits_timed[MODEL_NUM_CLASSES];
    StageTimingsUs stage_us;
    model_forward(TEST_DC_RAW[t], SELF_TEST_AC_RAW(t), SELF_TEST_CHROMA_RAW(t), logits_ref);
    model_forward_timed(TEST_DC_RAW[t], SELF_TEST_AC_RAW(t), SELF_TEST_CHROMA_RAW(t), logits_timed, stage_us);

    if (memcmp(logits_ref, logits_timed, MODEL_NUM_CLASSES) != 0) {
      Serial.printf("FAIL vector %d: model_forward_timed() logits differ from model_forward()\n", t);
      failures++;
    } else {
      Serial.printf("PASS vector %d: bit-exact\n", t);
    }
  }
  g_stage_timing_self_test_pass = N_TEST_VECTORS - failures;
  g_stage_timing_self_test_total = N_TEST_VECTORS;
  Serial.printf("model_forward_timed self-test: %d/%d bit-exact vs model_forward()\n\n", N_TEST_VECTORS - failures, N_TEST_VECTORS);
}

// Scene-independent T_jpeg_parse benchmark for esp32_decoder_optimization_
// prompt.md: decodes each embedded real_test_jpegs.h frame BENCH_ITERS
// times (fixed bytes, not a live camera capture) and averages the wall
// time per decode via esp_timer_get_time(), storing the result for
// /status to report. Run once at boot, after the correctness self-tests
// -- deliberately doesn't trust a fast path's timing without first
// knowing run_decoder_self_test() proved it bit-exact.
#define BENCH_ITERS 30
static void run_decode_benchmark() {
#if !REAL_TEST_JPEGS_MATCH
  // Same reason as run_decoder_self_test(): the benchmark decodes the embedded
  // corpus, which does not exist for this grid.
  g_bench_parse_us = 0.0f;
  Serial.println("Decode benchmark: SKIPPED (no embedded corpus for this grid)\n");
#else
  int32_t dc_plane[DCT_Y_ROWS][DCT_Y_COLS];
  int32_t ac_planes[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS];
  int32_t chroma_planes[2][DCT_C_ROWS][DCT_C_COLS];
  int64_t total_us = 0;
  int total_decodes = 0;
  for (int rep = 0; rep < BENCH_ITERS; rep++) {
    for (int i = 0; i < NUM_REAL_TEST_JPEGS; i++) {
      const RealTestJpeg &tj = REAL_TEST_JPEGS[i];
      const char *err = nullptr;
      int64_t t0 = esp_timer_get_time();
      dct_extract_coeffs(tj.data, tj.len, dc_plane, ac_planes, chroma_planes, &err);
      total_us += esp_timer_get_time() - t0;
      total_decodes++;
    }
  }
  g_bench_parse_us = (float)total_us / total_decodes;
  Serial.printf("Decode benchmark: %.1f us/frame avg over %d decodes (%d embedded frames x %d reps)\n\n",
                g_bench_parse_us, total_decodes, NUM_REAL_TEST_JPEGS, BENCH_ITERS);
#endif
}

// arduino-esp32's main.cpp declares this __attribute__((weak)), specifically
// so user code can override the Arduino loop task's stack size (default
// 8192 bytes) without fighting sdkconfig.h's own CONFIG_ARDUINO_LOOP_STACK_SIZE
// define (a build_flags -D here gets silently clobbered by it -- confirmed
// by the "redefined" warning that shows up if you try). Needed because
// setup() calls run_decoder_self_test()/run_model_forward_self_test() on
// this task, both of which nest a ~9.6KB dc_plane+ac_planes buffer (or
// TEST_DC_RAW/TEST_AC_RAW's stack copy) around a model_forward() call whose
// own conv-layer intermediate buffers sum to ~23.7KB -- see the matching
// comment on stream_config.stack_size in start_camera_server() for the
// itemized breakdown. Same worst-case total, same margin.
size_t getArduinoLoopTaskStackSize(void) {
  // Deliberately NOT raised for the 320x240 grid. This stack and the stream
  // server's both come out of the same ~320KB of internal SRAM, on top of ~91KB
  // static -- two 128KB stacks do not fit, and the failure mode is silent:
  // httpd_start() for the stream server simply returns an error, /status keeps
  // answering from the port-80 instance, and nothing streams. Measured
  // free_heap dropped to 81KB with both at 128KB.
  //
  // So the big stack goes to the task that needs it per frame (the stream
  // handler), and setup()'s model self-test is skipped when the grid is too
  // large for this one -- see run_model_forward_self_test().
  return 49152;
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

  start_camera_server();
  Serial.println("Camera stream ready.");

  // Run after networking is up, not before: gating WiFi/the web server
  // behind ~30 lines of Serial output risked a boot hang with no host
  // serial reader attached (which happened once, forcing a reflash).
  run_decoder_self_test();
  run_model_forward_self_test();
  run_stage_timing_self_test();
  run_decode_benchmark();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    connect_wifi();
  }
  delay(1000);
}
