// Minimal baseline-JPEG DC-coefficient decoder. See dct_features.h and
// minimal_decoder.md for the design rationale. Originally a purely
// bit-at-a-time Huffman decode (correctness-first); now includes a
// table-driven fast path per esp32_decoder_optimization_prompt.md levers
// #1+#2, with the original bit-at-a-time decoder kept intact as both the
// fallback for rare long codes and, with DCT_FAST_HUFFMAN=0, a standalone
// reference build for differential testing against the fast path.

#include "dct_features.h"
#include <string.h>

// Lever #3 (IRAM_ATTR on this hot path) was tried and measured: no change
// (925.8us vs 926.4us/frame on the scene-independent on-device benchmark,
// well within run-to-run noise). Per the optimization prompt's own
// caution -- "only do this after profiling shows flash-cache misses are
// actually contributing... don't apply it speculatively before confirming
// it matters" -- reverted rather than left in unproven. Worth
// reconsidering only if a future profiling pass shows flash-cache misses
// specifically, not as a default optimization.

// Lever #1 (table-driven Huffman decode) + #2 (bulk bit-buffer refill),
// coupled per the optimization prompt: #1 needs peek_bits(9) to have >=9
// bits already buffered, which the original one-byte-at-a-time fill_byte()
// can't guarantee cheaply. Set to 0 to force every symbol through the
// original bit-at-a-time path (decode_huff_symbol_slow) -- this is the
// kept-around correctness oracle for differential testing, not just a
// theoretical fallback.
#ifndef DCT_FAST_HUFFMAN
#define DCT_FAST_HUFFMAN 1
#endif
#define HUFF_FAST_BITS 9

namespace {

// ---- Bit-level entropy reader, MSB-first, with JPEG byte destuffing ----
//
// JPEG entropy-coded data escapes every literal 0xFF byte as 0xFF 0x00.
// A 0xFF NOT followed by 0x00 begins a real marker (RST0-7 at a restart
// boundary, or EOI at the end of the scan) -- fill_byte() stops there
// without consuming the marker, leaving `pos` pointing at the 0xFF so the
// caller can inspect/consume it explicitly at an expected MCU boundary.
struct BitReader {
    const uint8_t *data;
    size_t len;
    size_t pos;
    uint32_t bit_buf;
    int bit_count;

    bool fill_byte() {
        if (pos >= len) return false;
        uint8_t b = data[pos++];
        if (b == 0xFF) {
            if (pos >= len) { pos--; return false; }
            uint8_t next = data[pos];
            if (next == 0x00) {
                pos++;  // destuff: genuine 0xFF data byte
            } else {
                pos--;  // un-consume the 0xFF; caller handles the marker
                return false;
            }
        }
        bit_buf = (bit_buf << 8) | b;
        bit_count += 8;
        return true;
    }

    int get_bit() {
        if (bit_count == 0) {
            if (!fill_byte()) return -1;
        }
        bit_count--;
        return (bit_buf >> bit_count) & 1;
    }

    int get_bits(int n) {
        int v = 0;
        for (int i = 0; i < n; i++) {
            int b = get_bit();
            if (b < 0) return -1;
            v = (v << 1) | b;
        }
        return v;
    }

    void byte_align_discard() {
        bit_buf = 0;
        bit_count = 0;
    }

    // Lever #2: greedily top up bit_buf (32 bits wide) instead of the
    // single-byte-at-a-time refill get_bit() does on its own -- still walks
    // fill_byte() one byte at a time internally (preserving its existing
    // 0xFF-destuffing/marker-stop logic exactly), just calls it repeatedly
    // up front instead of once per exhausted bit. Stops early, with fewer
    // than 32 bits buffered, at a marker/end-of-data -- callers must check
    // bit_count, not assume a full refill succeeded.
    void refill() {
        while (bit_count <= 24) {
            if (!fill_byte()) break;
        }
    }

    // Non-consuming look-ahead at the next n bits (MSB-first, same bit
    // order get_bit() would yield one at a time). Returns -1 if fewer than
    // n bits are available even after refill() (near a marker/end of
    // data) -- callers must fall back to get_bit()-based decoding in that
    // case, same as the pre-optimization behavior.
    int peek_bits(int n) {
        if (bit_count < n) refill();
        if (bit_count < n) return -1;
        return (bit_buf >> (bit_count - n)) & ((1u << n) - 1);
    }

    // Consumes n bits already confirmed present by a prior peek_bits(n)
    // call that returned >= 0. Does not itself validate bit_count >= n.
    void consume(int n) {
        bit_count -= n;
    }
};

// ---- Huffman tables (standard JPEG Annex C/F code assignment) ----

struct HuffTable {
    uint8_t bits[17];      // bits[1..16] = count of codes of that length
    uint8_t huffval[256];
    int nvals = 0;
    int mincode[17];
    int maxcode[17];       // -1 => no codes of this length
    int valptr[17];

    // Lever #1: direct-lookup table for every canonical code <= HUFF_FAST_BITS
    // bits long, indexed by the next HUFF_FAST_BITS bits of the stream
    // (MSB-first, left-justified -- i.e. a short L-bit code's table region
    // is every HUFF_FAST_BITS-bit pattern sharing its top L bits).
    // fast_length[p] == 0 means "no code this short matches prefix p",
    // meaning the real code is > HUFF_FAST_BITS bits and must fall back to
    // the bit-at-a-time path.
    uint8_t fast_symbol[1 << HUFF_FAST_BITS];
    uint8_t fast_length[1 << HUFF_FAST_BITS];
};

// Builds mincode/maxcode/valptr exactly as before (same code/k progression,
// just restructured into a per-value loop instead of a per-length bulk
// add) so lever #1's fast_symbol/fast_length can be populated from the
// same single pass -- one source of truth for code assignment, no risk of
// the fast table and the slow-path tables disagreeing.
void build_huff_table(HuffTable &t) {
    memset(t.fast_length, 0, sizeof(t.fast_length));
    int code = 0, k = 0;
    for (int l = 1; l <= 16; l++) {
        if (t.bits[l] == 0) {
            t.maxcode[l] = -1;
            code <<= 1;
            continue;
        }
        t.valptr[l] = k;
        t.mincode[l] = code;
        for (int i = 0; i < t.bits[l]; i++) {
            if (l <= HUFF_FAST_BITS) {
                int shift = HUFF_FAST_BITS - l;
                int base = code << shift;
                int count = 1 << shift;
                uint8_t sym = t.huffval[k];
                for (int j = 0; j < count; j++) {
                    t.fast_symbol[base + j] = sym;
                    t.fast_length[base + j] = (uint8_t)l;
                }
            }
            code++;
            k++;
        }
        t.maxcode[l] = code - 1;
        code <<= 1;
    }
}

// Bit-at-a-time canonical Huffman decode -- the original correctness
// reference. Still load-bearing, not just a historical artifact: it's the
// fallback for codes longer than HUFF_FAST_BITS bits, and (with
// DCT_FAST_HUFFMAN=0) the entire decode path for differential testing
// against the fast table-driven one below.
int decode_huff_symbol_slow(BitReader &br, const HuffTable &t) {
    int code = 0;
    for (int l = 1; l <= 16; l++) {
        int bit = br.get_bit();
        if (bit < 0) return -1;
        code = (code << 1) | bit;
        if (t.maxcode[l] != -1 && code <= t.maxcode[l]) {
            int idx = t.valptr[l] + (code - t.mincode[l]);
            if (idx < 0 || idx >= t.nvals) return -1;
            return t.huffval[idx];
        }
    }
    return -1;
}

// Lever #1: table-driven fast path. Peeks HUFF_FAST_BITS bits without
// consuming; if a code of that length or shorter matches (the common case
// -- canonical JPEG Huffman assignment keeps frequent symbols short by
// construction), consumes exactly that many bits and returns directly, no
// bit-at-a-time branching. On a miss (code genuinely longer than
// HUFF_FAST_BITS bits, or fewer than HUFF_FAST_BITS bits left in the
// stream near a marker/EOB boundary) falls back to decode_huff_symbol_slow
// -- safe because peek_bits() never consumes, so the slow path re-walks
// the exact same unconsumed bits from scratch.
int decode_huff_symbol(BitReader &br, const HuffTable &t) {
#if DCT_FAST_HUFFMAN
    int peek = br.peek_bits(HUFF_FAST_BITS);
    if (peek >= 0) {
        int len = t.fast_length[peek];
        if (len > 0) {
            br.consume(len);
            return t.fast_symbol[peek];
        }
    }
#endif
    return decode_huff_symbol_slow(br, t);
}

// Standard JPEG DC/AC magnitude decode with sign extension (Annex F.2.2.1).
int32_t receive_extend(BitReader &br, int size, bool *ok) {
    if (size == 0) return 0;
    int v = br.get_bits(size);
    if (v < 0) { *ok = false; return 0; }
    if (v < (1 << (size - 1))) {
        v = v - (1 << size) + 1;
    }
    return v;
}

// Decodes one 8x8 block: DC (always present, DPCM against dc_pred) plus
// every AC symbol up to EOB. Per minimal_decoder.md's "decode to EOB, keep
// only what you need": each (run, size) symbol advances the zig-zag
// pointer by run+1 positions -- that position IS the zig-zag index, so a
// value landing at position k (1 <= k < DCT_NUM_COEFFS) is stored directly
// at ac_levels_out[k-1]; anything at k >= DCT_NUM_COEFFS is still decoded
// (required to stay bitstream-synced) but discarded. ac_levels_out may be
// nullptr for blocks whose AC values are never needed (Cb/Cr) -- the decode
// work still has to happen either way.
bool decode_block(BitReader &br, const HuffTable &dc_table, const HuffTable &ac_table,
                             int32_t &dc_pred, int32_t &dc_level_out, int32_t *ac_levels_out) {
    int s = decode_huff_symbol(br, dc_table);
    if (s < 0 || s > 11) return false;
    bool ok = true;
    int32_t diff = (s == 0) ? 0 : receive_extend(br, s, &ok);
    if (!ok) return false;
    dc_pred += diff;
    dc_level_out = dc_pred;

    if (ac_levels_out) {
        for (int i = 0; i < DCT_NUM_AC_COEFFS; i++) ac_levels_out[i] = 0;
    }

    int k = 1;
    while (k < 64) {
        int rs = decode_huff_symbol(br, ac_table);
        if (rs < 0) return false;
        int run = rs >> 4;
        int size = rs & 0x0F;
        if (size == 0) {
            if (run == 15) { k += 16; continue; }  // ZRL
            break;                                  // EOB
        }
        k += run;
        if (k >= 64) break;  // defensive: malformed stream
        int32_t val = receive_extend(br, size, &ok);
        if (!ok) return false;
        if (ac_levels_out && k < DCT_NUM_COEFFS) ac_levels_out[k - 1] = val;
        k += 1;
    }
    return true;
}

// ---- Marker-segment parsing ----

struct JpegParseState {
    // quant[tid][k] = dequantization multiplier for zig-zag position k of
    // DQT table tid. DQT segments store their 64 entries in zig-zag order
    // already (position 0 == natural (0,0), the DC entry), so this is a
    // direct slice of the first DCT_NUM_COEFFS bytes/words of the segment --
    // no reordering needed.
    int32_t quant[4][DCT_NUM_COEFFS] = {{0}};
    bool have_quant[4] = {false, false, false, false};

    HuffTable dc_huff[4];
    HuffTable ac_huff[4];
    bool have_dc_huff[4] = {false, false, false, false};
    bool have_ac_huff[4] = {false, false, false, false};

    int width = 0, height = 0, num_components = 0;
    struct { int id, h, v, qt; } comp[3];

    struct { int comp_id, dc_tbl, ac_tbl; } scan_comp[3];

    int restart_interval = 0;
};

bool parse_markers(const uint8_t *d, size_t len, JpegParseState &st, size_t &sos_data_start, const char **err) {
    if (len < 4 || d[0] != 0xFF || d[1] != 0xD8) { *err = "no SOI"; return false; }
    size_t p = 2;
    bool got_sof = false;

    while (p + 1 < len) {
        if (d[p] != 0xFF) { *err = "marker sync lost"; return false; }
        while (p < len && d[p] == 0xFF) p++;  // skip 0xFF fill bytes
        if (p >= len) { *err = "truncated marker"; return false; }
        uint8_t marker = d[p];
        p++;

        if (marker == 0xD9) { *err = "EOI before SOS"; return false; }
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;  // standalone, no payload

        if (p + 2 > len) { *err = "truncated segment length"; return false; }
        int seg_len = (d[p] << 8) | d[p + 1];
        size_t payload_start = p + 2;
        if (seg_len < 2 || payload_start + (size_t)(seg_len - 2) > len) { *err = "segment overruns buffer"; return false; }
        size_t payload_len = seg_len - 2;

        switch (marker) {
            case 0xDB: {  // DQT -- may pack multiple tables
                size_t q = payload_start, qend = payload_start + payload_len;
                while (q < qend) {
                    uint8_t pq_tq = d[q]; q++;
                    int precision = pq_tq >> 4;
                    int tid = pq_tq & 0x0F;
                    if (tid > 3) { *err = "bad DQT id"; return false; }
                    if (precision == 0) {
                        if (q + 64 > qend) { *err = "DQT truncated"; return false; }
                        for (int i = 0; i < DCT_NUM_COEFFS; i++) st.quant[tid][i] = d[q + i];
                        q += 64;
                    } else {
                        if (q + 128 > qend) { *err = "DQT16 truncated"; return false; }
                        for (int i = 0; i < DCT_NUM_COEFFS; i++)
                            st.quant[tid][i] = (d[q + 2 * i] << 8) | d[q + 2 * i + 1];
                        q += 128;
                    }
                    st.have_quant[tid] = true;
                }
                break;
            }
            case 0xC4: {  // DHT -- may pack multiple tables
                size_t q = payload_start, qend = payload_start + payload_len;
                while (q < qend) {
                    uint8_t tc_th = d[q]; q++;
                    int tclass = tc_th >> 4;
                    int tid = tc_th & 0x0F;
                    if (tid > 3) { *err = "bad DHT id"; return false; }
                    if (q + 16 > qend) { *err = "DHT truncated"; return false; }
                    HuffTable &t = (tclass == 0) ? st.dc_huff[tid] : st.ac_huff[tid];
                    int total = 0;
                    for (int i = 1; i <= 16; i++) { t.bits[i] = d[q + i - 1]; total += t.bits[i]; }
                    q += 16;
                    if (total > 256 || q + (size_t)total > qend) { *err = "DHT values truncated"; return false; }
                    for (int i = 0; i < total; i++) t.huffval[i] = d[q + i];
                    t.nvals = total;
                    q += total;
                    build_huff_table(t);
                    if (tclass == 0) st.have_dc_huff[tid] = true; else st.have_ac_huff[tid] = true;
                }
                break;
            }
            case 0xC0: case 0xC1: {  // SOF0 baseline / SOF1 extended-sequential
                size_t q = payload_start;
                int precision = d[q]; q++;
                if (precision != 8) { *err = "unsupported SOF precision"; return false; }
                st.height = (d[q] << 8) | d[q + 1]; q += 2;
                st.width = (d[q] << 8) | d[q + 1]; q += 2;
                st.num_components = d[q]; q++;
                if (st.num_components != 3) { *err = "expected 3 SOF components"; return false; }
                for (int c = 0; c < 3; c++) {
                    st.comp[c].id = d[q];
                    st.comp[c].h = d[q + 1] >> 4;
                    st.comp[c].v = d[q + 1] & 0x0F;
                    st.comp[c].qt = d[q + 2];
                    q += 3;
                }
                got_sof = true;
                break;
            }
            case 0xC2: { *err = "progressive JPEG not supported"; return false; }
            case 0xDD: {  // DRI
                if (payload_len != 2) { *err = "bad DRI length"; return false; }
                st.restart_interval = (d[payload_start] << 8) | d[payload_start + 1];
                break;
            }
            case 0xDA: {  // SOS -- entropy-coded data follows immediately after
                if (!got_sof) { *err = "SOS before SOF"; return false; }
                size_t q = payload_start;
                int ns = d[q]; q++;
                if (ns != 3) { *err = "expected 3 scan components"; return false; }
                for (int c = 0; c < 3; c++) {
                    st.scan_comp[c].comp_id = d[q];
                    st.scan_comp[c].dc_tbl = d[q + 1] >> 4;
                    st.scan_comp[c].ac_tbl = d[q + 1] & 0x0F;
                    q += 2;
                }
                q += 3;  // Ss, Se, AhAl -- fixed for baseline, unused here
                sos_data_start = q;
                return true;
            }
            default:
                break;  // APPn, COM, etc: not needed, skip via payload_len below
        }
        p = payload_start + payload_len;
    }
    *err = "reached end of buffer without SOS";
    return false;
}

}  // namespace

bool dct_extract_coeffs(const uint8_t *jpeg, size_t len,
                         int32_t dc_plane[DCT_Y_ROWS][DCT_Y_COLS],
                         int32_t ac_planes[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS],
                         int32_t chroma_planes[2][DCT_C_ROWS][DCT_C_COLS],
                         const char **err) {
    JpegParseState st;
    size_t sos_data_start = 0;
    if (!parse_markers(jpeg, len, st, sos_data_start, err)) return false;

    if (st.width != DCT_Y_COLS * 8 || st.height != DCT_Y_ROWS * 8) {
        *err = "unexpected JPEG dimensions (expected 160x120)";
        return false;
    }
    // The OV2640's JPEG encoder emits 4:2:2 (Y: H=2,V=1) at this
    // resolution -- confirmed on real hardware at both 96x96 and 160x120,
    // see jpeg_capture.md Sec7. Checked here so an unexpected sensor/format
    // change fails loudly instead of silently desyncing the Cb/Cr block
    // walk below -- the one-Cb-block-and-one-Cr-block-per-MCU chroma
    // indexing this function relies on is only valid for this exact 4:2:2
    // ratio.
    if (st.comp[0].h != 2 || st.comp[0].v != 1 ||
        st.comp[1].h != 1 || st.comp[1].v != 1 ||
        st.comp[2].h != 1 || st.comp[2].v != 1) {
        *err = "unexpected chroma subsampling (expected OV2640's native 4:2:2)";
        return false;
    }
    for (int c = 0; c < 3; c++) {
        int qt = st.comp[c].qt, dc_t = st.scan_comp[c].dc_tbl, ac_t = st.scan_comp[c].ac_tbl;
        if (qt > 3 || !st.have_quant[qt]) { *err = "missing DQT for a component"; return false; }
        if (dc_t > 3 || !st.have_dc_huff[dc_t]) { *err = "missing DC DHT for a component"; return false; }
        if (ac_t > 3 || !st.have_ac_huff[ac_t]) { *err = "missing AC DHT for a component"; return false; }
    }

    BitReader br;
    br.data = jpeg;
    br.len = len;
    br.pos = sos_data_start;
    br.bit_buf = 0;
    br.bit_count = 0;

    // DC predictors for all 3 components must still be tracked correctly
    // (DPCM is defined against the previous block of the *same* component,
    // regardless of whether we keep the result) even though Cb/Cr's
    // decoded values (dc_pred[1], dc_pred[2]) are never read afterward --
    // only Y's bit-stream position needs to end up correct, and that
    // depends on every preceding block (including Cb/Cr) being decoded,
    // not on what we do with their output.
    int32_t dc_pred[3] = {0, 0, 0};

    // MCU = 16(w)x8(h) pixels under 4:2:2 (Hmax=2, Vmax=1): 2 Y blocks side
    // by side, 1 Cb block, 1 Cr block (decoded and discarded). mcu_rows is
    // therefore DCT_Y_ROWS directly (one MCU row per Y block row).
    const int mcu_cols = st.width / 16;
    const int mcu_rows = st.height / 8;
    if (mcu_rows != DCT_Y_ROWS || mcu_cols * 2 != DCT_Y_COLS) {
        *err = "unexpected MCU grid for 160x120 4:2:2";
        return false;
    }
    int mcus_since_restart = 0;
    int next_rst = 0;  // cycles 0..7

    for (int my = 0; my < mcu_rows; my++) {
        for (int mx = 0; mx < mcu_cols; mx++) {
            for (int cx = 0; cx < 2; cx++) {
                int32_t dc_level;
                int32_t ac_levels[DCT_NUM_AC_COEFFS];
                if (!decode_block(br, st.dc_huff[st.scan_comp[0].dc_tbl], st.ac_huff[st.scan_comp[0].ac_tbl],
                                   dc_pred[0], dc_level, ac_levels)) {
                    *err = "Y block entropy decode failed";
                    return false;
                }
                int by = my, bx = mx * 2 + cx;
                const int32_t *y_quant = st.quant[st.comp[0].qt];
                dc_plane[by][bx] = dc_level * y_quant[0];
                for (int k = 0; k < DCT_NUM_AC_COEFFS; k++)
                    ac_planes[k][by][bx] = ac_levels[k] * y_quant[k + 1];
            }

            // Cb, Cr: one block each per MCU under this project's fixed
            // 4:2:2 sampling, so (my, mx) IS the chroma plane index
            // directly -- no separate indexing math, same as the Y blocks'
            // dc_plane[my][mx*2+cx] above. AC is still decoded to EOB (to
            // stay bitstream-synced for the next MCU's Y blocks) but never
            // requested (nullptr) -- this model is chroma-DC-only.
            for (int c = 1; c <= 2; c++) {
                int32_t dc_level;
                if (!decode_block(br, st.dc_huff[st.scan_comp[c].dc_tbl], st.ac_huff[st.scan_comp[c].ac_tbl],
                                   dc_pred[c], dc_level, nullptr)) {
                    *err = (c == 1) ? "Cb block entropy decode failed" : "Cr block entropy decode failed";
                    return false;
                }
                const int32_t *c_quant = st.quant[st.comp[c].qt];
                chroma_planes[c - 1][my][mx] = dc_level * c_quant[0];
            }

            mcus_since_restart++;
            bool is_last_mcu = (my == mcu_rows - 1 && mx == mcu_cols - 1);
            if (st.restart_interval > 0 && mcus_since_restart == st.restart_interval && !is_last_mcu) {
                br.byte_align_discard();
                if (br.pos + 1 >= br.len || jpeg[br.pos] != 0xFF || jpeg[br.pos + 1] != (0xD0 + next_rst)) {
                    *err = "expected restart marker not found";
                    return false;
                }
                br.pos += 2;
                dc_pred[0] = dc_pred[1] = dc_pred[2] = 0;
                next_rst = (next_rst + 1) & 0x07;
                mcus_since_restart = 0;
            }
        }
    }

    return true;
}
