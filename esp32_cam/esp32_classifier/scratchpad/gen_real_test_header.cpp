// Regenerates include/real_test_jpegs.h for the current DCT_NUM_AC_COEFFS=2
// + chroma-DC config, reusing the same embedded raw JPEG bytes as before
// (extracted into jpeg_bytes_only.h). Decodes each with the current
// (host-buildable) dct_features.cpp and prints a full replacement header
// to stdout, including a new chroma_planes field/array per test JPEG.
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include "dct_features.h"
#include "jpeg_bytes_only.h"

static void print_array_u8(const char *name, const uint8_t *data, size_t len) {
    printf("static const uint8_t %s[] = {\n", name);
    for (size_t i = 0; i < len; i++) {
        printf("%d, ", data[i]);
        if (i % 20 == 19) printf("\n");
    }
    printf("\n};\n\n");
}

static void print_dc(const char *name, int32_t dc[DCT_Y_ROWS][DCT_Y_COLS]) {
    printf("static const int32_t %s[DCT_Y_ROWS][DCT_Y_COLS] = {\n", name);
    for (int y = 0; y < DCT_Y_ROWS; y++) {
        printf("  {");
        for (int x = 0; x < DCT_Y_COLS; x++) printf("%d, ", dc[y][x]);
        printf("},\n");
    }
    printf("};\n\n");
}

static void print_ac(const char *name, int32_t ac[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS]) {
    printf("static const int32_t %s[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS] = {\n", name);
    for (int k = 0; k < DCT_NUM_AC_COEFFS; k++) {
        printf(" {\n");
        for (int y = 0; y < DCT_Y_ROWS; y++) {
            printf("  {");
            for (int x = 0; x < DCT_Y_COLS; x++) printf("%d, ", ac[k][y][x]);
            printf("},\n");
        }
        printf(" },\n");
    }
    printf("};\n\n");
}

static void print_chroma(const char *name, int32_t chroma[2][DCT_C_ROWS][DCT_C_COLS]) {
    printf("static const int32_t %s[2][DCT_C_ROWS][DCT_C_COLS] = {\n", name);
    for (int c = 0; c < 2; c++) {
        printf(" {\n");
        for (int y = 0; y < DCT_C_ROWS; y++) {
            printf("  {");
            for (int x = 0; x < DCT_C_COLS; x++) printf("%d, ", chroma[c][y][x]);
            printf("},\n");
        }
        printf(" },\n");
    }
    printf("};\n\n");
}

int main() {
    struct Entry { const uint8_t *data; size_t len; };
    Entry entries[] = {
        {REAL_JPEG_0, sizeof(REAL_JPEG_0)},
        {REAL_JPEG_1, sizeof(REAL_JPEG_1)},
        {REAL_JPEG_2, sizeof(REAL_JPEG_2)},
    };
    const int n = 3;

    printf("#pragma once\n");
    printf("// Real camera-captured 160x120 JPEGs (native 4:2:2) + their expected\n");
    printf("// dc_plane/ac_planes/chroma_planes, computed by this same decoder compiled\n");
    printf("// on the host (gcc). Proves the identical source produces the same result on\n");
    printf("// the actual ESP32-S3 target, not just on a PC. Regenerate with\n");
    printf("// scratchpad/gen_real_test_header.cpp if the decoder logic changes.\n");
    printf("#include <stdint.h>\n\n");
    printf("struct RealTestJpeg {\n");
    printf("  const uint8_t *data;\n");
    printf("  size_t len;\n");
    printf("  const int32_t (*dc_plane)[DCT_Y_COLS];\n");
    printf("  const int32_t (*ac_planes)[DCT_Y_ROWS][DCT_Y_COLS];\n");
    printf("  const int32_t (*chroma_planes)[DCT_C_ROWS][DCT_C_COLS];\n");
    printf("};\n\n");

    for (int i = 0; i < n; i++) {
        char nbuf[64];
        snprintf(nbuf, sizeof(nbuf), "REAL_JPEG_%d", i);
        print_array_u8(nbuf, entries[i].data, entries[i].len);

        int32_t dc_plane[DCT_Y_ROWS][DCT_Y_COLS];
        int32_t ac_planes[DCT_NUM_AC_COEFFS][DCT_Y_ROWS][DCT_Y_COLS];
        int32_t chroma_planes[2][DCT_C_ROWS][DCT_C_COLS];
        const char *err = nullptr;
        bool ok = dct_extract_coeffs(entries[i].data, entries[i].len, dc_plane, ac_planes, chroma_planes, &err);
        if (!ok) {
            fprintf(stderr, "decode failed for entry %d: %s\n", i, err);
            return 1;
        }

        snprintf(nbuf, sizeof(nbuf), "REAL_DC_%d", i);
        print_dc(nbuf, dc_plane);
        snprintf(nbuf, sizeof(nbuf), "REAL_AC_%d", i);
        print_ac(nbuf, ac_planes);
        snprintf(nbuf, sizeof(nbuf), "REAL_CHROMA_%d", i);
        print_chroma(nbuf, chroma_planes);
    }

    printf("#define NUM_REAL_TEST_JPEGS %d\n", n);
    printf("static const RealTestJpeg REAL_TEST_JPEGS[NUM_REAL_TEST_JPEGS] = {\n");
    for (int i = 0; i < n; i++) {
        printf("  { REAL_JPEG_%d, sizeof(REAL_JPEG_%d), REAL_DC_%d, REAL_AC_%d, REAL_CHROMA_%d },\n",
               i, i, i, i, i);
    }
    printf("};\n");

    return 0;
}
