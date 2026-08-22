/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.
 * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays
 * emitted into model_weights.h. Inputs are the LCG defined below, fed
 * planar uint8 0..255; the model centers by -128 itself.
 * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */

#define SYNTH_NUM_VECTORS 4
static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { 1, 2, 3, 12345 };

static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][7] = {
    {  -10,  -18,  -37,   48,  -35,  -15,  -35 },  /* seed 1 */
    {   -4,  -25,  -33,   36,  -22,  -11,  -28 },  /* seed 2 */
    {    8,  -31,  -33,   36,  -29,  -15,  -23 },  /* seed 3 */
    {    8,  -35,  -31,   27,  -25,   -8,  -30 }  /* seed 12345 */
};

static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { 3, 3, 3, 3 };

/* argmax: seed 1 -> fruit, seed 2 -> fruit, seed 3 -> fruit, seed 12345 -> fruit */

/* Structured patterns -- the PRIMARY geometry check. */
#define PATTERN_NUM_VECTORS 3
static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][7] = {
    {   -1,    7,   -7,  -41,  -17,    9,  -36 },  /* HRAMP */
    {    7,   -3,  -38,  -28,   -1,  -10,  -14 },  /* VRAMP */
    {  -25,   92,  -64,  -55,  -92,  -33,  -64 }  /* CHECKER */
};

static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { 5, 0, 1 };

/* argmax: HRAMP -> furniture, VRAMP -> people, CHECKER -> computer */
