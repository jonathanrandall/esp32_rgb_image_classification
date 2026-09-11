/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.
 * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays
 * emitted into model_weights.h. Inputs are the LCG defined below, fed
 * planar uint8 0..255; the model centers by -128 itself.
 * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */

#define SYNTH_NUM_VECTORS 4
static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { 1, 2, 3, 12345 };

static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][5] = {
    {  -25,   33,  -23,  -10,  -20 },  /* seed 1 */
    {  -21,   16,   -4,    3,  -13 },  /* seed 2 */
    {  -24,   22,  -13,    1,  -14 },  /* seed 3 */
    {  -31,   24,    5,   -3,  -14 }  /* seed 12345 */
};

static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { 1, 1, 1, 1 };

/* argmax: seed 1 -> fruit, seed 2 -> fruit, seed 3 -> fruit, seed 12345 -> fruit */

/* Structured patterns -- the PRIMARY geometry check. */
#define PATTERN_NUM_VECTORS 3
static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][5] = {
    {    9,  -52,  -18,   32,  -51 },  /* HRAMP */
    {    2,  -46,    8,   -4,  -19 },  /* VRAMP */
    {   82,  -66,  -33,  -51,  -79 }  /* CHECKER */
};

static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { 3, 2, 0 };

/* argmax: HRAMP -> doors, VRAMP -> people, CHECKER -> computer */
