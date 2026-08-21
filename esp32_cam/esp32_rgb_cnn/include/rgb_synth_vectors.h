/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.
 * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays
 * emitted into model_weights.h. Inputs are the LCG defined below, fed
 * planar uint8 0..255; the model centers by -128 itself.
 * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */

#define SYNTH_NUM_VECTORS 4
static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { 1, 2, 3, 12345 };

static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][5] = {
    {  -59,   -2,  -15,   34,  -62 },  /* seed 1 */
    {  -32,    8,   -1,   -5,  -60 },  /* seed 2 */
    {  -37,  -29,   -8,    0,  -14 },  /* seed 3 */
    {    9,  -35,  -29,   15,  -30 }  /* seed 12345 */
};

static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { 3, 1, 3, 3 };

/* argmax: seed 1 -> fruit, seed 2 -> computer, seed 3 -> fruit, seed 12345 -> fruit */

/* Structured patterns -- the PRIMARY geometry check. */
#define PATTERN_NUM_VECTORS 3
static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][5] = {
    {   33,  -18,   -6,  -20,  -47 },  /* HRAMP */
    {   19,   -6,  -15,   -8,  -31 },  /* VRAMP */
    {  -23,   83,  -41,  -66, -101 }  /* CHECKER */
};

static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { 0, 0, 1 };

/* argmax: HRAMP -> people, VRAMP -> people, CHECKER -> computer */
