/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.
 * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays
 * emitted into model_weights.h. Inputs are the LCG defined below, fed
 * planar uint8 0..255; the model centers by -128 itself.
 * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */

#define SYNTH_NUM_VECTORS 4
static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { 1, 2, 3, 12345 };

static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][5] = {
    {  -13,  -40,  -23,   46,  -23 },  /* seed 1 */
    {   -7,  -47,  -13,   32,   -5 },  /* seed 2 */
    {   -2,  -39,  -20,   28,   -7 },  /* seed 3 */
    {    8,  -37,  -16,   19,  -13 }  /* seed 12345 */
};

static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { 3, 3, 3, 3 };

/* argmax: seed 1 -> fruit, seed 2 -> fruit, seed 3 -> fruit, seed 12345 -> fruit */

/* Structured patterns -- the PRIMARY geometry check. */
#define PATTERN_NUM_VECTORS 3
static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][5] = {
    {  -19,   20,   17,  -54,  -16 },  /* HRAMP */
    {  -22,   17,    7,  -45,   -8 },  /* VRAMP */
    {    7,   76,  -74,  -68,  -94 }  /* CHECKER */
};

static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { 1, 1, 1 };

/* argmax: HRAMP -> computer, VRAMP -> computer, CHECKER -> computer */
