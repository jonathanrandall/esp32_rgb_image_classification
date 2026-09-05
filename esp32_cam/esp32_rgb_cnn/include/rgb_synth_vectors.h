/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.
 * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays
 * emitted into model_weights.h. Inputs are the LCG defined below, fed
 * planar uint8 0..255; the model centers by -128 itself.
 * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */

#define SYNTH_NUM_VECTORS 4
static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { 1, 2, 3, 12345 };

static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][6] = {
    {  -21,  -36,  -32,   53,    0,  -45 },  /* seed 1 */
    {  -23,  -29,  -12,   21,   15,  -46 },  /* seed 2 */
    {  -15,  -34,  -23,   37,    4,  -39 },  /* seed 3 */
    {   -1,  -50,  -23,   42,   -9,  -37 }  /* seed 12345 */
};

static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { 3, 3, 3, 3 };

/* argmax: seed 1 -> fruit, seed 2 -> fruit, seed 3 -> fruit, seed 12345 -> fruit */

/* Structured patterns -- the PRIMARY geometry check. */
#define PATTERN_NUM_VECTORS 3
static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][6] = {
    {  -34,   20,   56,  -65,  -42,  -17 },  /* HRAMP */
    {   12,    1,  -13,  -37,   -5,  -25 },  /* VRAMP */
    {  -40,   80,  -53,  -56,  -97,  -46 }  /* CHECKER */
};

static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { 2, 0, 1 };

/* argmax: HRAMP -> doors, VRAMP -> people, CHECKER -> computer */
