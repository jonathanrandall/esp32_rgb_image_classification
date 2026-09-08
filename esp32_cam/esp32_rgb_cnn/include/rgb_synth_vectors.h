/* Synthetic-input bit-exactness vectors for the RGB CNN int8 model.
 * Generated from output/rgb_cnn/quantized_model.npz -- the same arrays
 * emitted into model_weights.h. Inputs are the LCG defined below, fed
 * planar uint8 0..255; the model centers by -128 itself.
 * Regenerate with verify_rgb_cnn_c_export.py after any retrain. */

#define SYNTH_NUM_VECTORS 4
static const uint32_t SYNTH_SEEDS[SYNTH_NUM_VECTORS] = { 1, 2, 3, 12345 };

static const int8_t SYNTH_EXPECTED_LOGITS[SYNTH_NUM_VECTORS][5] = {
    {  -26,   36,  -31,  -29,   -5 },  /* seed 1 */
    {  -34,   38,  -14,  -16,  -12 },  /* seed 2 */
    {  -31,   33,   -2,  -19,  -10 },  /* seed 3 */
    {  -40,   46,   -1,  -21,  -19 }  /* seed 12345 */
};

static const uint8_t SYNTH_EXPECTED_CLASS[SYNTH_NUM_VECTORS] = { 1, 1, 1, 1 };

/* argmax: seed 1 -> fruit, seed 2 -> fruit, seed 3 -> fruit, seed 12345 -> fruit */

/* Structured patterns -- the PRIMARY geometry check. */
#define PATTERN_NUM_VECTORS 3
static const int8_t PATTERN_EXPECTED_LOGITS[PATTERN_NUM_VECTORS][5] = {
    {   13,  -45,   12,  -11,  -34 },  /* HRAMP */
    {   -4,  -25,   12,    9,  -47 },  /* VRAMP */
    {  118, -109,  -86,  -10, -109 }  /* CHECKER */
};

static const uint8_t PATTERN_EXPECTED_CLASS[PATTERN_NUM_VECTORS] = { 0, 2, 0 };

/* argmax: HRAMP -> computer, VRAMP -> people, CHECKER -> computer */
