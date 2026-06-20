#include "cnn_inference.h"
#include "config.h"
#include <math.h>
#include <string.h>
#include <float.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Verified tensor shapes (computed from model architecture):
//
//  Input            : (1,  50)   — 1 channel, 50 time steps
//  Conv1 (k=5,p=2)  : (32, 50)  — 32 filters, same-padded
//  MaxPool(2)        : (32, 25)
//  Conv2 (k=3,p=1)  : (64, 25)  — 64 filters, same-padded
//  MaxPool(2)        : (64, 12)  — floor(25/2) = 12
//  Conv3 (k=3,p=1)  : (64, 12)  — 64 filters, no pool
//  GlobalAvgPool     : (64,)
//  FC1              : (32,)
//  FC2              : (3,)       — logits → softmax → class
// ─────────────────────────────────────────────────────────────────────────────

// Static intermediate buffers — allocated once, reused every inference call.
// Largest is conv1_out / conv2_out at 32×50 = 64×25 = 1600 floats = 6.25 KB.
// All fit comfortably in internal SRAM; no PSRAM needed for inference.

static float s_conv1_out[CONV1_OUT_CH * 50];   // (32, 50)
static float s_pool1_out[CONV1_OUT_CH * 25];   // (32, 25)
static float s_conv2_out[CONV2_OUT_CH * 25];   // (64, 25)
static float s_pool2_out[CONV2_OUT_CH * 12];   // (64, 12)
static float s_conv3_out[CONV3_OUT_CH * 12];   // (64, 12)
static float s_gap_out  [CONV3_OUT_CH];         // (64,)
static float s_fc1_out  [32];
static float s_fc2_out  [NUM_CLASSES];

// ─────────────────────────────────────────────────────────────────────────────
//  Primitive operations
// ─────────────────────────────────────────────────────────────────────────────

// 1D convolution with same-padding and ReLU activation.
// Weights layout: W[out_ch][in_ch][kernel] — row-major, matching PyTorch default.
// Same-padding: output length == input length.
//
// out[oc][t] = ReLU( sum_{ic,k} W[oc][ic][k] * in[ic][t - half + k] + B[oc] )
//
// Boundary samples (outside the input) are treated as zero (zero-padding).

static void conv1d_relu_same(
    const float *in,   int in_ch,  int in_len,
    const float *W,
    const float *B,
          float *out,  int out_ch, int kernel)
{
    int half = kernel / 2;

    for (int oc = 0; oc < out_ch; oc++) {
        const float *W_oc = W + oc * (in_ch * kernel);
        float bias = B[oc];

        for (int t = 0; t < in_len; t++) {
            float acc = bias;

            for (int ic = 0; ic < in_ch; ic++) {
                const float *W_ic = W_oc + ic * kernel;
                const float *in_ic = in + ic * in_len;

                for (int k = 0; k < kernel; k++) {
                    int src = t - half + k;
                    if (src >= 0 && src < in_len) {
                        acc += W_ic[k] * in_ic[src];
                    }
                    // src out of bounds → zero-padding, contribution = 0
                }
            }

            // ReLU activation — BN was folded into weights during export
            out[oc * in_len + t] = acc > 0.0f ? acc : 0.0f;
        }
    }
}

// Max pooling with stride = pool_size, floor division on output length.
// out_len = in_len / pool_size  (integer division)

static int maxpool1d(
    const float *in,  int ch, int in_len,
          float *out, int pool_size)
{
    int out_len = in_len / pool_size;

    for (int c = 0; c < ch; c++) {
        const float *in_c  = in  + c * in_len;
              float *out_c = out + c * out_len;

        for (int t = 0; t < out_len; t++) {
            float mx = -FLT_MAX;
            for (int p = 0; p < pool_size; p++) {
                float v = in_c[t * pool_size + p];
                if (v > mx) mx = v;
            }
            out_c[t] = mx;
        }
    }
    return out_len;
}

// Global Average Pooling: average each channel over the time axis.
// (ch, length) → (ch,)

static void global_avg_pool(
    const float *in, int ch, int length,
          float *out)
{
    for (int c = 0; c < ch; c++) {
        float sum = 0.0f;
        const float *in_c = in + c * length;
        for (int t = 0; t < length; t++) sum += in_c[t];
        out[c] = sum / (float)length;
    }
}

// Fully-connected layer with optional ReLU.
// out[i] = ReLU( sum_j W[i*in_dim + j] * in[j] + B[i] )

static void linear_relu(
    const float *W, const float *B,
    const float *in,  int in_dim,
          float *out, int out_dim,
          int use_relu)
{
    for (int i = 0; i < out_dim; i++) {
        float acc = B[i];
        const float *Wi = W + i * in_dim;
        for (int j = 0; j < in_dim; j++) acc += Wi[j] * in[j];
        out[i] = (use_relu && acc < 0.0f) ? 0.0f : acc;
    }
}

// Softmax: converts raw logits to probabilities that sum to 1.
// Subtract max before exp for numerical stability.

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Full forward pass
// ─────────────────────────────────────────────────────────────────────────────

int cnn_predict(const float *window, float *probs)
{
    // ── Conv block 1: (1, 50) → (32, 50) → pool → (32, 25) ─────────────────
    conv1d_relu_same(
        window,      CONV1_IN_CH,  WINDOW_SIZE,
        CONV1_W, CONV1_B,
        s_conv1_out, CONV1_OUT_CH, CONV1_KERNEL);

    maxpool1d(s_conv1_out, CONV1_OUT_CH, WINDOW_SIZE, s_pool1_out, CONV1_POOL);
    // pool1_out: (32, 25)

    // ── Conv block 2: (32, 25) → (64, 25) → pool → (64, 12) ────────────────
    conv1d_relu_same(
        s_pool1_out, CONV2_IN_CH,  25,
        CONV2_W, CONV2_B,
        s_conv2_out, CONV2_OUT_CH, CONV2_KERNEL);

    maxpool1d(s_conv2_out, CONV2_OUT_CH, 25, s_pool2_out, CONV2_POOL);
    // pool2_out: (64, 12)

    // ── Conv block 3: (64, 12) → (64, 12), no pool ──────────────────────────
    conv1d_relu_same(
        s_pool2_out, CONV3_IN_CH,  12,
        CONV3_W, CONV3_B,
        s_conv3_out, CONV3_OUT_CH, CONV3_KERNEL);
    // conv3_out: (64, 12)

    // ── Global Average Pool: (64, 12) → (64,) ───────────────────────────────
    global_avg_pool(s_conv3_out, CONV3_OUT_CH, 12, s_gap_out);

    // ── FC1: (64,) → (32,) + ReLU ───────────────────────────────────────────
    linear_relu(FC1_W, FC1_B, s_gap_out, 64, s_fc1_out, 32, /*relu=*/1);

    // ── FC2: (32,) → (3,) — no activation, softmax follows ──────────────────
    linear_relu(FC2_W, FC2_B, s_fc1_out, 32, s_fc2_out, NUM_CLASSES, /*relu=*/0);

    // ── Softmax → probabilities ──────────────────────────────────────────────
    memcpy(probs, s_fc2_out, NUM_CLASSES * sizeof(float));
    softmax(probs, NUM_CLASSES);

    // ── Argmax → predicted class ─────────────────────────────────────────────
    int best = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > probs[best]) best = i;
    }

    // Apply confidence threshold — ambiguous predictions → REST
    // This is safer than outputting a wrong gesture action
    if (probs[best] < CONF_THRESHOLD) best = CLASS_REST;

    return best;
}
