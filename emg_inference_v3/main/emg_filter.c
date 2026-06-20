#include "emg_filter.h"
#include "model_weights_v3.h"
#include <string.h>

// Direct Form II Transposed IIR filter.
// For a 4th order filter with coefficients b[0..4] and a[0..4] (a[0]=1):
//
//   y[n]  = b[0]*x[n] + w[0]
//   w[0]  = b[1]*x[n] - a[1]*y[n] + w[1]
//   w[1]  = b[2]*x[n] - a[2]*y[n] + w[2]
//   w[2]  = b[3]*x[n] - a[3]*y[n] + w[3]
//   w[3]  = b[4]*x[n] - a[4]*y[n]
//
// FILTER_B and FILTER_A are declared in model_weights_v3.h

void emg_filter_init(emg_filter_t *f)
{
    memset(f->w, 0, sizeof(f->w));
}

float emg_filter_sample(emg_filter_t *f, float x)
{
    float y   = FILTER_B[0] * x + f->w[0];
    f->w[0]   = FILTER_B[1] * x - FILTER_A[1] * y + f->w[1];
    f->w[1]   = FILTER_B[2] * x - FILTER_A[2] * y + f->w[2];
    f->w[2]   = FILTER_B[3] * x - FILTER_A[3] * y + f->w[3];
    f->w[3]   = FILTER_B[4] * x - FILTER_A[4] * y;
    return y;
}
