#pragma once
#include <stdint.h>

// 4th order IIR bandpass filter (20–120 Hz)
// Implemented as Direct Form II Transposed — numerically stable, 4 state vars.
// Coefficients are loaded from model_weights_v3.h (FILTER_B, FILTER_A).
//
// This is the SAME filter applied in Python during training (scipy filtfilt
// uses zero-phase; here we use single-pass causal which adds ~8ms delay —
// acceptable for real-time use).

typedef struct {
    float w[4];   // delay-line state
} emg_filter_t;

void  emg_filter_init(emg_filter_t *f);
float emg_filter_sample(emg_filter_t *f, float x);
