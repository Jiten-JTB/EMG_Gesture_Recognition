#pragma once
#include "model_weights_v3.h"

// Run the trained 1D CNN on a 50-sample filtered EMG window.
//
// window   : float[WINDOW_SIZE] — filtered EMG samples (millivolts)
// probs    : float[NUM_CLASSES] — softmax probabilities output, caller-allocated
//
// Returns the argmax class index (0=rest, 1=quick_pinch, 2=quick_fist).
// If the max probability is below CONF_THRESHOLD, returns CLASS_REST instead.

int cnn_predict(const float *window, float *probs);
