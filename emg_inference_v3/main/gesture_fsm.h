#pragma once
#include <stdbool.h>

// Gesture finite state machine.
// Takes CNN class output every 100ms and resolves quick vs hold.

typedef enum {
    HID_ACTION_NONE = 0,
    HID_ACTION_LEFT_CLICK,      // quick pinch released
    HID_ACTION_LEFT_DOWN,       // pinch held >= HOLD_THRESHOLD_MS
    HID_ACTION_LEFT_UP,         // pinch hold released
    HID_ACTION_RIGHT_CLICK,     // quick fist released
    HID_ACTION_FIST_HOLD_START, // fist held >= HOLD_THRESHOLD_MS
    HID_ACTION_FIST_HOLD_END,   // fist hold released
} hid_action_t;

void         gesture_fsm_init(void);
hid_action_t gesture_fsm_update(int cnn_class);
bool         gesture_fsm_pinch_held(void);  // true while in pinch hold state
bool         gesture_fsm_fist_held(void);   // true while in fist hold state