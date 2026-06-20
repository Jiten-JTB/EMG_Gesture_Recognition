#include "gesture_fsm.h"
#include "config.h"

// ── States ────────────────────────────────────────────────────────────────────
typedef enum {
    FSM_REST = 0,
    FSM_PINCH_ONSET,    // pinch detected, timer running
    FSM_PINCH_HELD,     // pinch held past HOLD_THRESHOLD_MS
    FSM_FIST_ONSET,     // fist detected, timer running
    FSM_FIST_HELD,      // fist held past HOLD_THRESHOLD_MS
} fsm_state_t;

// Each CNN inference fires every STEP_SIZE samples = 100ms
#define MS_PER_STEP     100
#define HOLD_STEPS      (HOLD_THRESHOLD_MS / MS_PER_STEP)

static fsm_state_t s_state     = FSM_REST;
static int         s_hold_steps = 0;  // how many consecutive steps in gesture

void gesture_fsm_init(void)
{
    s_state      = FSM_REST;
    s_hold_steps = 0;
}

hid_action_t gesture_fsm_update(int cnn_class)
{
    switch (s_state) {

    // ── REST ──────────────────────────────────────────────────────────────────
    case FSM_REST:
        if (cnn_class == CLASS_PINCH) {
            s_state      = FSM_PINCH_ONSET;
            s_hold_steps = 1;
        } else if (cnn_class == CLASS_FIST) {
            s_state      = FSM_FIST_ONSET;
            s_hold_steps = 1;
        }
        return HID_ACTION_NONE;

    // ── PINCH ONSET ───────────────────────────────────────────────────────────
    case FSM_PINCH_ONSET:
        if (cnn_class == CLASS_PINCH) {
            s_hold_steps++;
            if (s_hold_steps >= HOLD_STEPS) {
                s_state = FSM_PINCH_HELD;
                return HID_ACTION_LEFT_DOWN;  // enter hold — mouse button down
            }
            return HID_ACTION_NONE;
        } else {
            // Released before threshold — it was a quick pinch
            s_state      = FSM_REST;
            s_hold_steps = 0;
            return HID_ACTION_LEFT_CLICK;
        }

    // ── PINCH HELD ────────────────────────────────────────────────────────────
    case FSM_PINCH_HELD:
        if (cnn_class != CLASS_PINCH) {
            s_state      = FSM_REST;
            s_hold_steps = 0;
            return HID_ACTION_LEFT_UP;   // released — mouse button up
        }
        return HID_ACTION_NONE;   // still held — cursor/volume handled in main.c

    // ── FIST ONSET ────────────────────────────────────────────────────────────
    case FSM_FIST_ONSET:
        if (cnn_class == CLASS_FIST) {
            s_hold_steps++;
            if (s_hold_steps >= HOLD_STEPS) {
                s_state = FSM_FIST_HELD;
                return HID_ACTION_FIST_HOLD_START;  // cursor movement enabled
            }
            return HID_ACTION_NONE;
        } else {
            // Released before threshold — quick fist = right click
            s_state      = FSM_REST;
            s_hold_steps = 0;
            return HID_ACTION_RIGHT_CLICK;
        }

    // ── FIST HELD ─────────────────────────────────────────────────────────────
    case FSM_FIST_HELD:
        if (cnn_class != CLASS_FIST) {
            s_state      = FSM_REST;
            s_hold_steps = 0;
            return HID_ACTION_FIST_HOLD_END;  // cursor movement disabled
        }
        return HID_ACTION_NONE;   // still held — cursor driven by IMU in main.c

    default:
        s_state = FSM_REST;
        return HID_ACTION_NONE;
    }
}

bool gesture_fsm_pinch_held(void) { return s_state == FSM_PINCH_HELD; }
bool gesture_fsm_fist_held(void)  { return s_state == FSM_FIST_HELD;  }