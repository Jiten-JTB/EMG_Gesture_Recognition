#pragma once

// ── ADC — Bioamp Candy on GPIO 1 ─────────────────────────────────────────────
#define EMG_ADC_UNIT        ADC_UNIT_1
#define EMG_ADC_CHANNEL     ADC_CHANNEL_0   // GPIO 1
#define EMG_ADC_ATTEN       ADC_ATTEN_DB_12
#define EMG_ADC_BITWIDTH    ADC_BITWIDTH_12

// ── IMU — MPU-6050 on I2C ────────────────────────────────────────────────────
#define IMU_SDA_GPIO        GPIO_NUM_5
#define IMU_SCL_GPIO        GPIO_NUM_6
#define IMU_I2C_PORT        I2C_NUM_0
#define IMU_I2C_FREQ_HZ     400000          // 400 kHz fast mode
#define IMU_I2C_ADDR        0x68            // AD0 pin LOW

// ── Sampling ──────────────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ      250
#define SAMPLE_INTERVAL_US  (1000000 / SAMPLE_RATE_HZ)

// ── Inference ─────────────────────────────────────────────────────────────────
#define STEP_SIZE           25              // infer every 100ms
#define CONF_THRESHOLD      0.75f

// ── Gesture class IDs ─────────────────────────────────────────────────────────
// Same model as v3 — class 1 was quick_pinch, class 2 was quick_fist.
// We rename them here. The weights are identical.
#define CLASS_REST          0
#define CLASS_PINCH         1
#define CLASS_FIST          2

// ── Gesture state machine ─────────────────────────────────────────────────────
// How long a gesture must be held before it becomes a HOLD action.
// Below this: quick gesture (click). At or above: hold gesture (drag/volume).
#define HOLD_THRESHOLD_MS   400

// ── IMU cursor and volume ─────────────────────────────────────────────────────
// Gyro sensitivity at ±250 dps full scale = 131 LSB per deg/s
#define GYRO_SENSITIVITY    131.0f

// Scale gyro angular velocity to cursor pixel delta per inference step.
// Tune this — higher = faster cursor.
#define CURSOR_SCALE        0.12f

// Minimum gyro rate (deg/s) to register as movement — suppresses drift.
#define GYRO_DEADZONE       1.5f

// Minimum gyro rate on wrist rotation axis to register as volume gesture.
#define VOLUME_DEADZONE     8.0f

// ── BLE HID ───────────────────────────────────────────────────────────────────
#define BLE_DEVICE_NAME     "EMG Mouse"