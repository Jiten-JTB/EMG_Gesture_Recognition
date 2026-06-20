#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

#include "config.h"
#include "emg_filter.h"
#include "cnn_inference.h"
#include "model_weights_v3.h"
#include "gesture_fsm.h"

#define TAG "emg_test"

// ── ADC ───────────────────────────────────────────────────────────────────────
static adc_oneshot_unit_handle_t s_adc;
static adc_cali_handle_t         s_cali;
static bool                      s_cali_ok;

static void adc_init(void)
{
    adc_oneshot_unit_init_cfg_t unit_cfg = { .unit_id = EMG_ADC_UNIT };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &s_adc));

    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten    = EMG_ADC_ATTEN,
        .bitwidth = EMG_ADC_BITWIDTH,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(s_adc, EMG_ADC_CHANNEL, &chan_cfg));

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cali_cfg = {
        .unit_id  = EMG_ADC_UNIT, .chan     = EMG_ADC_CHANNEL,
        .atten    = EMG_ADC_ATTEN,.bitwidth = EMG_ADC_BITWIDTH,
    };
    s_cali_ok = (adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_cali) == ESP_OK);
#elif ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
    adc_cali_line_fitting_config_t cali_cfg = {
        .unit_id  = EMG_ADC_UNIT, .atten    = EMG_ADC_ATTEN,
        .bitwidth = EMG_ADC_BITWIDTH,
    };
    s_cali_ok = (adc_cali_create_scheme_line_fitting(&cali_cfg, &s_cali) == ESP_OK);
#else
    s_cali_ok = false;
#endif
    ESP_LOGI(TAG, "ADC calibration: %s", s_cali_ok ? "OK" : "none");
}

static float emg_read_volt(void)
{
    int raw = 0;
    adc_oneshot_read(s_adc, EMG_ADC_CHANNEL, &raw);
    if (s_cali_ok) {
        int mv = 0;
        adc_cali_raw_to_voltage(s_cali, raw, &mv);
        return (float)mv / 1000.0f;
    }
    return ((float)raw / 4095.0f) * 3.3f;
}

// ── EMG pipeline ──────────────────────────────────────────────────────────────
static emg_filter_t s_filter;
static float        s_window[WINDOW_SIZE];
static int          s_head         = 0;
static int          s_sample_count = 0;
static int          s_step_counter = 0;

// ── FSM action name ───────────────────────────────────────────────────────────
static const char *action_name(hid_action_t action)
{
    switch (action) {
    case HID_ACTION_LEFT_CLICK:      return "LEFT CLICK     (quick pinch)";
    case HID_ACTION_LEFT_DOWN:       return "PINCH HOLD START";
    case HID_ACTION_LEFT_UP:         return "PINCH HOLD END";
    case HID_ACTION_RIGHT_CLICK:     return "RIGHT CLICK    (quick fist)";
    case HID_ACTION_FIST_HOLD_START: return "FIST HOLD START";
    case HID_ACTION_FIST_HOLD_END:   return "FIST HOLD END";
    default:                         return NULL;
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────
static void run_inference(void)
{
    esp_task_wdt_reset();

    float linear_win[WINDOW_SIZE];
    for (int i = 0; i < WINDOW_SIZE; i++)
        linear_win[i] = s_window[(s_head + i) % WINDOW_SIZE];

    float probs[NUM_CLASSES];
    int   cls = cnn_predict(linear_win, probs);

    hid_action_t action = gesture_fsm_update(cls);

    // Raw CNN output every 100ms
    ESP_LOGI(TAG, "%-6s conf=%.2f | rest=%.2f pinch=%.2f fist=%.2f",
            cls == CLASS_REST  ? "REST"  :
            cls == CLASS_PINCH ? "PINCH" : "FIST",
            probs[cls],
            probs[CLASS_REST],
            probs[CLASS_PINCH],
            probs[CLASS_FIST]);

    // FSM event — only printed on state transitions
    const char *name = action_name(action);
    if (name) {
        ESP_LOGI(TAG, ">>> %s", name);
    }

    // Current hold state
    if (gesture_fsm_pinch_held()) ESP_LOGI(TAG, "    [pinch held]");
    if (gesture_fsm_fist_held())  ESP_LOGI(TAG, "    [fist held]");
}

// ── Entry point ───────────────────────────────────────────────────────────────
void app_main(void)
{
    ESP_LOGI(TAG, "=== EMG Gesture Test ===");
    ESP_LOGI(TAG, "Hold threshold: %d ms  |  Conf threshold: %.2f",
             HOLD_THRESHOLD_MS, CONF_THRESHOLD);
    ESP_LOGI(TAG, "Quick pinch  -> LEFT CLICK");
    ESP_LOGI(TAG, "Pinch hold   -> PINCH HOLD START / END");
    ESP_LOGI(TAG, "Quick fist   -> RIGHT CLICK");
    ESP_LOGI(TAG, "Fist hold    -> FIST HOLD START / END");
    ESP_LOGI(TAG, "------------------------");

    adc_init();
    emg_filter_init(&s_filter);
    memset(s_window, 0, sizeof(s_window));
    gesture_fsm_init();
    esp_task_wdt_add(NULL);

    ESP_LOGI(TAG, "Ready");

    int64_t next_sample_us = esp_timer_get_time();

    while (true) {
        while (esp_timer_get_time() < next_sample_us) {}
        next_sample_us += SAMPLE_INTERVAL_US;

        esp_task_wdt_reset();

        float filtered = emg_filter_sample(&s_filter, emg_read_volt());

        s_window[s_head] = filtered;
        s_head = (s_head + 1) % WINDOW_SIZE;
        s_sample_count++;

        if (s_sample_count < WINDOW_SIZE) continue;

        if (++s_step_counter >= STEP_SIZE) {
            s_step_counter = 0;
            run_inference();
        }
    }
}