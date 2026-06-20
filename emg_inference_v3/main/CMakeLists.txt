idf_component_register(
    SRCS
        "main.c"
        "emg_filter.c"
        "cnn_inference.c"
        "gesture_fsm.c"
    INCLUDE_DIRS "."
    REQUIRES
        esp_adc
        driver
        esp_timer
        esp_system
)