#ifndef REMORA_ENGINE_H
#define REMORA_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define REMORA_MAX_LAYERS 64
#define REMORA_WINDOW_SIZE 4
#define REMORA_MAX_EXPERTS 64
#define REMORA_MAX_DIM 8192

/**
 * @brief Remora Phase State Tracker
 * Tracks residual activations, velocity (1st derivative), and acceleration (2nd derivative).
 */
typedef struct {
    float h[REMORA_MAX_LAYERS][REMORA_WINDOW_SIZE][REMORA_MAX_DIM];
    float v[REMORA_MAX_LAYERS][REMORA_MAX_DIM];
    float a[REMORA_MAX_LAYERS][REMORA_MAX_DIM];
    int t_head;
    int num_layers;
    int dim;
} remora_tracker_t;

/**
 * @brief Remora Predictive Prefetch Engine
 * Extrapolates activation waves and forecasts upcoming MoE routing targets.
 */
typedef struct {
    float W_router[REMORA_MAX_LAYERS][REMORA_MAX_EXPERTS * REMORA_MAX_DIM];
    int num_experts[REMORA_MAX_LAYERS];
    int top_k;
    int dim;
} remora_predictor_t;

/**
 * @brief Initialize Remora Tracker
 */
void remora_tracker_init(remora_tracker_t* tracker, int num_layers, int dim);

/**
 * @brief Update Tracker with fresh layer activation
 */
void remora_tracker_update(remora_tracker_t* tracker, int layer, const float* h_new);

/**
 * @brief Extrapolate next token state via 2nd-order Taylor expansion
 */
void remora_extrapolate_state(const remora_tracker_t* tracker, int layer, float* out_h_est);

/**
 * @brief Forecast Top-K MoE Expert Targets for Layer l
 */
int remora_predict_experts(const remora_predictor_t* predictor,
                           const remora_tracker_t* tracker,
                           int layer,
                           int* out_expert_ids);

/**
 * @brief Non-blocking prefetch hook for disk-streamed MoE (Colibri integration)
 */
void remora_async_prefetch_expert(int layer, int expert_id);

#ifdef __cplusplus
}
#endif

#endif /* REMORA_ENGINE_H */
