#include "remora_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

void remora_tracker_init(remora_tracker_t* tracker, int num_layers, int dim) {
    if (!tracker) return;
    memset(tracker, 0, sizeof(remora_tracker_t));
    tracker->num_layers = (num_layers < REMORA_MAX_LAYERS) ? num_layers : REMORA_MAX_LAYERS;
    tracker->dim = (dim < REMORA_MAX_DIM) ? dim : REMORA_MAX_DIM;
    tracker->t_head = 0;
}

void remora_tracker_update(remora_tracker_t* tracker, int layer, const float* h_new) {
    if (!tracker || layer < 0 || layer >= tracker->num_layers || !h_new) return;

    int cur = tracker->t_head;
    int prev = (cur - 1 + REMORA_WINDOW_SIZE) % REMORA_WINDOW_SIZE;
    int pprev = (cur - 2 + REMORA_WINDOW_SIZE) % REMORA_WINDOW_SIZE;

    // 1. Commit new activation state
    memcpy(tracker->h[layer][cur], h_new, tracker->dim * sizeof(float));

    // 2. Compute first derivative: Velocity v = h(t) - h(t-1)
    for (int i = 0; i < tracker->dim; i++) {
        tracker->v[layer][i] = tracker->h[layer][cur][i] - tracker->h[layer][prev][i];
    }

    // 3. Compute second derivative: Acceleration a = v(t) - v(t-1)
    for (int i = 0; i < tracker->dim; i++) {
        float v_prev = tracker->h[layer][prev][i] - tracker->h[layer][pprev][i];
        tracker->a[layer][i] = tracker->v[layer][i] - v_prev;
    }

    // Advance head when top layer updates
    if (layer == tracker->num_layers - 1) {
        tracker->t_head = (tracker->t_head + 1) % REMORA_WINDOW_SIZE;
    }
}

void remora_extrapolate_state(const remora_tracker_t* tracker, int layer, float* out_h_est) {
    if (!tracker || !out_h_est || layer < 0 || layer >= tracker->num_layers) return;
    int cur = (tracker->t_head - 1 + REMORA_WINDOW_SIZE) % REMORA_WINDOW_SIZE;

    // 2nd Order Taylor Extrapolation: h(t+1) = h(t) + v(t) + 0.5 * a(t)
    for (int i = 0; i < tracker->dim; i++) {
        out_h_est[i] = tracker->h[layer][cur][i] 
                     + tracker->v[layer][i] 
                     + 0.5f * tracker->a[layer][i];
    }
}

int remora_predict_experts(const remora_predictor_t* predictor,
                           const remora_tracker_t* tracker,
                           int layer,
                           int* out_expert_ids) {
    if (!predictor || !tracker || !out_expert_ids || layer < 0 || layer >= tracker->num_layers) {
        return -1;
    }

    int n_exp = predictor->num_experts[layer];
    int top_k = (predictor->top_k < n_exp) ? predictor->top_k : n_exp;
    int dim = predictor->dim;

    float h_est[REMORA_MAX_DIM];
    remora_extrapolate_state(tracker, layer, h_est);

    // Compute router projection logits
    float* logits = (float*)malloc(n_exp * sizeof(float));
    int* idx = (int*)malloc(n_exp * sizeof(int));
    if (!logits || !idx) {
        if (logits) free(logits);
        if (idx) free(idx);
        return -1;
    }

    const float* W = &predictor->W_router[layer][0];
    for (int e = 0; e < n_exp; e++) {
        float sum = 0.0f;
        const float* row = &W[e * dim];
        for (int i = 0; i < dim; i++) {
            sum += row[i] * h_est[i];
        }
        logits[e] = sum;
        idx[e] = e;
    }

    // Top-K selection
    for (int i = 0; i < top_k; i++) {
        int max_pos = i;
        for (int j = i + 1; j < n_exp; j++) {
            if (logits[idx[j]] > logits[idx[max_pos]]) {
                max_pos = j;
            }
        }
        int temp = idx[i];
        idx[i] = idx[max_pos];
        idx[max_pos] = temp;
        out_expert_ids[i] = idx[i];
    }

    free(logits);
    free(idx);
    return top_k;
}

void remora_async_prefetch_expert(int layer, int expert_id) {
    // Non-blocking OS prefetch stub for integration into Colibri / llama.cpp
    // In Colibri, mapped expert files or memory ranges are flagged for asynchronous paging.
    (void)layer;
    (void)expert_id;
}
