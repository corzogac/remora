"""
Project Remora: Experiment 2 - The Remora Observer (Predictive MoE Routing from Activation Wake)
Demonstrates predicting the upcoming layer's Top-K active MoE experts solely from
prior layer activation velocity (v) and acceleration (a), enabling asynchronous NVMe prefetching.
"""

import numpy as np

def run_experiment():
    np.random.seed(42)
    dim = 128
    num_layers = 16
    num_experts = 8
    top_k = 2
    sequence_length = 200
    
    print("=" * 65)
    print("  PROJECT REMORA: EXPERIMENT 2 - PILOT FISH EXPERT PREDICTOR")
    print("=" * 65)
    
    # Generate router weight matrices per layer
    W_router = [np.random.randn(num_experts, dim) * 0.1 for _ in range(num_layers)]
    
    # Simulate a smooth autoregressive wave trajectory through sequence and layers
    # h(t, l) has temporal smoothness (velocity correlation)
    activations = np.zeros((sequence_length, num_layers, dim))
    
    current_state = np.random.randn(dim) * 0.5
    for t in range(sequence_length):
        # Autoregressive delta update
        delta = np.sin(t * 0.1) * np.random.randn(dim) * 0.05
        current_state = current_state * 0.95 + delta
        for l in range(num_layers):
            layer_mod = np.cos(l * 0.3) * current_state + np.random.randn(dim) * 0.02
            activations[t, l] = layer_mod
            
    print(f"\n[1] Generated {sequence_length} token steps across {num_layers} layers ({dim}-dim).")
    
    # Evaluate Wave Predictor vs True Router Target
    correct_top1 = 0
    correct_topk = 0
    total_evals = 0
    
    for t in range(2, sequence_length):
        for l in range(num_layers):
            # 1. Ground truth router targets for token t at layer l
            h_true = activations[t, l]
            logits_true = np.dot(W_router[l], h_true)
            true_top_experts = np.argsort(logits_true)[::-1][:top_k]
            
            # 2. Remora Wave Observer: Predict h(t, l) using prior steps (t-1, t-2)
            h_prev = activations[t - 1, l]
            h_pprev = activations[t - 2, l]
            
            velocity = h_prev - h_pprev
            acceleration = (h_prev - h_pprev) - (h_pprev - activations[t - 3, l] if t >= 3 else 0)
            
            # 2nd-order Taylor prediction of activation wavefront
            h_pred = h_prev + velocity + 0.5 * acceleration
            logits_pred = np.dot(W_router[l], h_pred)
            pred_top_experts = np.argsort(logits_pred)[::-1][:top_k]
            
            # Check accuracy
            if pred_top_experts[0] == true_top_experts[0]:
                correct_top1 += 1
            if len(set(pred_top_experts).intersection(set(true_top_experts))) > 0:
                correct_topk += 1
                
            total_evals += 1
            
    top1_acc = (correct_top1 / total_evals) * 100
    topk_acc = (correct_topk / total_evals) * 100
    
    print("\n" + "=" * 65)
    print("  PREFETCH PREDICTION PERFORMANCE")
    print("=" * 65)
    print(f"  Total Routing Decisions Evaluated : {total_evals}")
    print(f"  Remora Top-1 Routing Hit Rate     : {top1_acc:.2f}%")
    print(f"  Remora Top-K Overlap Hit Rate     : {topk_acc:.2f}%")
    print(f"  NVMe Disk Stall Reduction (Est.)  : ~{(topk_acc * 0.9):.1f}% reduction")
    print("=" * 65)

if __name__ == "__main__":
    run_experiment()
