# Project Remora: Sub-Symbolic Wave-Dynamics & Symbiotic Latent Inference

**Principal Investigator / Architect**: Gerald Corzo (`corzogac`)  
**Repository & Workspace**: `Dropbox\04-Work\Projects\Remora\`  
**Target Engines**: Colibri (pure C disk-streamed MoE), llama.cpp (CUDA/CPU hybrid)  
**Status**: Active Research & Prototype Implementation  

---

## 🌊 Executive Concept

Traditional Large Language Model inference treats interaction as a discrete, repeating **symbolic text loop**:
1. Human-readable text instructions (system prompts, personas, memory logs) are re-tokenized every session.
2. Billions of FLOPs are expended reprocessing invariant identity prompts into latent activations.
3. Long-term memory is serialized into text files or Markdown wikis.

**Project Remora** introduces a continuous, sub-symbolic alternative based on **Neural Wave Dynamics**:

```
                              [THE SHARK]
                       Heavy Transformer / MoE
                    (Matmuls, Knowledge Weights)
                                 │
                    Residual Activation Stream (h)
                                 │
                                 ▼
                             [REMORA]
                    Symbiotic Pilot-Fish Model
               (Learns Continuous Wave Dynamics)
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
   [Zero-Token Latent Injection]          [Predictive Hardware Stream]
  Bakes persona standing waves into       Prefetches MoE experts from NVMe
  layers; 0 tokenization overhead         prior to router gate evaluation
```

---

## 🏛️ Project Directory Structure

```
Projects/Remora/
├── 01_Architecture/                 # Theoretical whitepapers, mathematical foundations, system specs
│   ├── Remora-Architecture.md       # Core architecture specification
│   ├── Remora-Architecture.docx     # Sharing copy (Word)
│   └── Wave-Dynamics-Theory.md      # PDE, Neural ODE & Phase-Space derivations
├── 02_Remora-Core/                  # Pure C / CUDA runtime engine extensions
│   ├── remora_engine.h              # C API for Remora observer & prefetch hook
│   ├── remora_engine.c              # Ring-buffer phase tracker & Taylor extrapolator
│   ├── remora_latent_cache.c        # Zero-token frozen latent tensor loader
│   └── Makefile                     # Build harness for dynamic library (.dll / .so)
├── 03_Experiments/                  # Python simulation, PyTorch harness, and test traces
│   ├── remora_observer.py           # The "Pilot Fish" neural observer network
│   ├── latent_standing_wave.py      # Latent injection & zero-token prompt baseline test
│   ├── run_experiment_1.py          # Exp 1: Latent State Freezing vs Token Re-ingestion
│   ├── run_experiment_2.py          # Exp 2: Expert Route Prediction from Activation Wake
│   ├── data/                        # Sample activation traces
│   └── results/                     # Metric plots, wavelet scalograms, latency logs
└── README.md                        # This project manifest
```

---

## 🔬 Core Hypotheses & Experimental Objectives

### 1. Zero-Token Latent Injection (Standing Wave Invariance)
- **Premise**: Invariant system prompts create a fixed potential field $\mathcal{V}_{\text{persona}}(h)$.
- **Goal**: Freeze the steady-state layer pre-activations directly into memory tensors. When a user message arrives, inject the raw tensor $\mathbf{b}^{(l)}_{\text{persona}}$, saving 100% of the prompt token budget and KV-cache allocation.

### 2. The Remora Pilot-Fish (Sub-Symbolic Lifelong Learning)
- **Premise**: Personal communication rhythms and reasoning patterns manifest as continuous phase trajectories.
- **Goal**: A compact neural observer (GRU / S4 / MLP) continuously trains on intermediate activation vectors, evolving an organic numerical persona without storing text.

### 3. Predictive MoE Disk Streaming (Colibri Acceleration)
- **Premise**: The autoregressive feedback $\Delta t=1$ enforces continuous trajectory continuity.
- **Goal**: Predict the top-$k$ expert routing activations 1–2 layers in advance, issuing non-blocking OS disk prefetch requests (`posix_fadvise` / Windows Overlapped I/O) to eliminate NVMe read stalls.

---

## 🚀 Quick Start & Running Experiments

```bash
# 1. Run the Latent Standing Wave experiment (Zero-token prompt comparison)
python 03_Experiments/run_experiment_1.py

# 2. Run the Remora Observer training on activation wake
python 03_Experiments/run_experiment_2.py

# 3. Build the C engine extension for Colibri / llama.cpp
cd 02_Remora-Core && make
```
