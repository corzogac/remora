# Remora V2 — Real-Model Validation Project

**Status**: Active (started 2026-08-20)
**PI**: Gerald Corzo (corzogac)
**Team**: Janus Deep (office server, experiment runs), Pegasus (analysis + orchestration)
**Repo**: https://github.com/corzogac/remora

---

## Why V2

V1 delivered the theory, the C engine prototype, and simulation-based validation.
Two gaps block the arXiv draft from being honest and publishable:

1. **Simulation-only numbers.** All headline metrics come from synthetic random-weight
   transformers (dim 256, 12 layers), not real models.
2. **Abstract/report mismatch.** The paper abstract claims up to **94.06% Top-K overlap**
   and **~86.5% stall reduction**; the repo's own experiment report shows **44.47% Top-K
   overlap** and **~40.9% estimated stall reduction**. The paper must not carry numbers
   its own validation report contradicts.

V2's single goal: **replace every headline number with reproducible real-model
measurements**, run the full ablation suite on real hardware, and update the paper.

## Hardware / host matrix

| Host | Role | Hardware |
|---|---|---|
| Office server (un-ihe, 100.120.94.49, Win11) | Primary benchmark host | RTX A2000 Laptop 4 GB, llama.cpp (CUDA), NVMe |
| Pegasus (gerald-air, macOS ARM) | Analysis, observer training, paper | Apple Silicon |
| WSL2 / AIWaterServer1 (Ubuntu) | Colibri pilot (pillar 3 native) | CPU/NVMe, later |

## Models

- **Primary real MoE model**: `LiquidAI/LFM2.5-8B-A1B-GGUF` (Q4_K_M, ~4.7 GB) —
  1B-active MoE; expert streaming is a real bottleneck on the 4 GB card. Doubles as the
  fleet's local backup model (see `fleet-update_2026-08-20`).
- **Harder disk-streaming case**: `Qwen2.5-MoE-A2.7B` (14B total) — weights exceed VRAM;
  NVMe streaming dominates; strongest test of the prefetch pillar.
- Dense control: `Qwen3-4B` (tool-calling reference).

See `Remora-V2-Experiment-Plan.md` for the full protocol.
