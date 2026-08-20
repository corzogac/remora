# Remora V2 — Experiment Plan (Real-Model Validation)

**Version**: 2.0 · **Date**: 2026-08-20 · **Author**: Gerald Corzo (with Janus Deep & Pegasus)

---

## 0. Integrity rules (non-negotiable)

1. Every headline number in the paper must trace to a reproducible run described in
   `03_Experiments/results/` or `04_Project/results/` (script + seed + model + prompt set).
2. Metrics are reported as **mean ± std across ≥5 runs**, never "up to" a cherry-picked max.
3. Define metrics precisely: Top-K means what K? "Stall reduction" measured how (wall-clock
   I/O wait, or estimated)? Both definitions must be stated wherever a number appears.
4. **Reconcile the v1 mismatch**: abstract's 94.06% Top-K / 86.5% stall vs report's
   44.47% / 40.9%. V2 either reproduces a defensible number or rewrites the abstract.
5. No claim of "eliminating X% of tokens" without stating the comparison baseline
   (naive re-tokenization vs KV-cache reuse vs Remora injection).

---

## 1. Instrumentation (llama.cpp fork `remora-llama`)

Build from source with CUDA on the office server (`-DGGML_CUDA=ON`, pinned to a stable
llama.cpp tag; keep a rebase branch). Add compile-flag-gated hooks (`-DREMORA_TRACE`):

- **Router capture** (MoE layers): per token, log top-k expert ids + router scores per layer.
- **Expert load timing**: per expert fetch — source (RAM hit / NVMe read), latency µs.
- **Activation taps**: residual stream `h_l` and velocity `v_l = h_l(t) − h_l(t−1)` at
  selected layers → binary trace file (`npz`-compatible) or JSONL.
- **Prefetch hook**: call the V1 C engine (`02_Remora-Core/remora_engine.c`) at layer `l`
  to issue `posix_fadvise` (Linux) / OVERLAPPED `ReadFile` (Windows) for predicted
  experts at layer `l+1`.

Deliverable: a `remora-llama` branch + Windows build script + trace recorder.

## 2. Experiment A — MoE expert prefetch (primary)

- **Model**: LFM2.5-8B-A1B Q4_K_M (and Qwen2.5-MoE-A2.7B as the hard case).
- **Workloads**: (i) agentic multi-turn tool-calling traces (export real Hermes/Janus
  sessions via `hermes sessions export`); (ii) long-context 8k–16k prompts.
- **Baseline**: stock llama.cpp server, same prompts/seeds.
- **Remora variant**: Taylor 2nd-order predictor (V1 C engine) + learned observer (V2).
- **Metrics**:
  - Router prediction: Top-1 / Top-5 accuracy per layer and per token. **Target ≥75% Top-1.**
  - Expert stall: mean/p95 expert-fetch latency µs, baseline vs prefetched. **Target ≥2× reduction.**
  - End-to-end: tokens/sec, TTFT, VRAM peak.
- **Output**: `04_Project/results/expA_*.json`, plots, `expA_report.md`.

## 3. Experiment B — Zero-token persona (standing wave)

- **Setup**: fixed persona block (Janus persona + Hermes system prompt ≈ 1–4k tokens).
- **Variants**: (i) naive re-tokenization each turn; (ii) llama.cpp KV-cache reuse;
  (iii) Remora frozen-latent bias injection (V1 `latent_standing_wave` on real model).
- **Metrics**: output cosine similarity vs (i); downstream task score on a small eval set
  (e.g., 100 QA items); measured prefill token savings; VRAM delta.
- **Success**: ≥0.99 cosine vs baseline, ≥90% measured prefill-token savings, no task regression.
- **Output**: `04_Project/results/expB_*`.

## 4. Experiment C — Observer / sub-symbolic memory

- **Offline (must-pass)**: train GRU/S4 observer on recorded activation traces (Pegasus,
  PyTorch) to predict next-token expert routing + residual dynamics. Report held-out Top-1.
  This replaces the 44.47% simulation number with a real-model number.
- **Online (stretch)**: Hebbian delta updates on live traces with drift monitoring
  (divergence from a frozen baseline on a fixed eval set; alarm if >1% per 1k tokens).

## 5. Colibri pilot (phase 2)

- Environment: WSL2 on the office server or AIWaterServer1 (Ubuntu).
- Goal: reproduce Exp A's stall reduction on Colibri's native disk-streamed MoE path.

## 6. Deliverables & timeline (2–3 weeks, part-time)

| Week | Deliverable |
|---|---|
| W1 | `remora-llama` fork + trace recorder; baseline Exp A runs (LFM2.5-8B-A1B) |
| W2 | Exp A prefetch numbers; Exp B setup + run |
| W3 | Exp B/C; Colibri pilot; paper v2 rewrite with real numbers + figures |

## 7. Risks

- **4 GB VRAM**: partial offload; A1B MoE keeps speed; Qwen2.5-MoE exercises NVMe path.
- **llama.cpp churn**: pin tag; fork rebased only when needed.
- **Real accuracy below simulation**: add router-logit + hidden-state features to the
  predictor (V1 used only h extrapolation); learned observer as fallback.
- **Windows toolchain**: build in WSL2 if MSVC fights back; benchmark scripts cross-platform.
