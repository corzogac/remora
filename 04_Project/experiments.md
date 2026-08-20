# Remora Experiments — Detail, Proof & Operational Path

**Version**: 2.1 · **Date**: 2026-08-20 · **Status**: baseline infrastructure ready;
instrumented fork pending

---

## 1. Experiment-by-experiment detail

### Experiment A — Predictive MoE Prefetch (primary, pillar 3)

- **Runs**: llama-bench (standard, before/after) + traced chat workloads on
  LFM2.5-8B-A1B-Q4_K_M. Hard case: Qwen2.5-MoE-A2.7B (14B total → NVMe streaming
  dominates). Agentic traces exported from real Hermes/Janus sessions.
- **How**: `run_bench.py --baseline-dir <stock llama.cpp> --remora-dir <remora fork>`
  (see `benchmarks/`); trace recorder captures router logits, expert-fetch µs,
  activation taps per token.
- **Expected proof**: ≥75% top-1 router prediction on real routers (replaces the
  44.47% simulated Top-K overlap); ≥2× measured expert-stall reduction; ≥10%
  end-to-end tok/s gain on the MoE workloads.
- **Status**: baseline llama.cpp b10509 + model running on the office server
  (live API timing works — first smoke numbers recorded). The remora-llama fork
  (instrumentation) is the missing piece.

### Experiment B — Zero-Token Persona (pillar 1)

- **Runs**: fixed persona block (Janus persona + Hermes system prompt, 1–4k tok)
  under three variants: naive re-tokenization · llama.cpp KV-cache reuse · Remora
  frozen-latent injection.
- **Expected proof**: ≥0.99 output cosine vs baseline; ≥90% measured prefill-token
  savings; no task regression on a 100-item eval set.
- **Status**: not started (needs the fork's latent-injection hook).

### Experiment C — Observer / Sub-Symbolic Memory (pillar 2)

- **Runs**: offline GRU/S4 observer trained on recorded activation traces (Pegasus,
  PyTorch MPS) to predict next-token routing + residual dynamics. Online Hebbian
  updates only after offline top-1 ≥75% with drift monitoring.
- **Expected proof**: held-out top-1 ≥75% (real models); drift <1% per 1k tokens
  on the online variant.
- **Status**: not started (needs Exp A traces).

---

## 2. Theory certainty per pillar (what is proved vs believed)

| Pillar | Certainty now | Why | Path to proof |
|---|---|---|---|
| 3. MoE prefetch | MEDIUM | Real mechanism (NVMe stalls are real; router prediction is testable today) | Exp A on LFM2.5-8B-A1B / Qwen2.5-MoE |
| 1. Zero-token persona | LOW-MEDIUM | Production KV caching already captures most of the win; "standing wave" bias injection is unproven on real attention | Exp B + honest comparison vs KV reuse |
| 2. Sub-symbolic memory | LOW | Online activation-level memory risks drift; simulation-only evidence | Exp C offline first |

**Main certainty to make the system fully operational**: Pillar 3. It is the only
pillar with a mechanism that is unambiguously real (disk-streamed MoE I/O stalls)
and an effect that is directly measurable. Pillars 1–2 are research bets; pillar 3
is the deliverable. **Target model for public release**: LFM2.5-8B-A1B (the fleet's
local MoE) — a public remora-llama build + benchmark report on this model is the
releaseable artifact.

---

## 3. Operational vision — frozen-activation memory ("memory of nodes, not processing units")

The end-state the theory points to: a document/context is read ONCE; its activations
(standing waves) are frozen into a latent store; every later interaction injects the
frozen latents instead of re-tokenizing/re-processing. Cost then scales with memory
(storage of tensors) instead of compute (re-running matmuls) — i.e., a RAG built on
**frozen activation values** rather than text chunks.

Status check: this is the *aspiration*, not yet a system. What exists today that is
real and close: llama.cpp KV-cache reuse and provider prompt caching (Anthropic,
OpenAI, DeepSeek) already skip re-processing of prompt prefixes — the "5–10 minutes
of reading then freeze" is, today, approximated by prefix caching. Remora's research
edge is extending this from text-prefix caching to true latent-state injection and
cross-session persona memory. Exp B is the first concrete measurement of whether the
frozen-latent approach beats KV caching; until Exp B numbers exist, the operational
claim should be phrased as "cache-aware inference with frozen latent memory" and the
cost math (storage vs compute) validated with real VRAM/price figures.

---

## 4. Hardware allocation (2026-08)

| Machine | Spec | Role in the project |
|---|---|---|
| Office server (Janus, Win11 i9 32 GB) | RTX A2000 4 GB | PRIMARY benchmark host — llama.cpp baseline + remora-llama; self-hosted GitHub runner |
| Pegasus (M5 Mac) | Apple Silicon | Observer training (PyTorch MPS), trace analysis, paper editing; optional MLX inference of LFM |
| Poseidon (Win10 i7) | CPU-only | Secondary CPU-speed benchmarks (llama.cpp CPU build) for comparison data |
| HF Spaces (T4 GPU) | paid, ~€0.35/hr | Autonomous scheduled runs + public leaderboard (budget ~€10 ≈ 25+ h T4; keep runs < 2 h each) |

M5 Mac answer: nothing blocks you on the M5 — it is the analysis/training seat, not
the benchmark seat (the GPU lives on the office server). If you want a local Mac
fallback model, LFM2.5-8B-A1B has an MLX conversion path via mlx_lm — say the word
and I'll set it up.

---

## 5. Public-release criteria (for corzogac/remora going public / paper v2)

1. Exp A real-model numbers in `04_Project/results/` (baseline vs remora, same
   prompts/seeds/flags, mean ± std).
2. Abstract reconciled with the report (no "up to" numbers without trace).
3. `remora-llama` fork + trace tools published (or linked) with the paper.
4. HF dataset of traces for reproducibility (private first, public on release).

NOTE: the repo is currently PUBLIC on GitHub (verified 2026-08-20). If you want it
private until the release criteria are met, say so and I'll flip it via the API.
