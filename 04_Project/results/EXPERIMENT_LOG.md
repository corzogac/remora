# Remora — Experiment Log

## 2026-08-20 — FIRST REAL TRACE (office A2000, LFM2.5-8B-A1B Q4_K_M)

Source: `results/trace_lfm_office_20260820.jsonl` (58,344 lines, 4.5 MB)
Engine: llama.cpp b10510 + remora-trace hooks, REMORA_TRACE_FILE, 99 tokens,
prompt "Write a haiku about water." (16 tokens), single run.

Measured facts (analyze_trace.py):

| metric | value | meaning |
|---|---|---|
| layers / experts | 22 / 64 | LFM2.5-8B-A1B |
| router top-1 prob (mean) | 0.310 | 20x over uniform (0.016) |
| router entropy (range) | 2.3–2.9 | moderate concentration |
| distinct experts fired | 32 of 64 | HALF the network never fires |
| top-4 expert share | 16.3% of selections | hot path exists |
| top-1 weight share | 67.8% of routed mass | knowledge concentrated in one expert |
| moe op cost layers 2–18 | 24–46 us mean | fast, uniform |
| moe op cost layers 19–23 | 274–380 us mean, p95 2–2.8 ms | **10x slower tail** (LFM2 recr layers) |
| traced run throughput | 13.6 tok/s | vs 17.7 untraced baseline (trace overhead ~23%) |

Prefetch simulation — naive persistence predictor (next token = same top-k set):
exact-set hit rate **4.0%** mean, Jaccard 0.19–0.48 (mean ~0.32).

Interpretation:
1. The router IS a path: 32 experts carry the conversation, top-1 holds 68% of
   the mass. PowerInfer-style hot/cold split is real on this model.
2. The last 5 layers (LFM2 recurrent layers) cost 10x the others — the stall
   surface is NOT uniform; prefetch/warm-up targeting those layers has the
   biggest ceiling.
3. Naive next-token expert prediction fails (4% exact) but overlaps ~32%
   (Jaccard) — a learned predictor (Exp C) has headroom; the trace format now
   supports training it (per-layer logits per token).
4. Trace overhead is real (~23% on this box) — before/after comparisons must
   keep instrumentation constant on both sides.

Next runs planned: same trace on T4 (HF job, custom image), Qwen2.5-MoE-A2.7B
(on Jared or office), gating experiment (--remora-gate K: compute only top-K
experts, measure drift vs speedup).
