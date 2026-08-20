# Remora — Experiment Log

## 2026-08-20 — HF T4 parallel runs (A/B/C) — REAL CROSS-HARDWARE/CROSS-MODEL DATA

Three parallel T4 jobs with the remora Linux build (static, sm_75). Traces:
`lfm_t4_trace.jsonl` (33.7k lines), `qwen15moe_t4_trace.jsonl` (117.6k),
`lfm_t4_gate4.jsonl` (33.7k). Key results:

| metric | LFM T4 | LFM A2000 (pilot) | Qwen1.5-MoE T4 |
|---|---|---|---|
| experts | 64 | 64 | 120 |
| distinct fired | 32/64 | 32/64 | 60/120 |
| router top-1 prob | 0.321 | 0.310 | 0.100 |
| top-1 mass share | 67.5% | 67.8% | 37.7% |
| naive exact-set hit | 6.6% | 4.0% | 0.0% |
| naive Jaccard | 0.33–0.56 | 0.19–0.48 | 0.05–0.10 |
| recurrent tail | GONE (12-14us) | 10x (274-380us) | — |
| throughput (traced) | 69.4 tok/s | 13.6 tok/s | — |

Findings:
1. Wave structure is hardware-INVARIANT (LFM: 32/64 + ~68% mass on both A2000
   and T4) and model-SPECIFIC (Qwen1.5-MoE: diffuse, 60/120, 38%).
2. The 10x recurrent-tail stall on the A2000 is an OFFLOAD artifact — uniform
   on full-GPU T4. Prefetch only pays in offloaded settings.
3. Naive predictor fails everywhere (0–6.6% exact); learned observer headroom
   exists where the router is structured (LFM Jaccard to 0.56).
4. GATE CONTROL: K=4 == native LFM top-k (nu=4) → A≡C token-for-token
   (timings within 0.3%). Gate K=2/K=1 runs in flight (2026-08-21).

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
