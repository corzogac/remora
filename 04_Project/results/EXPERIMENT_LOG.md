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

## 2026-08-21 — FIRST ORNITH-1.5-35B-A3B TRACE (office A2000)

Model: ornith-ai/Ornith-1.5-35B-A3B Q4_K_M (21.7 GB), remora llama.cpp build,
-ngl 6 (layers 0-5 GPU, rest CPU), -c 8192, 16 threads, reasoning off.
Server timings: 2.65 tok/s gen, 2.33 tok/s prompt. Trace: trace_ornith.jsonl
(15.8 MB, 76 tokens traced). Full results: results/ornith-office/.

Architecture from trace: 40 MoE layers, 256 experts/layer, top-2 routing,
shared expert + Qwen3 nextn heads (seen in model-load tensor log).

Router profile — EXTREMELY flat (opposite of LFM2):
- consecutive-token exact top-k persistence: 0.0% (LFM2: 4.0%)
- consecutive-token Jaccard: 21.5% (LFM2: ~32%)
- router softmax entropy 5.06-5.46 nats vs ln(256)=5.545 uniform max (91-98%)
- all 256 experts selected within 76 tokens; 207-229 distinct per layer
- top expert selection share 0.93%; top-1 weight median 3.1%
-> No hot experts, no persistent path. Naive same-top-k prefetch has ZERO
   hit rate here. Ornith-1.5's GRPO training appears to produce deliberately
   balanced routing. The learned-observer story is now the ONLY prefetch path.

Stall data (the remora target is real):
- expert FFN (down+gate+up) = 96% of traced op time, ~394 ms/token total
- steady-state op median 1.3 ms; p99 61.5 ms; MAX single-op stall 406 ms
- 718 ops > 5 ms and 264 ops > 20 ms of 76,000 op records
- spikes = cold mmap page-ins of expert weights from the 21.7 GB file
  (first-touch of each layer's experts); prefetch/warm-up would hide them.

Operational notes:
- -ngl 99 on 4 GB A2000 = cudaMalloc OOM (19.9 GB buffer) -> model fails to
  load. This build has no auto-fallback; cap layers to fit VRAM.
- First analysis pass had a units bug (us/1e6 labelled "ms" -> was seconds);
  corrected in results/ornith-office/ornith_office_trace.json.
- Download via resumable dl_ornith.py (msys64 python, schtasks SYSTEM) at
  ~120 MB/s office link; byte-verified against HF x-linked-size.

Next: baseline (untraced) vs traced delta on Ornith with identical flags,
then --remora-gate K pilot on the flat router (expect small drift but also
small compute savings since mass is spread), then learned-observer training
data prep from this trace.

## 2026-08-21 — ORNITH UNTRACED BASELINE (office A2000) — trace overhead

Same flags as the traced run (-ngl 6, -c 8192, 16 threads), REMORA_TRACE_FILE
unset (hook is a compile-time no-op without it), 3 identical completions.

| run | gen tok/s | prompt tok/s | note |
|---|---|---|---|
| 1 | 3.09 | 2.88 | cold: model load + first-touch page-ins |
| 2 | 4.61 | 8.82 | warm pages, prompt cached (cache_n=24) |
| 3 | 6.36 | 9.01 | fully warm |

vs traced (single cold completion): 2.65 gen / 2.33 prompt.
-> Trace overhead (cold vs cold): ~14.2% gen, ~19.1% prompt (LFM pilot: 23%).
Caveat: traced run was single-shot; warm-vs-warm delta still TBD.

Warmup observation (the remora stall story in one number): cold->warm decode
speeds up 2.06x (3.09 -> 6.36 tok/s) purely from the 21.7GB mmap working set
warming in page cache. Expert page-ins are ~half the achievable throughput
on this host; prefetch hides exactly this.

Results: results/ornith-office/ornith_office_baseline.json.

## 2026-08-21 — ORNITH GATE K=1 (office A2000) — compute gating is NOT the lever

REMORA_GATE_K=1: forced top-1 routing (2nd routed expert matmuls skipped),
3 identical completions, same flags as baseline.

| run | gate1 tok/s | baseline tok/s |
|---|---|---|
| 1 (cold) | 3.05 | 3.09 |
| 2 (warm) | 5.75 | 4.61 |
| 3 (warm) | 5.91 | 6.36 |

Verdict: no consistent speedup (warm mean +6.3%, within noise). Output drift:
NONE on this task — both produce the correct iterative fibonacci (functionally
identical; single sample, not generalizable).

Why: at -ngl 6 with 1-token batches the wall is CPU<->GPU offload sync + cold
mmap page-ins, not expert FLOPs. nvidia-smi showed 0% GPU util during the
traced run; traced op wall-times overlap (sum >> decode wall time), so
removing ops does not remove wall time 1:1. Expert gating can only pay off
where expert matmuls ARE the critical path -> full-GPU host (HF T4/A10G).

Honest boundary result for the paper: on the offload-bound regime, remora's
prefetch/observer direction is the lever; compute gating is a dead end.

Results: results/ornith-office/ornith_office_gate1.json.
Next: gate1 on HF T4 (expert matmuls on-GPU), traced warm multi-shot delta.
