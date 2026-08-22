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

Verdict: **INVALID — RETRACTED 2026-08-21.** The office binary (b10510, commit
1d66c71d5) predates the gate feature commit fe2673647: REMORA_GATE_K was
silently IGNORED. The office "gate1" run was just another untraced baseline
(no gating happened). The real gate test is the L4 full-GPU run (see below);
its first attempt crashed on the pre-fix gate bug (GGML_ASSERT ggml_view_2d,
fixed in 3e894f758 "gate K<4 crash"), and is being rerun with the fixed
binary. The numbers below are retained only as an extra baseline datapoint.

Why: at -ngl 6 with 1-token batches the wall is CPU<->GPU offload sync + cold
mmap page-ins, not expert FLOPs. nvidia-smi showed 0% GPU util during the
traced run; traced op wall-times overlap (sum >> decode wall time), so
removing ops does not remove wall time 1:1. Expert gating can only pay off
where expert matmuls ARE the critical path -> full-GPU host (HF T4/A10G).

## 2026-08-21 — ORNITH L4 FULL-GPU BASELINE (HF Jobs l4x1) — the right regime

sm8x remora build (b10512 fe2673647, no gate), -ngl 90, traced, 3 runs:

| run | gen tok/s | prompt tok/s | tokens | finish |
|---|---|---|---|---|
| 1 (cold) | 27.13 | 92.78 | 73 | stop |
| 2 | 27.51 | 42.64 | 73 | stop |
| 3 | 27.30 | 41.45 | 73 | stop |

Steady-state ~27.3 tok/s, no warmup curve (model fully in VRAM) — 4.3x the
office warm rate (6.36). GPU is now the critical path: this is the host where
gating can show a real effect. Trace: ornith_l4_base.jsonl (41.5MB, 257k
lines) in gcorzo/remora-traces; results JSON in results/ornith-office/l4/.

NOTE: this binary predates the gate fix (3e894f758) but runs WITHOUT
REMORA_GATE_K, so it is a valid baseline. The gate1 rerun uses the rebuilt
binary (bundle + remora-gate-fix.patch).

## 2026-08-21 — ORNITH L4 GATE K=1 FINAL (fixed binary) — gating is NOT viable

Rebuilt sm8x binary WITH gate fix 3e894f758, REMORA_GATE_K=1, same flags.
Server loaded and passed health — the fix works (no crash).

But ALL 3 completions returned HTTP 500: "The model produced output that
does not match the expected peg-native format". No content, no timings.

Trace forensics (ornith_l4_gate1.jsonl, 89MB):
- ~193 tokens decoded per run vs 73-74 baseline: the gated model rambled to
  near max_tokens and never produced a valid template ending (output garbage).
- Per-token expert FFN roughly halved (down 1.4->0.6 ms, gate 1.3->0.5 ms)
  — the gate does what it says on the compute side.
- 256 experts still selected (top-1 routing over 579 tokens/layer); exact
  persistence 13.6% (single-element sets).
- Same failure class seen earlier on LFM gate runs (empty responses).

Verdict: expert FFN is only ~10% of L4 decode time (3.9 ms of ~37 ms/token
at 27 tok/s), so even perfect gating caps at ~5% speedup — while destroying
output fidelity on a flat router (top-1 weight median 3.1%; the 2nd expert
carries too much mass to drop). Hard gating is a dead end on this model
class on ANY host. Remora's lever remains prefetch/latency-hiding.

Results: results/ornith-office/l4/ornith_l4_gate1.json. This closes the gate
experiment (office attempt retracted; L4 attempt is definitive).

## 2026-08-21 — LEARNED-OBSERVER PROBE — the thesis holds on Ornith

Dataset: 22,440 records (per-layer logits_t -> topk_{t+1}) from the 3 valid
traces (office 3,000 / office3 10,600 / L4 8,840). Probes (pure numpy):

1. Within-host cosine-NN (70/30 random split): exact 57.2%, Jaccard 78.9%
   (argmax persistence 6.6%, chance 11.8%; naive same-topk 0.03%/21.1%).
2. CROSS-HOST: train on office traces only, test on L4 full-GPU greedy
   stream — different token streams (office sampled temp 0.8, L4 greedy):
   exact 51.2%, Jaccard 87.4% (per-layer Jaccard 79.3-95.6%).

Interpretation:
- Next-token expert selection is predictable from router logits alone via
  logit-space similarity; the observer beats the identity heuristic by ~4x
  Jaccard and TRANSFERS across hosts (cheap box trains it, fast box uses it).
- Prefetch value: Jaccard 87% = at least one of the two next experts hit on
  nearly every token (hides half the stall); exact 51% = both experts (hides
  it all).
- Caveats: same model + same task family (one coding prompt) — task-type
  generalization unmeasured; cosine-NN is a lazy probe, a trained MLP must
  match it at lower cost (next step).

Results: results/ornith-office/observer_probe_results.json. Scripts:
cosine_probe.py, cross_host_probe.py, convert_traces_to_observer_dataset.py.

## 2026-08-22 — POSEIDON: REMORA STATIC-CRT CORE + 3RD-HOST TRACE

Deployed the remora-llama fork as the OPERATIONAL core on Poseidon
(HP Zbook 15 G3, i7-6700HQ Skylake, 32GB, Win10):
- Single 19.3MB exe: MSVC static CRT (/MT, BUILD_SHARED_LIBS=OFF), CPU-only,
  AVX2/FMA/F16C (GGML_NATIVE=OFF).
- Endpoint live: http://100.118.223.14:11435/v1, 64K ctx, q8 KV, --parallel 2,
  API key; auto-restart on boot (schtasks onstart) + tray monitor
  (ornith_tray.ps1, green/red dot with Start/Stop).
- Warm steady-state: 7.6-7.8 tok/s (~2.4x the official Clang b10509 measured
  earlier on this host).
- TRACE HOOK CONFIRMED on the third host: trace_poseidon.jsonl (15.7MB)
  captured during the smoke test. Same model weights -> router logits are
  deterministic vs office/L4; adds a third timing profile to the stall data.

Build lessons (committed in tools/poseidon/build_remora_mt.bat):
- 0xC0000135 on clean Win10 = missing VC runtime -> static CRT (/MT) fixes it.
- 0xC000001D illegal instruction = GGML_NATIVE=ON baked AVX-512 (build
  machine = Tiger Lake office box) -> target AVX2/FMA/F16C for Skylake+.

Results: results/ornith-office/poseidon/poseidon_remora_core.json.

Honest boundary result for the paper: on the offload-bound regime, remora's
prefetch/observer direction is the lever; compute gating is a dead end.

Results: results/ornith-office/ornith_office_gate1.json.
Next: gate1 on HF T4 (expert matmuls on-GPU), traced warm multi-shot delta.

## 2026-08-21 — ORNITH TRACED-WARM 3x (office A2000) — warm trace overhead

3 completions WITH the trace hook, same flags as baseline (run_ornith_trace3.bat,
port 11440). Trace evidence: trace_ornith3.jsonl (49.4 MB).

| run | traced tok/s | baseline tok/s | overhead |
|---|---|---|---|
| 1 (cold) | 1.88 | 3.09 | ~39% (also run-to-run variance) |
| 2 (warm) | 3.99 | 4.61 | 13.5% |
| 3 (warm) | 3.93 | 6.36 | 38.2% |

Warm mean overhead ~28% (baseline warm mean 5.49 vs traced warm mean 3.96).
Cold overhead from the single traced run: ~14%.

Interpretation: the trace hook's cost grows on warm runs — when compute is
fast (warm pages), the synchronous per-op JSONL write (router logits etc.)
becomes a larger fraction of decode. Range: 14-38%, call it ~25% typical on
this host. No output drift from tracing (all 3 completions correct).

Implication for the paper: before/after comparisons must keep instrumentation
constant on both sides AND control for warm-up state (run count parity).

## 2026-08-20 — GATE EXPERIMENT RESULTS (LFM on T4, fixed engine)

| config | throughput | output |
|---|---|---|
| K=4 (native, traced) | 69.4 tok/s | baseline haiku |
| K=2 (gated) | 73.0 tok/s (+5.2%) | coherent DIFFERENT haiku (drift) |
| K=1 (gated) | HTTP 500 | peg-format violation — model breaks |

- Gate fix (views/aggregation loops now use gated n_expert_used) validated:
  K=2 runs end-to-end; K=1 crashes the output parser, not the engine.
- Speedup on full-GPU T4 is small (+5.2%) — expert matmuls are a minor
  fraction of decode there AND both runs pay identical trace overhead.
  The offload regime (office A2000, 10x recurrent tail) is where the gate
  payoff should be measured next (office rebuild pending).
- Drift is qualitative: same prompt, different (still good) haiku at K=2.
  K=1 is below the model's viability threshold — the cliff sits in (2,4].
- Control stands: K=4 == native == byte-identical to the ungated run.

## 2026-08-21 — GATE K=2 ON OFFICE A2000 (offload regime) — op-level results

Per-op MoE cost, K=4 vs K=2 (same box, traced, compare_traces.py):

| layer group | K=4 mean | K=2 mean | ratio |
|---|---|---|---|
| layers 2-18 | 24-29 us | 23-25 us | ~0.90 |
| recurrent tail 19-23 | 274-380 us | 192-250 us | **0.66-0.73** |

- The gate cuts expert-matmul cost ~34% in the stall-heavy recurrent tail
  (offload regime) and ~7% elsewhere. Mechanism proven at op level.
- End-to-end tok/s dropped (13.6 -> 7.7) because K=2 CHANGES OUTPUT BEHAVIOR
  on LFM (verbose/self-correcting, 256 vs 99 tokens) and trace I/O scales
  with tokens. Behavior drift confounds end-to-end timing on this small box;
  op-level cost is the clean measurement.
- Combined with T4 full-GPU (+5.2% at K=2), the honest picture: gating halves
  the expert work where it blocks (offload tail), gains depend on hardware
  regime and on the drift-vs-behavior tradeoff of the target model.
