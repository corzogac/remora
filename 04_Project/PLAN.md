# Remora — Experiment Plan (2026-08-20, post-consultation)

## 1. Mission (honest framing)

Prove — with real, reproducible measurements — that expert-aware inference
(prefetch + learned expert importance) improves MoE LLM inference, **or prove
it doesn't**. The negative result is publishable too. We do NOT claim to have
invented MoE routing; we measure where routing knowledge buys latency.

## 2. Measured status (all real)

| item | value | machine |
|---|---|---|
| LFM2.5-8B-A1B baseline (untraced) | 17.7 tok/s gen | office A2000 (partial offload) |
| LFM2.5-8B-A1B baseline | ~138 tok/s gen | HF T4 (full GPU, validated) |
| Traced run (pilot, 102 tokens) | 13.6 tok/s (23% trace overhead) | office A2000 |
| Experts fired in pilot | 32 of 64 | — |
| Top-1 expert mass share | 67.8% of routed mass | — |
| Naive prefetch predictor | 4.0% exact-set / ~32% Jaccard | — |
| Recurrent tail (last 5 layers) | 274–380 us vs 24–46 us (10x) | — |
| Weekly T4 baseline | scheduled Mon 03:00 UTC | HF Jobs |

Pilot artifacts: `results/trace_lfm_office_20260820.jsonl` (58k lines),
`benchmarks/analyze_trace.py`, `results/EXPERIMENT_LOG.md`.

## 3. External consultation (Codex, 2026-08-20)

Verdict: **publishable as a hybrid recurrent-MoE prefetch/predictability study
with an honest negative baseline. NOT publishable as "we invented learned MoE
routing".**

- Expert-prefetching is crowded (PowerInfer, Mixtral-offloading, MoE-Infinity,
  Pre-gated MoE). The 4%/32% naive-predictor result is valuable as a *negative
  baseline*: it refutes the trivial "same top-k as last token" heuristic.
- **The genuinely novel claim:** *"In hybrid recurrent-MoE models (LFM2), the
  dominant inference cost and prefetch-predictability structure differ
  qualitatively from transformer MoE; a learned importance observer conditioned
  on router logits + recurrent state predicts next-layer expert sets better
  than the identity heuristic, enabling prefetch that hides offload latency."*
- Kill weak claims: KV caching, speculative decoding, Mixture-of-Depths are
  orthogonal — cite, don't compete.

## 4. Unsloth integration

Use Unsloth (github.com/unslothai/unsloth — local train+run platform, 74k
stars, 2x training at 70% less VRAM; LoRA/QLoRA/GRPO/DPO/FP8; GGUF/NVFP4
export) for ONE thing: **training the expert-importance observer (Exp C)**.

- Observer: tiny head, input = per-layer router logits (+ recurrent state
  where present), output = next-token expert set. Trained on our traces.
- Unsloth's fast LoRA/QLoRA makes the train loop minutes not hours on a T4.
- Matched Q4_K_M/Triton kernels give a consistent quant baseline across
  office A2000 and HF T4 (removes quant confounds).
- Model breadth: native Qwen3-MoE / DeepSeek-V4 / MiniMax-H3 support = free
  extra testbeds for generality.
- **Hard rule: instrumented llama.cpp stays the latency ground truth.**
  Unsloth trains; it never produces the timing numbers.

## 5. Experiment matrix

| id | experiment | model | where | status |
|---|---|---|---|---|
| A | trace (wave) | LFM2.5-8B-A1B | HF T4 | queued (build job gate) |
| B | trace (wave) | Qwen1.5-MoE-A2.7B (64exp top-8) | HF T4 | queued |
| C | gate K=4 vs K=8, drift + speedup | LFM2.5-8B-A1B | HF T4 | queued (needs REMORA_GATE_K build) |
| D | observer training (Unsloth LoRA) | traces from A/B/C | T4/local | planned |
| E | scale-up: ≥10k tokens, multi-prompt, 2+ models | all | HF T4 weekly | planned |
| F | prefetch impl: expert warm-up hiding PCIe/offload latency | LFM2 + Qwen1.5-MoE | office A2000 | planned |

## 6. Guardrails (from consultation + ours)

1. Simulated hit-rate ≠ speedup: report end-to-end wall-clock tok/s with
   prefetch ON/OFF on the SAME offload config.
2. Trace overhead is real (23% in pilot): final speedup measured against the
   UNTRACED baseline; trace overhead reported separately.
3. Pilot is 102 tokens/1 prompt/1 model: no distributional claims until
   ≥10k tokens, multiple prompts/domains, ≥2 models. "32/64 experts" is
   prompt-dependent — say so.
4. Never compare across quant/offload/hardware for a headline number; fix
   quant + offload split per comparison.
5. K=4 vs K=8: report drift (perplexity/task accuracy), not just speedup —
   a speedup that degrades output is not a win.
6. Before/after always on identical build + instrumentation configuration.

## 7. Pipeline & reproducibility

- Fork: corzogac/llama-cpp-remora, branch remora-trace (b10509); trace via
  REMORA_TRACE_FILE, gate via REMORA_GATE_K; compiles CUDA on office (Windows)
  and HF (Linux static, sm_75).
- HF datasets: gcorzo/remora-bin (bundle + Linux binary + scripts),
  gcorzo/remora-traces (results, private until paper).
- Weekly T4 baseline: hf jobs scheduled, Mon 03:00 UTC.
- Analyzer: benchmarks/analyze_trace.py (router confidence, expert path,
  mass concentration, stall per layer, prefetch hit-rate sim).

## 8. Next actions

1. [in flight] HF build job → binary on remora-bin → fire A, B, C in parallel.
2. Analyze A/B/C traces; compare wave structure across models + hardware.
3. Drift analysis C: gated vs ungated responses (greedy), token-level.
4. Build the ≥10k-token trace corpus (multi-prompt, both models).
5. Unsloth observer prototype on the trace corpus; measure real prefetch
   speedup on the A2000 with prefetch ON/OFF (Exp F).
6. Write-up: experiment log + figures; paper framing per section 3.
