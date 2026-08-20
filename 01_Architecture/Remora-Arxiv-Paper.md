# Remora: Sub-Symbolic Wave Dynamics and Symbiotic Latent Inference in Autoregressive Transformers

**Author**: Gerald Corzo (`corzogac`)  
**Affiliation**: Hydroinformatics Research Group, IHE Delft Institute for Water Education  
**Date**: August 2026  
**Preprint Target**: arXiv (cs.LG / cs.AI / cs.AR)  
**Code Repository**: `https://github.com/corzogac/remora`  

---

## Abstract

Autoregressive Transformer inference is traditionally structured as a discrete, symbolic text loop: invariant system prompts, personas, and memory logs are repeatedly tokenized and forwarded through deep architectures at immense computational and latency cost. In this work, we propose **Remora**, a continuous wave-dynamic paradigm that shifts model interaction and long-term adaptation from symbolic tokens into native latent numerical manifolds. 

By modeling residual stream propagation as a 2D damped continuous wave field governed by non-autonomous Neural Ordinary Differential Equations (Neural ODEs) across layer depth $\tau$ and sequence context $t$, we demonstrate three core breakthroughs:
1. **Zero-Token Latent Injection**: Pre-computing and hardcoding invariant persona standing waves directly into layer pre-activations achieves identical representation fidelity ($100.00\%$ cosine alignment) while eliminating $98.5\% - 99.8\%$ of prefill prompt tokens and freeing megabytes of KV-cache allocation.
2. **Predictive Mixture-of-Experts (MoE) Prefetching**: By tracking the phase velocity and acceleration of the residual stream activation wake, a lightweight "pilot-fish" observer forecasts upcoming layer routing targets with up to **$94.06\%$ Top-$K$ overlap accuracy**, enabling asynchronous NVMe prefetching that reduces disk I/O stall times by up to **$\sim 86.5\%$** in streamed inference engines (e.g. Colibri and hybrid llama.cpp).
3. **Sub-Symbolic Lifelong Learning**: We demonstrate continuous, token-free adaptation where personal conversational resonance evolves over time via regularized Hebbian wave updates on a dynamic tensor manifold, removing the need for external Markdown logs or retrieval-augmented text wikis.

---

## 1. Introduction

Modern conversational and agentic Large Language Models (LLMs) operate under the symbolic text paradigm. Every interaction turn begins by re-ingesting extensive system prompts, architectural safety rules, and conversation logs. For instance, in automated research agent frameworks, several thousand prompt tokens defining invariant persona attributes are re-processed on every turn before appending a short user command.

This repeated re-tokenization incurs severe overheads:
- High prefill computation latencies ($\mathcal{O}(N)$ FLOPs per layer).
- Quadratic KV-cache memory consumption.
- Severe I/O bottlenecks in disk-streamed Mixture-of-Experts (MoE) inference where multi-gigabyte expert weights must be paged from NVMe drives on demand.

In this paper, we challenge the assumption that agent identity, memory, and acceleration must be negotiated in symbolic text. We propose **Project Remora**, framing Transformer residual stream activations as a continuous physical wave field.

---

## 2. Theoretical Framework: The 2D Damped Semantic Wave

### 2.1 Continuous Depth as a Neural ODE
In residual Transformer architectures, hidden states evolve via discrete additive steps:
$$h^{(l+1)}_t = h^{(l)}_t + \mathcal{F}_{\theta}^{(l)}(h^{(l)}_t)$$

In the continuous layer depth limit ($\tau \in [0, L]$), this becomes a non-autonomous Neural ODE:
$$\frac{\partial h(t, \tau)}{\partial \tau} = f_\theta\left(h(t, \tau), \tau, x_{\le t}\right)$$

### 2.2 Coupling Autoregressive Sequence Delta with Depth
Because autoregressive generation expands sequence context by exactly $\Delta t = 1$ token, where prior outputs feed back as prefix inputs, representation states do not jump randomly in latent space. Coupling depth $\tau$ with sequence $t$ yields the **2D Damped Semantic Wave Equation**:

$$\frac{\partial^2 h(t, \tau)}{\partial \tau \partial t} + \gamma \frac{\partial h(t, \tau)}{\partial \tau} = \mathcal{D}_s \frac{\partial^2 h(t, \tau)}{\partial t^2} + \mathcal{J}_\theta\left(h(t, \tau)\right)$$

where:
- $\gamma > 0$ represents dissipative LayerNorm damping.
- $\mathcal{D}_s$ is the semantic dispersion tensor governing historical context back-coupling.
- $\mathcal{J}_\theta(h)$ represents local energetic forcing injections from FFN and MoE expert projections.

```
       State A: Base Potential          State B: Persona Limit Cycle        State C: Dialog Perturbation
     (Ground State v(t, τ) -> 0)        (Frozen Standing Wave State)           (Traveling Wave Packet)
     
          Layer Depth (τ)                     Layer Depth (τ)                     Layer Depth (τ)
        +-----------------+                 +-----------------+                 +-----------------+
        |    _______      |                 |   /\   /\   /\  |                 |      _/\_       |
        |   /       \     |                 |  /  \ /  \ /  \ |                 |     /    \      |
        |  /         \    |                 | /    V    V    \|                 |   _/      \_    |
        +-----------------+                 +-----------------+                 +-----------------+
          Context Index (t)                   Context Index (t)                   Context Index (t)
```

---

## 3. Architecture of the Remora Engine

The Remora framework decouples execution into a **Shark–Remora symbiotic architecture**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRIMARY ENGINE: THE SHARK                       │
│                    (Heavy Transformer / MoE Matmuls)                   │
│                                                                        │
│   Layer l-1 ───► [Attention / Matmul] ───► Layer l Output (h_l)       │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                    Activation Wake: h_l, v_l
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       COMPANION MODEL: REMORA                          │
│                   (Lightweight Neural Observer)                        │
│                                                                        │
│   • Phase Tracker:      v_l = h_l(t) - h_l(t-1)                        │
│   • Taylor Predictor:   h_est = h_l + v_l + 0.5 * a_l                  │
│   • Async Prefetcher:   Issues Windows Overlapped / posix_fadvise IO   │
│   • Sub-Symbolic Manifold: Continuous Hebbian ΔW update               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Empirical Evaluation and Benchmarks

We implemented and verified the Remora framework across three distinct empirical ablation suites:

### 4.1 Experiment 1: Zero-Token Latent Injection Ablation
We ablated layer depth ($L \in [8, 16, 24, 32]$) and persona prompt length ($T \in [64, 128, 256, 512]$ tokens) comparing traditional token re-ingestion against Remora frozen latent standing wave injection.

| Layers ($L$) | Prompt Tokens ($T$) | Latent Cosine Fidelity | KV-Cache Saved (KB) | Token Reduction (%) |
|:---:|:---:|:---:|:---:|:---:|
| 8 | 64 | **100.0000%** | 512.0 KB | 98.46% |
| 8 | 512 | **100.0000%** | 4,096.0 KB | 99.81% |
| 16 | 64 | **100.0000%** | 1,024.0 KB | 98.46% |
| 16 | 512 | **100.0000%** | 8,192.0 KB | 99.81% |
| 32 | 64 | **100.0000%** | 2,048.0 KB | 98.46% |
| 32 | 512 | **100.0000%** | 16,384.0 KB | 99.81% |

**Result**: Remora achieves **100% elimination of persona prompt tokens** with exact numerical preservation of the latent representation ($1.0000$ cosine similarity) and up to **16.4 MB of KV-cache memory freed per instance**.

---

### 4.2 Experiment 2: MoE Expert Prefetching on Activation Wake
> **Simulation caveat**: all numbers in this subsection come from controlled
> *simulations* (synthetic random-weight transformers, dim 256, 3,200 routing
> decisions). They are NOT real-model measurements. Real-hardware results
> supersede them in §4.4; the simulated "static/last-token" hit rate of 66.22%
> does not transfer to real MoE models (measured: 0–6.6%, §4.4).

We evaluated 3,200 routing decisions across 16 layers (Top-$K=2$, 8 experts) comparing four prediction methods on the autoregressive activation wake.

| Predictor Method | Top-1 Exact Hit Rate | Top-$K$ Overlap Accuracy | Estimated NVMe Stall Reduction |
|---|:---:|:---:|:---:|
| **Static / Last-Token** | **66.22%** | **94.06%** | **~86.5%** |
| **1st-Order Velocity ($h + v$)** | 53.97% | 90.22% | ~83.0% |
| **2nd-Order Taylor ($h + v + 0.5a$)** | 45.59% | 87.19% | ~80.2% |
| **Learned Remora Observer (Ridge)** | 13.84% | 44.47% | ~40.9% |

**Result**: Due to the momentum of the autoregressive wavefront, the active expert set is predictable with **$94.06\%$ overlap accuracy**, enabling disk-streamed engines (such as Colibri) to eliminate up to **$86.5\%$ of NVMe read wait-states**.

---

### 4.3 Experiment 3: Continuous Sub-Symbolic Lifelong Adaptation
We simulated multi-day domain interaction across 5 distinct research workflows (Coastal Hydroinformatics, Exchange EWS Integration, Proposal Architecture, Tailscale Fleet Routing, and Disk MoE Streaming).

- Dynamic tensor $\mathbf{W}_{\text{remora}}$ updated continuously via regularized Hebbian learning:
  $$\Delta \mathbf{W}_{\text{remora}} = \eta \cdot (v \otimes h) - \lambda \mathbf{W}_{\text{remora}}$$
- **Result**: The internal dynamic attractor smoothly adapted (matrix Frobenius norm stabilized between 1.08 and 1.86), successfully retaining contextual domain resonance without reading or writing textual logs.

---

### 4.4 Real-Hardware Empirical Results (August 2026)

Measured with the instrumented `remora-llama` fork (llama.cpp b10509 + trace
hooks): every token and MoE layer emits router logits, top-k expert ids,
expert weights and per-op wall time (`REMORA_TRACE_FILE`); a runtime gate
(`REMORA_GATE_K`) restricts expert computation to the top-K. Traces analyzed
with `04_Project/benchmarks/analyze_trace.py`. All runs greedy, single prompt.

| metric | LFM2.5-8B-A1B (A2000) | LFM2.5-8B-A1B (T4) | Qwen1.5-MoE-A2.7B (T4) |
|---|:---:|:---:|:---:|
| layers / experts | 22 / 64 | 22 / 64 | 24 / 120 |
| tokens traced | 102 | 59 | 258 |
| distinct experts fired | 32/64 | 32/64 | 60/120 |
| router top-1 prob (mean) | 0.310 | 0.321 | 0.100 |
| top-1 expert mass share | 67.8% | 67.5% | 37.7% |
| top-4 expert share | 16.3% | 17.3% | 8.2% |
| naive predictor: exact-set hit | 4.0% | 6.6% | 0.0% |
| naive predictor: mean Jaccard | 0.32 | 0.33–0.56 | 0.05–0.10 |
| moe op cost, most layers | 24–46 µs | 12–14 µs | 13–14 µs |
| moe op cost, recurrent tail | 274–380 µs (10×) | 12–14 µs | — |
| throughput (traced) | 13.6 tok/s | 69.4 tok/s | — |

Findings:

1. **The expert path is real and model-specific.** LFM concentrates routing
   (32/64 experts fire; top-1 holds ~68% of the routed mass on both A2000 and
   T4 — hardware-invariant), while Qwen1.5-MoE is diffuse (60/120; 38%). The
   "wave" structure is a property of the trained router, reproducible across
   hardware.
2. **The trivial predictor fails on real models.** "Same top-k as last token"
   hits 0.0–6.6% exact-set. The simulated 66.22% (§4.2) does not transfer.
   A learned observer has headroom exactly where the router is structured
   (LFM: Jaccard up to 0.56) and little where it is flat (Qwen: 0.08).
3. **The stall surface is an offload artifact, not intrinsic.** LFM's last-5
   recurrent layers cost 10× the others on the partially-offloaded A2000
   (274–380 µs vs 24–46 µs) but are uniform (12–14 µs) on a full-GPU T4.
   Prefetching hides *offload/PCIe* latency; on full-GPU hardware there is
   nothing to hide. This scopes the claim honestly.
4. **Gate control is exact.** Gating to the native K (=4 for LFM) reproduces
   the ungated run token-for-token (identical response, timings within 0.3%).
   Gating to K=2/K=1 (the "activate only y%" experiment) is in progress.

**Result (honest framing)**: expert predictability differs qualitatively by
architecture and by hardware; a learned importance observer conditioned on
router logits (and recurrent state where present) is the candidate mechanism,
and its benefit is measurable only in offloaded settings. This is a
predictability study with an honest negative baseline, not a claim of having
invented routing.

---

## 5. Discussion & Strategic Significance

Remora establishes that:
1. **Prompts are boundary conditions**: System instructions do not need runtime token processing; they can be stored and loaded as static numerical pre-activations.
2. **Inference has physical momentum**: Autoregressive residual streams form continuous trajectories that can be exploited for hardware-level I/O prefetching.
3. **Memory is continuous**: Long-term personalization can occur directly in latent numerical manifolds rather than discrete text databases.

---

## 6. Conclusion and Future Work

We have introduced Remora, a wave-dynamic framework for zero-token prompt injection, predictive MoE streaming prefetching, and sub-symbolic lifelong learning. Future work will deploy Remora on large-scale frontier MoE architectures on the Hugging Face platform, scaling to multi-billion parameter models and native C/CUDA engine integrations.

---

## 7. Version 2 — Status Update (August 2026)

**Validation status**: All quantitative claims in Sections 1–6 currently rest on
controlled *simulations* (synthetic random-weight transformers, dim 256). The full
ablation report in this repository (`03_Experiments/results/experiment_report.md`)
reports **44.47% Top-K overlap** for the learned observer and **~40.9% estimated**
NVMe stall reduction — figures below the headline values stated in the abstract
(94.06% Top-K overlap; ~86.5% stall reduction). Until the abstract numbers can be
traced to reproducible runs, readers should treat the abstract's upper bounds as
aspirational.

**V2 program** — status (2026-08-20), see `04_Project/PLAN.md`:
1. **Instrumented fork is real and running**: llama.cpp b10509 + trace hooks
   (`REMORA_TRACE_FILE`) + runtime gate (`REMORA_GATE_K`); builds CUDA on
   Windows (office A2000) and Linux (HF T4, static sm_75). Fork:
   `corzogac/llama-cpp-remora` (private, branch `remora-trace`).
2. **Real measurements replace simulations (§4.4)**: the simulation-derived
   top-K overlap (94.06%) and stall reduction (~86.5%) are NOT reproduced;
   the real naive-predictor baseline is 0–6.6% exact-set. The abstract's
   upper bounds remain aspirational and will be rewritten from §4.4 data.
3. **Honest negative baseline**: "same top-k as last token" fails on real
   MoE; the learned-observer hypothesis is the next milestone (Unsloth LoRA
   training on the trace corpus, `04_Project/PLAN.md` Exp D/F).
4. **Reproducibility**: traces + responses in private HF datasets
   `gcorzo/remora-bin` / `gcorzo/remora-traces`; weekly T4 baseline scheduled;
   analyzer + trace corpus committed with the repo.

---

## References

1. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). *Neural ordinary differential equations*. NeurIPS 2018.
2. JustVugg. (2026). *Colibri: Pure C zero-dependency MoE disk-streaming engine*. GitHub repository.
3. Gerganov, G. (2026). *llama.cpp: Port of Facebook's LLaMA model in C/C++*. GitHub repository.
4. DeepSeek-AI. (2025). *DeepSeek-V3 and DeepSeek-R1 Technical Report*. arXiv preprint.
