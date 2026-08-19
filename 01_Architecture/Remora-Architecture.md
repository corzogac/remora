# Project Remora: Architecture & Mathematical Foundations

**Author**: Gerald Corzo (`corzogac`)  
**Project**: Remora — Continuous Wave-Dynamic Latent Inference & Sub-Symbolic Learning  
**Date**: August 2026  

---

## 1. Introduction & Motivation

Contemporary Large Language Models (LLMs) operate under the discrete symbolic paradigm: every concept, identity, system rule, and past interaction is translated into tokenized strings. In multi-turn agentic workflows (e.g. Janus running at IHE Delft), the system re-tokenizes thousands of tokens of identical persona and governance definitions before appending a brief user query.

**Project Remora** reformulates inference and adaptation through the lens of continuous physical wave mechanics. By treating the Transformer residual stream as a continuous state-space trajectory $\mathbf{z}(t, \tau)$, we decouple identity and memory from symbolic text, achieving:
1. **Zero-Token Latent Ingestion**: Pre-computing and hardcoding the persona's standing wave directly into layer biases.
2. **Sub-Symbolic Lifelong Learning**: A lightweight "pilot-fish" companion network (Remora) that learns directly from the activation wake of the primary model.
3. **Predictive NVMe Prefetching**: Forecasting Mixture-of-Experts (MoE) routing targets to eliminate I/O wait-states in disk-streamed engines like Colibri.

---

## 2. Mathematical Formalization

### 2.1 The Residual Stream as a Continuous 2D Wave Field
Let $\tau \in [0, L]$ represent continuous layer depth, and let $t \in [1, T]$ represent the discrete sequence token index. In the continuous depth limit, the residual stream evolution satisfies a non-autonomous Neural Ordinary Differential Equation (Neural ODE):

$$\frac{\partial h(t, \tau)}{\partial \tau} = f_\theta\left(h(t, \tau), \tau, x_{\le t}\right)$$

Coupling the depth dynamics $\tau$ with the autoregressive context expansion $\Delta t = 1$ yields the **2D Damped Semantic Wave Equation**:

$$\frac{\partial^2 h(t, \tau)}{\partial \tau \partial t} + \gamma \frac{\partial h(t, \tau)}{\partial \tau} = \mathcal{D}_s \frac{\partial^2 h(t, \tau)}{\partial t^2} + \mathcal{J}_\theta\left(h(t, \tau)\right)$$

where:
- $\gamma > 0$ is the dissipative damping coefficient (enforced by LayerNorm and residual normalization).
- $\mathcal{D}_s$ is the semantic dispersion tensor governing historical token back-coupling.
- $\mathcal{J}_\theta(h)$ is the non-linear forcing function representing FFN and MoE expert projections.

### 2.2 Phase Space Decomposition: Three Wave Regimes

```
   State A: Ground Potential         State B: Persona Limit Cycle        State C: Dialog Perturbation
      (Base System Wave)               (Standing Wave Invariance)             (Traveling Waveform)
      
          Layer Depth                        Layer Depth                          Layer Depth
        +-------------+                    +-------------+                      +-------------+
        |   ______    |                    |  /\  /\  /\ |                      |    _/\_     |
        |  /      \   |                    | /  \/  \/  \|                      |   /    \    |
        | /        \  |                    |             |                      | _/      \_  |
        +-------------+                    +-------------+                      +-------------+
          Context Token                      Context Token                        Context Token
```

1. **The Ground Potential $\mathcal{V}_0(h)$**: The neutral language prior.
2. **The Persona Standing Wave $\mathcal{V}_{\text{persona}}(h)$**: Invariant identity rules (e.g. Janus persona) establish a stable limit cycle $\mathbf{z}^*(t, \tau)$. Because this attractor is static across sessions, it can be computed once and stored as a static latent bias tensor $\mathbf{b}^{(l)}_{\text{persona}} \in \mathbb{R}^{d_{\text{model}}}$.
3. **The Dialog Perturbation $\delta \mathbf{z}$**: User input injects kinetic energy, launching a localized wave packet that travels diagonally through $(t, \tau)$ space, activating specific expert clusters before settling back to the persona attractor.

---

## 3. The Remora Symbiotic Engine (Pilot Fish)

### 3.1 Dual-Model Hierarchy

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRIMARY MODEL: THE SHARK                        │
│                   (Large Transformer / MoE Engine)                     │
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
│   • Ring-Buffer Velocity Tracker:  v_l = h_l(t) - h_l(t-1)             │
│   • Sub-Symbolic Wave Memory:      Continuous manifold adaptation      │
│   • Expert Target Predictor:       Top-k routing forecast              │
│   • Async Prefetch Signal:         posix_fadvise / Overlapped I/O      │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Expert Prefetching Mechanics (Colibri Integration)
In disk-streamed MoE execution (Colibri), reading expert weights from NVMe into memory takes $100\,\mu\text{s} - 1\,\text{ms}$. 

Remora observes the activation state $h^{(l-1)}$ and velocity $v^{(l-1)}$, calculates a second-order Taylor extrapolation:

$$\hat{h}^{(l+1)} \approx h^{(l)} + v^{(l)} + \frac{1}{2} a^{(l)}$$

and projects against the router matrix $\mathbf{W}_r^{(l+1)}$ to issue asynchronous OS prefetch calls (`ReadFile` with `OVERLAPPED` on Windows, `posix_fadvise` on Linux) **while Layer $l$ attention is still computing**. By the time the router layer executes, the expert weights are already resident in RAM.

---

## 4. Sub-Symbolic Memory & Lifelong Learning

Instead of updating textual knowledge graphs or writing Markdown logs:
1. Every conversation produces an activation trace trajectory $\mathcal{T} = \{h(t, \tau)\}$.
2. Remora updates a continuous low-rank dynamic tensor $\mathbf{W}_{\text{remora}}$ via regularized Hebbian / delta updates:
   $$\Delta \mathbf{W}_{\text{remora}} = \eta \cdot \left( v(t, \tau) \otimes h(t, \tau) - \lambda \mathbf{W}_{\text{remora}} \right)$$
3. Over weeks of interaction, Remora's internal attractor basins mold to the user's specific domain vocabulary, reasoning rhythms, and decision style.

---

## 5. Summary of Experimental Verification Steps

| Experiment | Focus | Core Metric | Expected Outcome |
|---|---|---|---|
| **Exp 1: Latent State Freezing** | Zero-token prompt initialization | Perplexity, Output Cosine Sim, Token Count | 100% elimination of system prompt token cost with $<1\%$ divergence in output distribution |
| **Exp 2: Wave Routing Forecast** | Remora expert prefetch accuracy | Top-1 & Top-2 routing accuracy, I/O wait time | $>75\%$ Top-1 routing hit rate; $2\times$ reduction in NVMe disk stall time in Colibri |
| **Exp 3: Sub-Symbolic Memory** | Continuous activation manifold adaptation | Trajectory alignment across days | Organic adaptation to research style without text prompt updates |
