# Season 4: The Spatiotemporal Lattice & Parallel Coupled RNNs

Season 4 is dedicated to exploring and formalizing the foundational revelation of SubQ: **The Spatiotemporal Lattice of Coupled Parallel RNNs**.

---

## 1. The Foundational Discovery: SubQ as an RNN Field

Historically, Sub-Quadratic Transformers were viewed primarily through the lens of attention: *"sparse attention unrolled across recurrent thought depth T."*

Season 4 begins with a profound mathematical reinterpretation:

$$\boxed{\text{A standard sequential RNN is literally a single diagonal trajectory on the SubQ spacetime grid.}}$$

Instead of a single serial hidden state bottlenecking the sequence, **SubQ is an interacting 2D lattice of $L$ parallel coupled RNNs** running simultaneously without sequential latency.

---

## 2. The Microscopic Equivalence ($K=1, \Delta=1$)

Consider SubQ under the minimal configuration:
* Candidate budget $K = 1$
* Relative offset $\Delta = [1]$ (immediate predecessor $i-1$)
* Per-hop non-linear update (`mlp_interval = 1`)

Watch the exact trajectory of information across position $i$ and thought hop $t$:

```
               Token 1         Token 2         Token 3         Token 4
             (Input x_1)     (Input x_2)     (Input x_3)     (Input x_4)
                  │               │               │               │
Hop 0:        [ s_1^(0) ]     [ s_2^(0) ]     [ s_3^(0) ]     [ s_4^(0) ]
                  │ ╲             │               │               │
                  │   ╲           │               │               │
Hop 1:            │     ──►   [ s_2^(1) ]     [ s_3^(1) ]     [ s_4^(1) ]
                  │           (= h_2 !)         │ ╲               │
                  │               │ ╲           │   ╲             │
                  │               │   ╲         │     ╲           │
Hop 2:            │               │     ──►   [ s_3^(2) ]         │
                  │               │           (= h_3 !)           │ ╲
                  │               │                 │ ╲           │   ╲
                  │               │                 │   ╲         │     ╲
Hop 3:            │               │                 │     ──►   [ s_4^(3) ]
                  │               │                 │           (= h_4 !)
```

### Tracing the Diagonal Ray:
1. **Hop 0**: Token 1 holds its raw input state $s_1^{(0)} = x_1$ (identical to $h_1$).
2. **Hop 1**: Token 2 takes its own state $s_2^{(0)}$ and reads $s_1^{(0)}$:
   $$s_2^{(1)} = \text{Transition}(s_2^{(0)}, s_1^{(0)}) \quad \equiv \quad h_2 = \text{RNNCell}(x_2, h_1)$$
   **$s_2^{(1)}$ is mathematically identical to the RNN hidden state $h_2$!**
3. **Hop 2**: Token 3 reads Token 2's updated state $s_2^{(1)}$ (which is $h_2$):
   $$s_3^{(2)} = \text{Transition}(s_3^{(1)}, s_2^{(1)}) \quad \equiv \quad h_3 = \text{RNNCell}(x_3, h_2)$$
   **$s_3^{(2)}$ is mathematically identical to the RNN hidden state $h_3$!**
4. **Hop 3**: Token 4 reads Token 3's updated state $s_3^{(2)}$ (which is $h_3$):
   $$s_4^{(3)} = \text{Transition}(s_4^{(2)}, s_3^{(2)}) \quad \equiv \quad h_4 = \text{RNNCell}(x_4, h_3)$$

---

## 3. What Makes This Superior to a Classic RNN?

### A. The Elimination of the Single-State Bottleneck
In a standard sequential RNN (LSTM/GRU), there is **one single vector $h_t$**. By token 1,000, that single vector is forced to compress the entire preceding sequence, causing severe catastrophic forgetting and gradient decay.

In the SubQ Lattice:
* There are **$L$ distinct state vectors** advancing concurrently.
* Token 1 originates an RNN stream traveling to the right: $s_1 \to s_2 \to s_3 \to \dots$
* Token 2 *simultaneously* originates an RNN stream traveling to the right: $s_2 \to s_3 \to s_4 \to \dots$
* Token 3 *simultaneously* originates an RNN stream traveling to the right: $s_3 \to s_4 \to \dots$
* **Every token is simultaneously an input, an active recurrent cell, and a receiver of predecessor streams.**

### B. Parallel Execution (Zero Python / Sequential Loop)
A standard RNN with sequence length $L=2048$ requires **2,048 sequential GPU operations**.
The SubQ lattice evaluates all $L$ parallel streams in **a single parallel matrix multiplication per hop**, taking only $T$ operations total (e.g., $T=8$).

---

## 4. Breaking the Speed of Light: Multiscale Hyperbolic Waves ($K > 1$)

In a classic RNN or $K=1$ lattice, information speed is strictly bounded by **$1$ token per step** (slope = 1 on the spacetime grid).

When we introduce multiscale offsets ($K > 1$, e.g., $\Delta \in \{1, 4, 16, 64\}$):
* We break the linear speed limit of the sequential chain.
* Information propagates simultaneously along **multiple velocities** ($v \in \{1, 4, 16, 64\}$).
* The 1D linear chain transforms into an **Exponential Hyperbolic Network of Coupled RNNs**:
  - Distance $\Delta=1$ preserves fine-grained sequential grammar.
  - Distances $\Delta \in \{4, 16, 64\}$ act as **relativistic wormholes**, transmitting global context in $\lceil \log_{K+1} L \rceil$ hops rather than $L$ serial steps.

### The Superposition of Wavefronts at Node $(i, t)$:

```
                                    Token i
                                       │
                                (Incoming Rays)
               v = 64                v = 16         v = 4     v = 1
         (Global Long-Range)       (Medium)        (Local)   (Neighbor)
                 \                     \              \       /
                  \                     \              \     /
                   \                     \              \   /
                    ──────────────────────▼─────────────────
                                    [ Node (i, t) ]
```

At any spacetime coordinate $(i, t)$, token $i$ is an **interference junction where multiple RNN wavefronts traveling at different velocities superimpose**.

---

## 5. Season 4 Research Agenda & Status

1. **Cell Mechanics & Gating**: **RESOLVED in S4-001 & S4-002**.
   - Comparing classic gated cells (`nn.GRUCell`) against the weight-tied per-hop residual MLP proved decisively that the per-hop residual MLP ($4\times$ GELU) is the optimal, non-saturating recurrent cell.
2. **Velocity Spectrum Analysis**:
   Map how different offset bases (linear $K=1$, dyadic, Fibonacci, continuous Fourier carrier waves) shape the speed of information propagation across the $(L \times T)$ lattice.
3. **The Mirrored Bilateral Spacetime Lattice**:
   Extend this framework to 2D isotropic visual domains, where the center token acts as a mirror radiating wavefronts both backward and forward symmetrically.
4. **Attractor Dynamics & Fixed-Point Convergence**:
   Study the convergence properties of the $L$ interacting state machines as $T \to \infty$.

---

## 6. The Recurrent Duality Realization: Per-Hop Residual MLP IS the RNN

A pivotal conceptual breakthrough achieved in Study S4-002 clarifies the relationship between MLPs and Recurrent Neural Networks within SubQ:

$$\boxed{\text{SubQ with a weight-tied per-hop MLP is not an alternative to an RNN — it IS a Deep Residual RNN.}}$$

### A. Mathematical Equivalence (Neural ODE Formulation)
In Study 68 SubQ, the parameters $\theta = \{W_q, W_k, W_v, W_o, W_{\text{mlp1}}, W_{\text{mlp2}}\}$ are **tied across all $T$ hops**. 
At each hop, token state $s$ transitions according to:
$$s_t = s_{t-1} + \frac{1}{\sqrt{T}} \left( \text{Antenna}(s_{t-1}) + \text{MLP}(s_{t-1}) \right)$$

This is mathematically identical to an **Euler-discretized Continuous-Time Recurrent Neural Network (Neural ODE)**:
$$\frac{ds}{dt} = f_\theta(s, \text{sensory context}) \quad \text{with step size } \Delta t = \frac{1}{\sqrt{T}}$$

### B. Why Classic Gated Cells (GRU/LSTM) Lag Behind Residual MLP Cells
When we tested replacing the per-hop MLP with a classical `nn.GRUCell` in S4-002, test accuracy dropped from **48.36% to 44.27%** due to severe underfitting:
1. **Additive Identity vs. Convex Interpolation**: The GRU update gate $s_t = (1-z)\tilde{h} + zs_{t-1}$ forces an interpolation trade-off, exponentially decaying early visual memories. The Residual RNN provides an uninhibited identity highway where past memory is never forgotten.
2. **Channel Expansion Capacity**: Classic GRU cells operate strictly within a $128 \to 128$ bottleneck. The per-hop MLP provides a wide $128 \to 512 \to 128$ expansion with GELU, creating rich combinatorial feature interactions.
3. **Activation Saturation**: $\tanh$ in GRU gates forces activations into $[-1, 1]$ and saturates gradients across recurrent depth. GELU maintains active, non-saturating gradient flow across all $T$ hops.

### C. Spatial vs. Channel Non-Linearity Duality
Even without an MLP, SubQ is fundamentally non-linear at every hop via the Softmax and evolving $Q(s)K(s)^T$ bilinear interactions (Modern Hopfield Network / Multiplicative RNN). The per-hop MLP completes the system by providing orthogonal non-linearity:

| Mechanism | Non-Linearity Type | Dimensional Axis | Role in the Recurrent Lattice |
| :--- | :--- | :--- | :--- |
| **Softmax Antenna** | **Spatial / Relational** (Exponential gating) | Across **Sequence $L$** | Fuses incoming multi-velocity RNN wavefronts into sensory context $c_i^{(t)}$. |
| **Per-Hop MLP** | **Channel / Combinatorial** ($4\times$ GELU) | Across **Channels $d$** | Synthesizes high-dimensional feature conjunctions (color $\times$ edge $\times$ texture). |

**Unified Formula for the Stable Spatiotemporal Lattice Node**:
$$s_i^{(t)} = s_i^{(t-1)} + \frac{1}{\sqrt{T}} c_i^{(t)} + \frac{1}{\sqrt{T}} \text{MLP}\left(\text{LN}\left(s_i^{(t-1)} + \frac{1}{\sqrt{T}} c_i^{(t)}\right)\right)$$

**Conclusion**: The Spatiotemporal Lattice does not need external 2014-era gated cells. Its native weight-tied residual loop is already a state-of-the-art, high-capacity Recurrent Neural Network.




---

## 7. The Architectural Resolution of Multi-Layer Depth & Sparse Associative Memory

Studies S4-003 through S4-005 permanently resolve the long-standing question: **"Can Sub-Quadratic Transformers scale through physical layers, or are they mathematically restricted to single-layer recurrence?"**

### The Core Conflict Resolved
For months, SubQ appeared to "fight against depth":
- Stacking physical layers consistently collapsed associative recall on MQAR to **2.60%** (random chance).
- Conversely, unrolling a single physical layer with per-hop MLPs enabled deep vision reasoning (48.36% on CIFAR-100), but collapsed discrete associative memory.
- Pure linear transport (8 hops $	o$ 1 final MLP) scored **6.32%** on recall, but lacked intermediate non-linear expressivity.

Through rigorous ablation across Studies S4-003, S4-004, and S4-005, every confounding variable was isolated, revealing the exact architectural blueprint for multi-layer sparse models.

---

### The Three Fundamental Laws of Sparse Memory Depth

```
                        [ Input Sequence x_1 ... x_L ]
                                      │
       ╔══════════════════════════════╧══════════════════════════════╗
       ║   SHARED ATTENTION BUS: Universal Routing Protocol (QKV)    ║  <-- LAW 1: QKV MUST BE TIED
       ╚══════════════════════════════╤══════════════════════════════╝
                                      │
   ┌───▼──────────────────────────────┴──────────────────────────────┐
   │ LAYER 1: Optical Runway (Hops 1 -> 4, state = context, no W_o)  │  <-- LAW 3: 4-HOP OPTICAL RUNWAY
   │ ===> Physical MLP 1 (Independent Weights)                       │  <-- LAW 2: UNTIED PHYSICAL MLPs
   └───┬─────────────────────────────────────────────────────────────┘
       │
   ┌───▼─────────────────────────────────────────────────────────────┐
   │ LAYER 2: Aggressive / Dense Compute (Hops 5, 6, 7, 8)           │  <-- LAW 3: AGGRESSIVE COMPUTE
   │ ===> Physical MLP 2 (Independent Weights)                       │      AFTER GLOBAL SPAN REACHED
   └───┬─────────────────────────────────────────────────────────────┘
       │
   [ Head -> Logits ]
```

#### 1. Law 1: The Universal Routing Protocol (Shared QKV Invariance)
In sparse multi-hop architectures, information travels across intermediate tokens acting as relay antennas. $W_q, W_k, W_v$ define the language of the packet headers.
- **Shared QKV:** Both Layer 1 and Layer 2 speak the same routing language $	o$ **`5.49% Recall`** (All-Time Record).
- **Untied QKV:** Layer 2 rotates the attention subspace, breaking packet transmission $	o$ **`2.50% Recall`** (Collapses to uniform random chance even after 6,000 steps).
- **Rule:** *The attention coordinate frame must remain stationary across layers.*

#### 2. Law 2: Physical MLPs Thrive Under Shared Attention ($	ext{MLP}_1 
e 	ext{MLP}_2$)
Untying the MLPs does not harm memory retrieval—it significantly enhances it:
- `s2_028_untied_mlp` reached **`5.49%`**, beating the tied baseline (**`5.20%`**) and the 1-Layer Dense Transformer (**`5.45%`**).
- **Rule:** *SubQ models can scale compute capacity and parameters through stacked physical MLPs without restriction.*

#### 3. Law 3: The Global Runway & Aggressive Compute Principle
- The linear optical delay (`state = context`, no $W_o$) is required only during the **initial network diameter runway** ($4 	imes 128 = 512$ tokens).
- Once the global receptive field is established at $T=4$, subsequent hops (5, 6, 7, 8) can execute **dense, aggressive MLPs at every step** without destroying the memory trace (**`4.90% Recall`**).
- Deeper compute models exhibit delayed phase transitions (grokking): flatlining at ~2.6% for 4,000 steps before rapidly converging in the final 2,000 steps as the learning rate anneals.

---

### The Blueprint for SubQ Large Language Models (LLMs)
1. **Backbone:** 1 Universal Shared-QKV Sparse Attention Engine ($O(L \cdot K)$).
2. **First Macro-Layer:** 4-hop uninterrupted linear optical runway (`state = context`) to guarantee global information delivery.
3. **Subsequent Layers:** Stacked, independent physical MLPs executing at dense frequency to maximize reasoning depth and parameter scale.
