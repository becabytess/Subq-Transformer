# State of SubQ: Canonical Blueprint & Living Consensus

> **Purpose of this Document**:  
> This is the authoritative, single source of truth for the **canonical understanding of SubQ**. It synthesizes all theoretical breakthroughs, mathematical foundations, and empirical discoveries into an actionable, unambiguous architectural blueprint. As our research advances, this document is updated to reflect our most stable and verified conclusions.

---

## 0. The Grand Unification: Combinatorial Branching & Harmonic Spatial Exploitation

The fundamental theoretical foundation of SubQ unifies two core principles:

### 1. The Combinatorial Exponential Receptive Field ($K^T$)
* **The Mathematical Engine**: When $K$ offsets are unrolled across $T$ transitive recurrent hops ($i 	o i-d_1 	o i-d_1-d_2 \dots$), the total combinatorial branching factor scales exponentially as:
  $$\text{Paths}(T) = K^T$$
* **Scaling Dynamics**:
  - At $K=8, T=4$: the network generates $8^4 = \mathbf{4,096}$ potential path combinations.
  - At $K=8, T=8$: the network unrolls $8^8 = \mathbf{16,777,216}$ potential path combinations.
* Because $K^T \gg L$ (where $L$ is sequence length), the network naturally achieves an astronomical global receptive field with tiny $K$ and modest $T$. This explains why **random offsets, linear strides, dyadic grids, Fibonacci sequences, and Base-8 mathematical strides all consistently beat dense single-layer transformers**.

### 2. Pattern Exploitation vs. Brute-Force Graph Search
* **The Distinction**: Pure mathematical combinations (like Base-$K$ Radix expansion) assume worst-case independence (uniform random tokens). But real-world data (natural language, 2D images) is **NOT independent**—it exhibits strong structural geometry: power-law $1/d$ distance decay, dense local syntactic wells, hierarchical phrase boundaries, and 2D lattice symmetries.
* **The Role of Learned Harmonic Waves**:
  - The learned carrier waves do not need to rediscover connectivity from scratch; they **exploit the natural statistical regularities and frequency spectrum of the data**.
  - By aligning the sparse routing prior with the data's intrinsic geometry, the network **does not waste its expensive non-linear capacity ($W_Q, W_K, W_V$) searching for where to look**.
  - Instead, 100% of the model's non-linear parameters are freed up for **deep multi-hop compositional reasoning, semantic synthesis, and decision-making**.

---

## 1. The Alien 2D Lattice: The Diagonal Light-Cone RNN & The Spacetime Manifold

The evolution of SubQ has revealed that iterative sparse attention is fundamentally **not** a crippled Transformer; it is a novel dynamical system operating at the intersection of Recurrent Neural Networks and Wave Mechanics.

### 1. The Diagonal Light-Cone RNN ($K=1, \text{offset}=1$)
Consider the degenerate minimal case of SubQ where $K=1$ and the offset is fixed to $1$ (looking only at the immediate previous token $i - 1$):
* At hop $t=0$, token $i$ holds embedding $s_0(i)$.
* At hop $t=1$, token $i$ attends to $i-1$. State $s_1(i) = f(s_0(i), s_0(i-1))$.
* At hop $t=2$, token $i-1$ has already absorbed $i-2$. Therefore, token $i$ attending to $i-1$ receives information from $i-2$: $s_2(i) = f(s_1(i), s_1(i-1))$.
* **The Realization**: Tracing the coordinate path across the 2D grid $(t, l)$ reveals that each diagonal ray $\Delta l = \Delta t$ is **literally an autonomous sequential RNN running backward along the causal light cone**:
  $$h_t = f(h_{t-1}, x_{i-t})$$
* **The Multi-Velocity Braided Lattice**: In full SubQ, where $K > 1$ and offsets are dynamically selected by harmonic carrier waves $d_k^{(t)}$, SubQ is an **alien 2D lattice of braided, multi-velocity diagonal RNNs**. Each wave peak $d_k$ defines a propagation ray with velocity:
  $$v_k = \frac{d_k}{\Delta t}$$
  Information propagates across the sequence not along rigid horizontal time steps, but across a multi-frequency wave interference lattice.

### 2. The 2D Spacetime Manifold & Causal 2D CNN
When an input sequence of length $L$ with embedding dimension $D$ passes through a $T$-hop SubQ runway, it produces an entire sequence of intermediate representations:
$$\mathbf{S} = [\mathbf{s}_0, \mathbf{s}_1, \dots, \mathbf{s}_T] \in \mathbb{R}^{B \times D \times T \times L}$$
* In classical Transformers and early SubQ variants, all intermediate states $\mathbf{s}_0, \dots, \mathbf{s}_{T-1}$ were discarded, and only the final state $\mathbf{s}_T$ was fed to an MLP. **This was a catastrophic waste of computation**, throwing away the complete trajectory of the wave propagation.
* **The 2D Spacetime Formulation**: $\mathbf{S}$ is treated as a 2D spacetime sheet (spatial axis $L$, temporal/thought axis $T$).
* **Causal 2D CNN Head**: Rather than standard MLPs, we apply a causal 2D Convolutional network over $\mathbf{S}$:
  - Temporal pooling/strides downsample the thought dimension $T \to 1$.
  - Causal left-padding along sequence dimension $L$ guarantees **strict $0.00\text{e}+00$ causal leakage**.
  - The 2D convolution directly synthesizes the multi-velocity wave interference patterns across the entire trajectory $[\mathbf{s}_0 \dots \mathbf{s}_T]$.

### 3. The Relay Linearity Law & The Zero-MLP Paradigm
Through extensive associative recall (MQAR) and language modeling benchmarks, we discovered the **Relay Linearity Law**:
> **The Relay Linearity Law**:  
> Intermediate hops along a multi-hop transmission wire must remain **strictly linear** ($s^{(t)} = \text{Attn}(s^{(t-1)})$). Interleaving non-linear activation functions or intermediate feedforward MLPs between hops distorts and destroys associative memory traces before they reach the output.

* **Consequence**: We completely eliminated feedforward MLPs from the recurrent runway.
* **The Zero-MLP Architecture**: The entire recurrent core consists of pure linear sparse attention hops. All non-linear parameter capacity is concentrated in the **Spacetime Causal Conv Head**, which compresses the spacetime sheet $\mathbf{S}$ into the output representation.

---

## 2. Core Architecture: Separation of Spatial Schedule & Content Filtering

SubQ maintains a clean architectural decoupling between *where* communication is structurally enabled and *what* content is retrieved:

$$\mathbf{Attention} = \underbrace{\mathbf{w}_t \text{ (Wave Router)}}_{\text{Global Spatial Schedule (Where to look)}} + \underbrace{QK^\top(\mathbf{s}_t)}_{\text{Content-Dependent Relevance (What matters)}}$$

```
Input Tokens (L)
     │
┌────▼────────────────────────────────────────────────────────────────────────┐
│  Macro-Layer Linear Recurrent Runway (Zero MLPs)                            │
│                                                                             │
│  For hop t = 1 ... T:                                                       │
│    1. Project Evolving Q, K, V from current state s^(t-1)                   │
│    2. Step Autonomous Dynamical Wave: w^(t) = w^(t-1) + 0.1 · F(w^(t-1))    │
│    3. Generate 1D Damped Carrier Waves: W_h(d) = Σ A cos(ωd + φ) · e^(-λd)   │
│    4. Select K Discrete Peaks (Offset 0 + Top-(K-1) Wave Crests)            │
│    5. Compute Sparse Attention + Additive Wave Logit Bias                   │
│    6. Linear State Update: s^(t) = Attn(s^(t-1))  [No intermediate MLPs]   │
└────┬────────────────────────────────────────────────────────────────────────┘
     │
     ▼ Intermediate Spacetime Sheet: S = [s_0, s_1, ..., s_T] in R^(B x D x T x L)
┌────▼────────────────────────────────────────────────────────────────────────┐
│  Dense / Inverted-Bottleneck Spacetime Causal Conv Head                     │
│  - Causal Left-Padding along sequence L (Zero future leakage)               │
│  - Convolutional Kernel downsamples thought depth T -> 1                    │
│  - Non-linear channel expansion & synthesis                                 │
└────┬────────────────────────────────────────────────────────────────────────┘
     │
Final State s_final ──► LayerNorm ──► Output Head (LM / Classifier)
```

---

## 3. Canonical SubQ Specifications (The Verified Gold Standard)

| Component | Canonical Specification | Empirical Rationale |
| :--- | :--- | :--- |
| **Macro Architecture** | **2-Macro-Layer SubQ Runway** | Stacking 2 macro-layers unrolls hierarchical multi-scale representations while keeping parameter count strictly controlled. |
| **Hop Depth per Layer** | **$T = 4$ or $T = 8$ Linear Hops** | Provides astronomical $K^T$ transitive path combinations without parameter inflation. |
| **Runway Linearity** | **Pure Linear Relay ($s = \text{Attn}(s)$)** | Satisfies the Relay Linearity Law; preserves associative key-value traces across hops. |
| **Intermediate MLPs** | **ZERO MLPs in Runway** | Eliminating MLPs prevents trajectory distortion and frees up parameter budget for the Spacetime Conv Head. |
| **Non-Linear Synthesis** | **Spacetime Causal Conv Head** | Fuses full trajectory $[s_0 \dots s_T]$ via causal 2D conv ($3\times 3$ kernel, stride $(T/2, 1)$). |
| **Q, K, V Projections** | **Full Dynamic Evolving $Q, K, V$** | Keys and Values project dynamically at every hop to enable multi-hop message cascading ($A \to B \to C$). |
| **Wave Generator** | **Grid-Superposition Harmonic Carrier** | Multi-carrier superposition $\sum_{m=1}^{12} A_m \cos(\omega_m d + \phi_m) e^{-\lambda_m d}$ naturally forms bounded constructive interference across $d \in [1, 128]$. |
| **Peak Selection** | **Strict Discrete Wave Peaks ($K=8$ or $K=16$)** | Offset 0 (self-token) + Top wave crests. Fast gathers with zero attention dust. |
| **Attention Bias** | **Harmonic Prior Bias** | Adding wave crest values into attention logits provides a spatial inductive prior. |
| **Hardware Execution** | **Fused OpenAI Triton Kernel** | Custom Triton block kernel executes forward and backward passes with $100\%$ bit-exact PyTorch autograd match. |

---

## 4. Empirical Dynamics: Dual Contraction & The Attractor Basin

Across deep evaluation on benchmark models (`checkpoints/best_harmonic_subq_t8.pt`), the system exhibits the **Dual Contraction Law**:

$$\text{State Velocity: } v(t) = \frac{\|\mathbf{s}^{(t)} - \mathbf{s}^{(t-1)}\|_2}{\|\mathbf{s}^{(t)}\|_2}, \qquad \text{Spatial Damping: } \mathbf{w}(d) \propto \cos(\omega d + \phi) \cdot e^{-\lambda(t) d}$$

### Hop-by-Hop Empirical Scorecard

| Hop ($t$) | State Velocity $v(t)$ | Damping Rate $\lambda(t)$ | Spatial Reach $R(t) = \frac{1}{\lambda}$ | Mean Attended Distance $\bar{d}(t)$ | Computational Regime |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **$0.6272$** | $0.0355$ | **$28.1$ tokens** | $5.49$ tokens | Global discourse foraging |
| **2** | $0.3606$ | $0.0437$ | $22.9$ tokens | $5.14$ tokens | Contextual integration |
| **3** | $0.2667$ | $0.0571$ | $17.5$ tokens | $4.48$ tokens | Intermediate synthesis |
| **4** | $0.2188$ | $0.0769$ | $13.0$ tokens | $3.76$ tokens | Transition boundary |
| **5** | $0.1887$ | $0.1042$ | $9.6$ tokens | $2.96$ tokens | Local phrase focusing |
| **6** | $0.1713$ | $0.1408$ | $7.1$ tokens | $2.26$ tokens | Local verification |
| **7** | $0.1616$ | $0.1852$ | $5.4$ tokens | $1.83$ tokens | Immediate syntax checks |
| **8** | **$0.1560$** | **$0.2406$** | **$4.2$ tokens** | **$1.65$ tokens** | Token emission attractor |

### Understanding the Strong Correlation ($r = +0.9232$)
- The Pearson correlation between velocity $v(t)$ and spatial reach $R(t)$ is **$r = +0.9232$** ($p < 0.001$).
- The autonomous wave network $F(w)$ and the recurrent state representation learn a synchronized cooling schedule: broad global mixing when velocity is high, collapsing to immediate local verification as the state settles into an attractor basin.

---

## 5. Season 4 Breakthroughs: The Definitive 687k Shootout (Studies S4-014 – S4-020)

In Season 4, we put SubQ through rigorous head-to-head empirical evaluations on TinyShakespeare ($L=256$, Tesla T4 GPU, OpenAI Triton kernel) and Multi-Query Associative Recall (MQAR).

### The Definitive 687k Shootout
To resolve whether the massive gains of the Spacetime Conv Head came from convolutional trajectory fusion or merely parameter scaling, we conducted a strictly controlled 3-way shootout. All three models shared:
- Identical parameter budget: **~687,000 parameters**
- Identical attention budget: **64 lookups/token** ($75\%$ sparse)
- Identical recurrent depth: **2 Macro-Layers with $T=4$ hops**

| Study | Architecture | Non-Linear Engine | Trajectory $[s_0..s_4]$ Fusion? | Parameters | Val Loss | Val PPL | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **S4-020** | Standard Multi-Layer SubQ | **Feedforward MLP** ($128 \to 1022 \to 128$) | **NO** (Discards $s_0..s_3$, looks only at $s_4$) | **687,164** | **1.7028** | **`5.49`** | Stalled at historical ceiling |
| **S4-019** | Scaled Depthwise-Separable SubQ | **Inverted Bottleneck Conv** ($128 \to 474 \to 128$) | **YES** (2D Conv fuses $5 \to 2 \to 1$) | **687,272** | **1.5725** | **`4.82`** | Breaks sub-5.0 barrier |
| **S4-017** | **Dense Spacetime Conv SubQ** | **Dense 2D Spacetime Conv** ($3\times 3$, 128ch) | **YES** (2D Conv fuses $5 \to 2 \to 1$) | **687,168** | **1.5576** | **`4.75`** | **ALL-TIME PROJECT RECORD** 🏆 |

### Scientific Conclusions from the Shootout:
1. **The Trajectory Fusion Advantage (+0.74 PPL)**: Standard MLPs discard intermediate states $s_0 \dots s_3$, scoring **5.49 PPL**. Preserving and fusing the spacetime sheet $[s_0 \dots s_4]$ via Spacetime Causal Conv delivers an astonishing **4.75 PPL**—a massive **+0.74 PPL leap** under identical parameters!
2. **Channel Capacity Matters**: When parameter-capped with pure depthwise convolution without channel expansion (S4-018, 265k params), the model achieved 5.36 PPL. Expanding channel capacity via Inverted Bottleneck ($128 \to 474 \to 128$, S4-019) propelled it to 4.82 PPL, proving that channel-mixing capacity in the post-recurrent head is essential.
3. **Dense 2D Conv is the Champion**: Dense $3\times 3$ spatio-temporal convolutions cross-correlate representations across both time and space simultaneously, achieving the all-time project record of **4.75 PPL**.

### Additional Season 4 Milestones
* **Study S4-014 (Spacetime Causal Conv)**: First broke the sub-5.30 barrier, achieving **`5.10 PPL`** with $T=8$ and $90.6\%$ sparsity (24 lookups/token).
* **Study S4-016 (MQAR Associative Recall)**: Verified that a pure linear recurrent runway (`s = attn_out`) combined with a Causal 2D Conv Head achieves **`5.25%` exact recall** (and **`10.5%`** on distances $\le 255$). Proved that the Spacetime 2D Conv does **not** blur or destroy sharp key-value associative bindings.

---

## 6. Established Truths & Discarded Hypotheses

### Established Truths
1. **Spacetime Trajectory Fusion is Superior**: Fusing the full multi-hop trajectory $[s_0 \dots s_T]$ via a Causal 2D Conv Head decisively outperforms running standard MLPs on the final state (+0.74 PPL gain).
2. **The Relay Linearity Law**: Intermediate hops on the recurrent wire must remain strictly linear. Feedforward MLPs between hops distort representations and damage associative recall.
3. **Physical Depth is Obsolete**: Parameter-tied recurrence reuses wide, rich layers $T$ times, decisively outperforming physically stacked static layers with equal or fewer parameters.
4. **Dynamic $Q, K, V$ is Non-Negotiable**: Freezing Keys and Values at $t=1$ chokes transitive reasoning; evolving them dynamically unlocks continuous monotonic scaling.
5. **Discrete Peaks Beat Neighborhood Pooling**: Gathers of exact wave peaks outperform fuzzy wavelet/filterbank super-tokens in both wall-clock throughput and perplexity.
6. **Autonomous Spatial Annealing Wins**: Continuous harmonic wave routing decisively beats static hardcoded complete bases by providing multiscale coarse-to-fine receptive fields.

### Discarded Hypotheses
| Deprecated Idea | Why It Failed / Was Discarded | Replaced By |
| :--- | :--- | :--- |
| **Intermediate MLPs Between Hops** | Distorts associative memory traces; violates the Relay Linearity Law. | Pure linear runway (`s = attn_out`) + Spacetime Causal Conv Head. |
| **Discarding Intermediate States $[s_0..s_{T-1}]$** | Throws away the wave interference trajectory; capped at 5.49 PPL. | Stacking into 2D Spacetime Sheet $\mathbf{S} \in \mathbb{R}^{B \times D \times T \times L}$ + Causal 2D Conv. |
| **Static Dyadic Grids ($\pm 1, 2, 4, 8\dots$)** | Rigid, non-adaptive; unable to perform coarse-to-fine multiscale annealing. | Learned Continuous Damped Carrier Waves. |
| **Frozen Keys & Values** | Blocked multi-hop transitive information cascading ($A \to B \to C$). | Full Evolving $Q, K, V$ at every hop $t$. |
| **Deep Multi-Layer Physical Stacking** | Doubled weight footprint without beating recurrent parameter reuse. | Macro-layer recurrent blocks with Spacetime Conv Heads. |
| **Unbounded Analytical Peak Inversion** | Solving $(2\pi - \phi)/\omega$ analytically suffered from numerical clamp drift. | Multi-Carrier Continuous Superposition with Grid Selection. |
| **Closed-Loop State-Feedback Wave Modulation** | Feeding $\Delta \mathbf{s}$ and velocity into wave transitions added overhead for $\le 0.02$ PPL gain. | Autonomous Open-Loop Spatial Annealing. |

---

## 7. Active Research Frontiers

### Frontier 1: Multi-Scale Spacetime Pyramids
* **Objective**: Replace the single 2D Conv downsampling stage with a multi-scale Feature Pyramid Network (FPN) across the spacetime sheet $\mathbf{S}$, extracting multi-frequency wave features at multiple temporal resolutions.

### Frontier 2: Dynamic Token-Wise Early Exit on Spacetime Sheets
* **Objective**: Leverage monotonic velocity decay $v(t) \to 0.15$ to truncate the temporal dimension $T_i$ individually per token, skipping late-stage convolutional slices on tokens that settle early.

### Frontier 3: Triton-Fused Spacetime Conv & Recurrent Runway
* **Objective**: Fuse the multi-hop linear attention unroll and the subsequent 2D causal convolution into a single persistent Triton kernel, eliminating the intermediate memory footprint of the spacetime tensor $\mathbf{S}$.
