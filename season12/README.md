# The Recurrent Lattice: A Generalized Framework Unifying Recurrence and Attention

---

## 1. Overview: Opening the Design Space

In modern machine learning, sequence models are almost universally divided into two distinct families:
1. **Recurrent Models & State Space Models (RNNs, SSMs, Mamba):** Process information sequentially along the time dimension, maintaining a hidden state that updates token by token.
2. **Attention Models (Transformers):** Compute pairwise relationships across the entire sequence in an all-to-all graph.

Because these two paradigms are derived from very different mathematical roots, they are usually studied in isolation. 

**The Recurrent Lattice** (or **The Lattice**) is a conceptual framework that views recurrence and attention not as opposing paradigms, but as two extreme configurations of a single, generalized sequence operator. 

Rather than proposing a single static model, The Lattice introduces a set of continuous architectural knobs—the spatial lookback budget ($K$), the temporal hop depth ($T$), the offset geometry, and the state update function. By adjusting these knobs, the framework naturally simplifies to a standard Linear RNN on one end, full dense attention on the other, and opens up an unmapped spectrum of hybrid multi-scale architectures in between.

---

## 2. The Core Problem: Observations vs. Markov States

To understand what The Lattice is doing, it helps to start from a simple probabilistic intuition: **sequence generation as a Partially Observable Markov Decision Process (POMDP).**

* **Raw tokens are impoverished observations:** A single token $x_t$ viewed in isolation contains almost no predictive power. A word like `"charge"` could refer to an electrical property, a legal accusation, a military maneuver, or a financial cost. 
* **The role of the model:** The model's job is to transform this sequence of ambiguous observations $x_1, \dots, x_t$ into a rich **Sufficient Statistic**—a hidden Markov state $S_t$ that condenses the relevant context so that the future depends only on $S_t$:
  $$\mathbb{P}(x_{t+1} \mid x_1, \dots, x_t) \approx \mathbb{P}(x_{t+1} \mid S_t)$$

---

## 3. Two Ways of Constructing States

How can a model construct this sufficient state across $L$ tokens?

### Perspective A: Sequential Construction (The Standard RNN)
A classical RNN constructs the state **strictly from left to right**:
* Token 0 creates state $h_0$.
* Token 1 reads $h_0$ and combines it with its own input to produce $h_1$.
* Token 2 reads $h_1$ and produces $h_2$, and so on.
* **The Mental Picture:** Imagine 512 people walking in a strict single-file line through the dark. Person 1 tries to describe the terrain to Person 2, who whispers it to Person 3. Each person only hears from the single person directly ahead of them.

### Perspective B: Collective Parallel Relaxation (The Lattice)
Instead of perfecting one state at a time along a sequential chain, The Lattice **describes the entire board at once and enriches all states simultaneously in parallel**:
* **Hop 0 (Micro Initialization):** Every position starts with a cheap, local estimate of its immediate surroundings.
* **Hop 1 (Local Sharing):** Every position looks back at nearby positions and incorporates their local states. The entire sequence becomes slightly more informed together.
* **Hop 2 & 3 (Multi-Scale Expansion):** Tokens inspect positions at larger strides, folding regional and global context into their representations.
* **The Mental Picture:** Imagine all 512 people standing abreast in a unified grid, taking small, coordinated steps forward together. Nobody has the full picture at step 0, but by continually communicating and relaxing across multiple scales, the entire line converges toward a clear global representation simultaneously.

---

## 4. The Vector Field & Collective Denoising

Because every position in The Lattice maintains an evolving state at every hop, the sequence forms a **2D State Vector Field**:

* **Consensus over Isolation:** In a 1D chain, an error or noisy vector at token 15 must be carried forward directly through tokens 16, 17, and beyond. In a parallel lattice, every token is surrounded by neighboring states evolving in parallel. The collective momentum of the field acts as a natural stabilizer, helping smooth out localized noise.
* **A Predictable Denoising Trajectory:** Instead of relying on a single vector to carry all historical memory across hundreds of steps, the network can perceive the net direction in which the sequence states are evolving. Each hop acts like a relaxation step moving the entire sequence closer to a globally consistent manifold.

---

## 5. The State Fusion Rule

To make this multi-hop relaxation behave like a proper recurrent system, the state update at each position must follow a fundamental principle:

$$\mathbf{h}_i^{(t)} = f\left(\mathbf{h}_{\text{incoming}}^{(t-1)}, \; \mathbf{x}_i^{(\text{own})}\right)$$

Every position requires two inputs at each hop:
1. **The incoming state ($\mathbf{h}_{\text{incoming}}$):** The message arriving from the positions looked at during the hop.
2. **The local observation anchor ($\mathbf{x}_i^{(\text{own})}$):** The token's own original identity.

If a token simply replaces its representation with what arrives from other tokens, it acts like a shift register—passing information along but forgetting who it was. By explicitly fusing the incoming historical belief with its own local anchor, each position maintains its identity as an active query while progressively accumulating surrounding context.

---

## 6. The Continuum: Connecting Known Architectures

The Lattice provides a continuous spectrum connecting different sequence modeling families through a unified set of knobs:

```
                      THE CONTINUOUS SEQUENCE SPECTRUM

       LEFT BOUNDARY                                     RIGHT BOUNDARY
    (Pure Linear RNN / SSM)          THE LATTICE         (Full Transformer)
  K = 1, Offset = [1], T = L     1 < K << L, T = log(L)    K = L, T = 1
  ─────────────────────────     ───────────────────────  ───────────────────
  • Softmax vanishes (1.0)      • Competitive Softmax    • Global Softmax
  • 1D Diagonal Wavefront       • Multi-Scale Mesh       • All-to-All Graph
  • 1 token lookback per hop    • Structured Strides     • All tokens looked at
```

### The Left Boundary: Standard Linear Recurrence
* When the lookback budget is set to $K = 1$, and each token looks strictly at the token immediately before it ($\text{offset} = [1]$):
  - The attention softmax over a single score collapses to a pure scalar: $\text{softmax}([s]) = 1.0$.
  - Information flows strictly along the diagonal wavefront: $(pos - 1, hop - 1) \to (pos, hop)$.
  - The update equation $\mathbf{h}_i^{(t)} = \mathbf{A} \mathbf{h}_{i-1}^{(t-1)} + \mathbf{B} \mathbf{x}_i$ becomes mathematically equivalent to a standard Linear RNN / State Space Model.

### The Right Boundary: Standard Full Attention
* When the lookback budget is expanded to cover all past tokens ($K = L$) in a single hop ($T = 1$):
  - Softmax normalizes across all available positions simultaneously.
  - Every token directly inspects every other token in one all-to-all step.
  - This reproduces the standard Dense Self-Attention mechanism of the Transformer.

### The General Interior: Exploring New Hybrids
The value of a generalized framework is that it reveals the vast, unexplored territory between these two extremes:
* What happens when tokens look at $K = 4$ or $K = 8$ offsets using geometric strides ($B^t$)?
* How does the behavior change when using linear, harmonic, or learned dynamic offsets?
* What update functions $f(\mathbf{h}_{\text{incoming}}, \mathbf{x}_{\text{own}})$ provide the best trade-off between expressive capacity and training stability?

---

## 7. Summary

The Recurrent Lattice provides a unified mathematical formulation connecting recurrence and attention. By treating sequence modeling as a **parallel multi-scale state relaxation process**, it uncovers a rich, continuous design space between 1D sequential recurrences and all-to-all attention.
