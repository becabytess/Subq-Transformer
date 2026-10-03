# Season 12 Research Log: The Grand Unification

This document provides the chronological record of empirical investigations, architectural formulations, and experimental breakthroughs for **Season 12**.

---

## 📜 Study S11-Bridge: The Diagonal Recurrence & State Fusion Breakthrough
* **Context:** Following the collapse of ad-hoc Mamba gating in S11-002 and S11-003, we reduced SubQ to its minimal base skeleton: $K = 1, \text{offset} = [1]$.
* **The Mathematical Discovery:**
  1. **Vanishing Softmax:** When $K = 1$, $\text{softmax}([s]) \equiv 1.0$. The attention score completely disappears, leaving a pure linear transfer.
  2. **The Diagonal Wavefront:** When every token looks at offset 1 across hops $t$, information naturally moves along the diagonals: $(pos - 1, hop - 1) \to (pos, hop)$. At hop $t$, position $t$ holds the complete prefix history from token 0.
  3. **The Shift-Register Diagnosis:** Direct replacement ($x = \text{attn\_out}$) failed because token $i$ replaced its state with the incoming token, dropping its own input $x_i$ and losing its query identity. A true RNN requires fusing the incoming state with the current token's input:
     $$h_{\text{next}} = f(h_{\text{incoming}}, x_{\text{own}})$$
  4. **The Grand Unification Spectrum:**
     - $K = 1, \text{offset} = [1], T = L \implies$ Classical Linear RNN / Mamba.
     - $K = L, T = 1 \implies$ Dense Transformer Multi-Head Attention.
     - $1 < K \ll L, T = \log_B(L) \implies$ The Sub-Quadratic Master Interior (SubQ).

---

## 🎯 Season 12 Studies & Progress

| Study | Title | Architecture | Core Finding | Status |
|---|---|---|---|---|
| **S12-002** | Spatial Relaxation & Deafness Horizon | $K=1, \text{offset}=1, T=64$ ($\tanh$ update) | PPL 5.53. Velocity froze by hop 24. Proved deafness horizon at $d \approx 30$ ($0.70^d$ decay). | Completed |
| **S12-003** | Binary Doubling Lattice | $K=1, \text{offsets}=[1,2,4,8,16,32], T=6$ | PPL 6.72. Solved gradient reach (12M× stronger), but broke local contiguous syntax via dilated leaps. | Completed |
| **S12-004** | Cyclical Binary Doubling | 3 Cycles of $[1,2,4,8,16,32]$ ($T=18$) | PPL 7.04. Turbulent velocity oscillations; resetting offsets caused representation shocks. | Completed |
| **S12-005** | Overlapping Wavefront & Similarity Mixing | Stride-1 ($T=32$), content-aware convex blend ($W_q, W_k$) | PPL 5.66. Nearly matched S12-002 in half the hops ($T=32$). Monotonic velocity cooling. Learned 8-token spelling gate ($g \approx 0.06$). | Completed |
| **S12-006** | Pure Parameter-Free Cosine Wavefront | Stride-1 ($T=32$), pure geometric cosine gating | PPL 5.99 with 173k params. Discovered negative cosine repulsion ($\cos \approx -0.48$) preventing representational collapse. | Completed |
| **S12-007** | Parameter-Handicapped Enriched Lattice | Stride-1 ($T=64$), 128 scalar channel gate | PPL 5.46 with 182k params (-8k params vs baseline). 5% static momentum accelerated freezing to Hop 20. | Completed |
| **S12-008** | 3-Way Recurrent Fusion Lattice | Stride-1 ($T=64$), 1-line learned weighted fusion | PPL 5.45 with 190,144 params (-64 vs baseline). Discovered 42%/31%/27% Triad Law. | Completed |
| **S12-009** | Causal Rolled-Escrow Lattice | Stride-1 ($T=64$) + Causal Rolled Escrow Accumulator | **ALL-TIME SEASON SOTA: PPL 5.41** (Val 1.6882, Train 1.4560). Small Intestine principle completely verified. 187,948 params (-2.2k vs baseline). | Completed |
| **S12-010** | Pure Canonical RNN + Direct Digestion Escrow | Pure S12-002 RNN + Direct $h_i$ Escrow Diffusion | **NEW ALL-TIME RECORD: PPL 5.40** (Val 1.6860). 1-Billion-fold gradient reach jump at $d=60$ ($2.91 \times 10^{-9}$). 189,908 params (-300 vs baseline). | Completed |
| **S12-011** | Canonical FEN-Lattice | Pure S12-002 RNN + Dynamic Roll Gate + Lossless Additive Residual | **FULL AUDIBILITY HORIZON $d=60$!** PPL 5.41. Gradient at $d=60$ reached $1.42 \times 10^{-3}$ ($500,000\times$ vs S12-010). Eliminates decay bug. 189,909 params (-299 vs baseline). | Completed |
| **S12-012** | Hop-Escrow Lattice (Vertical Digestion) | 2-term tube ($h_{i-1}, x_i$) + 3rd term ($h_i$) absorbed into vertical escrow | PPL 5.81. **Pathology Diagnosed:** Vertical escrow caused norm explosion (71.36) by accumulating frozen attractor 44×. Audibility shrank to $d=16$. Proves escrow MUST be horizontal across space. | Completed |
| **S12-013** | Dual Cross-Symmetric Recurrent Lattice | 4-way pairwise balance: $h_{i-1}, h_i, x_{i-1}, x_i$ | **ALL-TIME SEASON SOTA: PPL 5.26!** Val Loss plunged to 1.6605 (-0.0255 nats). Discovered 52% Left / 48% Right Tetrad Law. 190,080 params (-128 vs baseline). | Completed |
| **S12-014** | 4-Way Cross-Symmetric Shootout Tournament | Shootout across 5 fusion candidates: Flat Sum, Obs Gating, Two-Stream Gated, Differential, Residual Highway | **TOURNAMENT WINNER: Cand 2 (Two-Stream Gated, PPL 5.27, Val 1.6622)** with $10^7\times$ higher gradient reach at $d=60$ ($2.31 \times 10^{-15}$ vs $4.52 \times 10^{-22}$ for Flat Sum). Flat Sum achieved PPL 5.32 in fastest time (92.7s). | Completed |
| **S12-015** | Fast Two-Stream & Zero-Weight Tournament | Flat Sum vs Zero-Weight Diffusion/Tanh vs Fast Two-Stream 50/50 | **Flat Sum maintains Val Loss crown (1.6605).** Zero-weight uniform averaging suffered blur-collapse (PPL 6.41). Fast Two-Stream 50/50 ran in **69.4s** (fastest parameterized model) and yielded **$10^9\times$ gradient reach at $d=60$ ($9.88 \times 10^{-12}$)**. | Completed |















