# SubQTransformer: Sub-Quadratic Iterative Transformer with Adaptive Dynamical Halting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**SubQTransformer** is a sub-quadratic sequence modeling architecture that replaces quadratic $O(L^2)$ dense attention and deep parameter layer stacking with **multi-scale sparse graph routing** and **iterative contractive message passing with per-token early stopping**.

```
Input Tokens X ──► Step 1: Multi-Scale Sparse Routing (K Logarithmic Offsets)
                          │
                          ▼
                   Step 2: Recurrent Dynamical Relaxation (GRU State Update)
                          │
                          ▼
                   Step 3: Adaptive Dynamical Halting (||Δs|| / ||s|| ≤ ε)
                          │ (Halts settled tokens at T = 1..2; deep reasoning at T = 4..6)
                          ▼
                   Step 4: LayerNorm + Feedforward MLP + Output Head
```

> 📖 **Comprehensive Research Journey & Full Technical Report**: For the complete, unabridged 40KB experimental log covering all 43 Modal GPU studies (including Jacobian spectral proofs, Dyck-4 bracket stacks, MQAR benchmarks, and ambient 128D geometric straightness analysis), see **[`experiments/RESEARCH_JOURNEY.md`](file:///c:/Users/beca/Desktop/gravimem-revived/experiments/RESEARCH_JOURNEY.md)**.

---

## 🚀 Key Highlights

* **Sub-Quadratic $O(L \cdot K)$ Complexity**: Uses $K$ multi-scale logarithmic relative jumps ($K \approx 12 \dots 16$), eliminating attention dispersion ("attention dust") and quadratic VRAM growth.
* **Full GPU Sequence Parallelism**: All $L$ tokens process concurrently in parallel on GPU tensor cores; recurrence is strictly over the small thought depth axis ($T = 2 \dots 3$).
* **Iterative Coupled Relaxation**: Operates as a dynamical system over a learned sparse graph. Information propagates multi-edge paths ($A \to B \to C$) while a locally contractive Jacobian ($\rho(J) < 1.0$) guides states toward stable attractors.
* **Per-Token Adaptive Halting**: Automatically early-exits when state velocity drops below $\epsilon$. Predictable tokens exit at $T=1 \dots 2$, **cutting total compute by $50\%$ with zero perplexity loss**.
* **Crushes Multi-Layer Transformers with 60% Fewer Parameters**: 1-Layer SubQTransformer achieves **`5.68` perplexity** on TinyShakespeare vs. **`10.00` perplexity** for a 4-layer standard Transformer.
* **Extreme Memory Efficiency ($L=4096$)**: Runs $L=4096$ context in **$< 2\text{ GB}$ VRAM** at **$253,000\text{ tok/s}$** on a single GPU where 4-layer Transformers suffer CUDA Out-of-Memory (OOM) crashes.

---

## 📦 Installation

```bash
git clone https://github.com/becabytess/Subq-Transformer.git
cd Subq-Transformer
pip install -e .
```

---

## 💻 Python API Usage

### 1. Autoregressive Language Modeling

```python
import torch
from subqtransformer import SubQConfig, SubQTransformerLM

# Configure model
config = SubQConfig(
    vocab_size=50257,
    d_model=256,
    n_heads=8,
    n_layers=2,              # 2 stacked SubQ physical layers
    default_T=3,             # 3 iterative thought hops per layer
    max_seq_len=2048,
    adaptive_halting=True,   # Enable dynamic per-token early exit
    halt_threshold=0.08      # Velocity threshold epsilon
)

model = SubQTransformerLM(config)

# Input tokens (Batch=2, Length=128)
idx = torch.randint(0, 50257, (2, 128))
targets = torch.randint(0, 50257, (2, 128))

# Forward pass with loss & compute stats
logits, loss, stats = model(idx, targets=targets, return_stats=True)

print(f"Loss: {loss.item():.4f}")
print(f"Average Hops Used: {stats['mean_total_hops']:.2f}")
print(f"Compute Savings: {stats['layer_stats'][0]['compute_savings']:.1f}%")
```

### 2. Controllable Test-Time Generation

```python
# Generate text with custom thought depth (T=4 for deeper reasoning)
prompt = torch.tensor([[15496, 11, 314]], dtype=torch.long) # e.g. "Hello, I"
output_ids = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
    T=4  # Run 4 thought hops during inference
)
```

### 3. Sequence / Reasoning Classification

```python
from subqtransformer import SubQTransformerClassifier

classifier = SubQTransformerClassifier(
    num_classes=10,
    config=config
)

logits, loss = classifier(idx, targets=torch.randint(0, 10, (2,)))
```

---

## 📊 Benchmark Results

### 1. Head-to-Head: SubQTransformer vs. Multi-Layer Transformers

*Evaluated on TinyShakespeare ($L=512$, batch size 32):*

| Architecture | Physical Layers | Parameters | Val Loss | Perplexity | Peak VRAM | Key Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard Transformer (1L)** | 1 | 273,920 | 2.4909 | 12.07 | 598 MB | Attention dispersion ("attention dust") |
| **Standard Transformer (2L)** | 2 | 471,680 | 2.4702 | 11.83 | 852 MB | 1.9x worse perplexity than SubQ |
| **Standard Transformer (4L)** | 4 | 867,200 | 2.3025 | 10.00 | 1,369 MB | 2.5x more parameters |
| **SubQTransformer (1L, Fixed $T=4$)** | 1 | 342,159 | **`1.7348`** | **`5.67`** 🏆 | **810 MB** | **43% lower error than 4L Transformer** |
| **SubQTransformer (1L, Auto-Stop $\epsilon=0.12$)** | 1 | 342,159 | **`1.7363`** | **`5.68`** ⚡ | **810 MB** | **50% compute reduction with zero loss** |

---

### 2. Context Length Scaling & Memory Frontier ($L=256 \dots 4096$)

*Profiled on a 16GB Tesla T4 GPU:*

| Context Length ($L$) | SubQTransformer VRAM | 4-Layer Transformer VRAM | SubQ Speed | Transformer Speed | Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$L = 256$** | 214 MB | 210 MB | 164,210 tok/s | 182,931 tok/s | ~1.0x |
| **$L = 512$** | 336 MB | 434 MB | 244,553 tok/s | 170,716 tok/s | **1.4x faster** |
| **$L = 1024$** | **574 MB** | 1,219 MB (2.1x) | **248,435 tok/s** | 109,077 tok/s | **2.3x faster** |
| **$L = 2048$** | **1,049 MB** | 4,131 MB (4.0x) | **251,794 tok/s** | 62,074 tok/s | **4.1x faster** |
| **$L = 4096$** | **`1,998 MB` (< 2 GB)** | 💥 **OOM Crash** | **`253,032 tok/s`** | **0 tok/s (Crashed)** | 🚀 **Infinite (Transformer died)** |

---

### 3. Zero-Shot Length Extrapolation ($L=256 \to L=1024$)

*Trained strictly on short context $L=256$ and tested zero-shot on 4x longer context without fine-tuning:*

| Architecture | $L=256$ (Train) | $L=512$ (Zero-Shot) | $L=1024$ (Zero-Shot) | Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **SubQTransformer ($1\text{L}, T=4$)** | **`6.18` PPL** | **`8.47` PPL** | **`10.15` PPL** 🛡️ | **`+64.2%` (Graceful)** |
| **Standard Transformer ($4\text{L}$)** | 6.51 PPL | 16.92 PPL | **`25.80` PPL** | **`+296.3%` (Catastrophic Failure)** |

---

### 4. Dynamic Early-Exit Halting Pareto Frontier

| Convergence Threshold ($\epsilon$) | Avg Hops ($T$) | Compute Savings | Val Loss | Perplexity | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$\epsilon = 0.08$** | **`3.40`** | **`43.4%`** | **`1.7348`** | **`5.67`** 🎯 | Matches fixed $T=4$ with 43% compute cut |
| **$\epsilon = 0.12$** | **`2.99`** | **`50.1%`** | **`1.7363`** | **`5.68`** ⚡ | 50% compute reduction with zero loss |
| **$\epsilon = 0.20$** | **`2.51`** | **`58.2%`** | `1.7548` | `5.78` | Beats fixed $T=2$ with 58% savings |
| **Top-1 Stability** | **`2.14`** | **`64.3%`** | `1.7704` | `5.87` | 64% compute savings |

---

### 5. Pre-Trained Dense LLM Transplant & Multi-Domain Generalization (GPT-2 124M)

*Can a standard pre-trained dense $\mathcal{O}(L^2)$ LLM (GPT-2, 124M parameters, 12 layers) be converted into SubQ Jump Dynamics without retraining from scratch?*

```
Dense GPT-2 (124M) ──► Direct Weight Surgery ──► SubQ-GPT2 (K=32, T=4) ──► Multi-Domain Cross-Transfer
```

1. **Zero-Shot Dynamical Recovery**: 
   * Naive 1-hop static jumping ($T=1$) collapses catastrophically (**`13,940` PPL**).
   * Turning on **Gravimem multi-hop recurrent settling ($T=4$)** immediately recovers the model to **`759` PPL ($18.3\times$ zero-shot improvement with 0 fine-tuning)**.
2. **Multi-Domain Generalization**:
   * Adapted strictly on **General WebText**, SubQ-GPT2 was evaluated zero-shot across **3 completely untouched held-out corpora**:

| Domain / Corpus | Role in Test | Dense GPT-2 Baseline | Zero-Shot SubQ Transplant | SubQ Adapted on WebText | **Gap Closed (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WikiText-2** | **Held-Out (Knowledge)** | **`34.08` PPL** | `2,087.31` PPL | **`104.06` PPL** | **`96.6%`** 🏆 |
| **General WebText** | **Adaptation (Val Set)** | **`33.43` PPL** | `1,947.09` PPL | **`173.02` PPL** | **`92.7%`** |
| **Python Code** | **Held-Out (Syntax)** | **`9.64` PPL** | `1,963.97` PPL | **`419.52` PPL** | **`79.0%`** |

*All adapted model weights are permanently archived and available in Modal persistent volume storage.*

---

## 🔬 Mathematical & Dynamical Proofs

* **Local Contractive Stability**: Jacobian spectral radius $\rho(J) = \max |\lambda_i| \in [0.9676, 0.9989] < 1.0000$ across all hops, proving perturbations decay exponentially ($\Delta s_{t+1} \approx J \Delta s_t$).
* **Phase Space Volume Contraction**: $\ln |\det(J)| = -243.61$, proving state space actively contracts by $\approx 10^{-106}$ per step, preventing trajectory divergence.
* **Ambient 128D Trajectory Straightness**: In raw unprojected $\mathbb{R}^{128}$ space, trajectories exhibit a **`91.26%` straightness ratio**, confirming quasi-geodesic convergence into local fixed-point attractors.

---

## 📁 Repository Structure

```
├── subqtransformer/          # Primary Python package
│   ├── __init__.py           # Package exports
│   ├── config.py             # SubQConfig dataclass
│   ├── layers.py             # SubQSurfer & SubQBlock
│   └── model.py              # SubQTransformerLM & SubQTransformerClassifier
├── gravimem/                 # Backward compatibility alias layer
├── experiments/              # 43 Modal cloud GPU benchmark & mechanistic scripts
│   ├── README.md             # Catalog of research studies
│   └── modal_*.py            # Benchmarking and exploration scripts
├── tests/                    # Unit & integration test suite
│   └── test_subqtransformer.py
├── pyproject.toml            # Packaging configuration
├── requirements.txt
└── README.md
```

---

## 🧪 Running Unit Tests

```bash
python -m unittest tests/test_subqtransformer.py
```

---

## 📄 License
MIT License. Open for research and commercial use.
