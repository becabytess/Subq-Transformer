# SubQTransformer Empirical & Mechanistic Research Suite

> **Full Detailed Research Log & Discoveries**: For the complete chronological research report, exhaustive benchmark logs, Jacobian spectral proofs, and ablation studies, please refer to **[`RESEARCH_JOURNEY.md`](file:///c:/Users/beca/Desktop/gravimem-revived/experiments/RESEARCH_JOURNEY.md)**.

This directory contains all 43 dedicated research, benchmark, and mechanistic study scripts executed on Modal cloud GPU infrastructure during the research and discovery phase of SubQTransformer.

## Catalog of Studies

### 1. Mechanistic & Dynamical Systems Suite
- `modal_mech1_dense_approximation.py`: Measures hidden trajectory cosine alignment and KL divergence vs dense Transformers.
- `modal_mech2_dynamic_vs_static_routing.py`: Proves static graph freezing ($\pi^{(1)}$) achieves identical perplexity to recomputing topology.
- `modal_mech3_attractor_perturbation.py`: Tests perturbation recovery and basin stability under Gaussian noise injection.
- `modal_mech4_jacobian_spectral_radius.py`: Computes exact local Jacobian $J = \partial s^{(t+1)} / \partial s^{(t)}$, proving contractive dynamics ($\rho(J) < 1.0$) and phase-space contraction ($\ln |\det(J)| = -243.6$).
- `modal_mech5_fixed_graph_message_passing.py`: Ambient 128D space trajectory straightness (91.3%) and minimal-pair agreement resolution on frozen graphs.
- `modal_mech6_message_passing_depth_vs_width.py`: Iso-FLOP frontier proving balanced iterative passing ($K=16, T=2$) beats flat wide lookup ($K=32, T=1$).
- `modal_mech7_k_vs_t_scaling_law.py`: 2D $(K, T)$ compensation grid sweep evaluating multi-hop over-squashing bottlenecks.
- `modal_attractor_steering_and_multistability.py`: Monte Carlo basin multiplicity sweep & quantitative directional steering.
- `modal_attractor_text_steering.py`: Zero-weight-update runtime text generation steering via vector bias inoculation.

### 2. Interpretability & Representation Probing
- `modal_interp1_language_jump_profile.py`: Analyzes relative jump distance utilization and syntactic attention distributions.
- `modal_interp2_gru_gate_dynamics.py`: Tracks reset ($r$) and update ($z$) gate saturations, proving asymptotic fixed-point settling.
- `modal_interp3_language_probing.py`: Linear diagnostic probing for POS and syntactic role recovery across thought hops $T$.
- `modal_interp4_language_pca_trajectories.py`: 2D PCA trajectory visualization and semantic attractor basins.

### 3. Frontier Benchmarks & Stress Tests
- `modal_benchmark_adaptive_halting.py`: Dynamic velocity early-stopping ($\|\Delta s\|/\|s\| \le \epsilon$) on TinyShakespeare (50% compute reduction).
- `modal_benchmark_deep_gravimem_scaling.py`: Stacking multiple SubQ layers vs multi-layer standard Transformers.
- `modal_benchmark_vs_multilayer_transformer.py`: Head-to-head 1-layer vs 1L, 2L, 4L Transformers at $L=512$.
- `modal_nightmare1_mqar.py`: Multi-Query Associative Recall ($L=512$, 16 interleaved pairs, 100% recall).
- `modal_nightmare2_dyck_grammar.py`: Deep nested Dyck-4 bracket matching up to depth 30+.
- `modal_exp3_length_extrapolation.py`: Zero-shot context length generalization ($L=256 \to 1024$).
- `modal_exp4_extreme_context.py`: Extreme context scaling ($L=256 \dots 4096$) and OOM memory frontier.
- `modal_exp_extreme_natural_language.py`: Natural language modeling at $L=4096$ with 1.1M tokens/sec.
- `modal_exp_extreme_long_context_learning.py`: Extreme context scaling at $L=8,192 \dots 16,384$ tokens on Tesla T4.
- `modal_exp_dynamic_vs_static_needle.py`: Dynamic recomputed routing vs static frozen graph needle recall.
- `modal_exp_shared_attention_unique_mlps.py`: Cross-layer shared attention vs independent attention ablation.

### 4. Pre-Trained Dense LLM Transplant & Architectural Transfer (GPT-2 124M)
- `modal_exp_gpt2_subq_phase0_diagnostic.py`: Study 23 — Attention mass profiling across 144 heads (12 layers $\times$ 12 heads), proving layer-by-layer sparsity patterns on real text.
- `modal_exp_gpt2_subq_transplant_adaptive_t.py`: Study 24 — Direct weight surgery from dense GPT-2 to SubQ; multi-hop dynamical settling ($T=4..6$) reduces zero-train perplexity from `13,940` to `767` ($48.8\times$ reduction).
- `modal_exp_gpt2_subq_full_transplant_adaptation.py`: Study 25 — Unlocked full model adaptation on NVIDIA A10G (Attention + MLPs + GRU), dropping perplexity from `811` $\to$ `174` PPL.
- `modal_exp_gpt2_subq_multidomain_transfer.py`: Study 26 — Multi-domain generalization and cross-corpus transfer benchmark across WebText, WikiText-2, Python Code, and Shakespeare with permanent Modal Volume checkpoint persistence (`subq-gpt2-checkpoints`).


