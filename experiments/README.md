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
