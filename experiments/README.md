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

### 4. Pre-Trained Dense LLM Transplant & Multi-Domain Adaptation (GPT-2 124M)
- `modal_exp_gpt2_subq_phase0_diagnostic.py`: Study 23 — Attention mass profiling across 144 heads (12 layers $\times$ 12 heads), proving layer-by-layer sparsity patterns on real text.
- `modal_exp_gpt2_subq_transplant_adaptive_t.py`: Study 24 — Direct weight surgery from dense GPT-2 to SubQ; multi-hop dynamical settling ($T=4..6$) reduces zero-train perplexity from `13,940` to `767` ($48.8\times$ reduction).
- `modal_exp_gpt2_subq_full_transplant_adaptation.py`: Study 25 — Unlocked full model adaptation on NVIDIA A10G (Attention + MLPs + GRU), dropping perplexity from `811` $\to$ `174` PPL.
- `modal_exp_gpt2_subq_multidomain_transfer.py`: Study 26 — Multi-domain generalization and cross-corpus transfer benchmark across WebText, WikiText-2, Python Code, and Shakespeare with permanent Modal Volume checkpoint persistence (`subq-gpt2-checkpoints`).

### 5. Non-Autoregressive Attractor Diffusion & Parallel Block Settling
- `modal_exp_subq_parallel_block_diffusion.py`: Study 27 — Fine-tuning SubQ-GPT2 to predict blocks of $N=8$ tokens in 1 parallel pass; tracks continuous state relaxation ($17.1 \to 4.8$ velocity decay).
- `modal_exp_subq_iterative_block_diffusion.py`: Study 28 — Position-specific learned slot embeddings and dynamic masking schedules ($k \in [1, 8]$).
- `modal_exp_subq_bidirectional_diffusion.py`: Study 29 — Bidirectional intra-block graph attention hops with unigram frequency debiasing and repetition penalty.

### 6. Bidirectional Wave Lattices, Full 12-Layer Foundation Transplants & Controlled Benchmarks
- `experiments/test_subq_bidirectional_wave.py`: Studies 30-31 — Symmetrical logarithmic wave attention ($\mathcal{M} = \{\pm 1, \pm 2, \dots, \pm 64\}$), Green's function dispersion proof, and gradient equivalence vs dense attention.
- `modal_exp_bert_subq_transplant.py`: Studies 32-33 — BERT-Base 110M foundation transplant with physical attractor relaxation traces ($>77\%$ velocity contraction).
- `modal_exp_bert_subq_kl_distillation.py`: Study 34 — Soft KL logit distillation from 12-layer dense BERT teacher into SubQ student.
- `modal_exp_bert_12layer_full_subq.py`: Study 35 — Full 12-layer SubQ-BERT zero-shot transplant & adaptation, beating dense BERT (**`58.94%` vs `52.42%` Top-1 Acc**, Mask PPL `8.49` vs `15.24`).
- `modal_exp_bert_12layer_multitoken_generation.py`: Study 36 — Multi-token future span generation ($N=4 \dots 48$ tokens) and non-autoregressive infilling dynamics.
- `modal_exp_gpt2_12layer_full_subq.py`: Study 37 — Full 12-layer SubQ-GPT2 (124M params) full-depth transplant, rapid adaptation, and autoregressive paragraph text generation.
- `modal_exp_controlled_gpt2_vs_subq.py`: Study 38 — Rigorous apples-to-apples controlled benchmark (12L Dense GPT-2 vs. 12L SubQ-GPT2) under identical training budgets, minibatches, and random seeds across in-domain (WikiText-2) and out-of-domain (Penn Treebank) datasets.
- `modal_exp_gpt2_long_context_scaling.py`: Study 39 — Full 12-layer 124M foundation model long-context scaling frontier ($L = 1024 \to 32768$).
- `modal_exp_subq_triton_kernel.py`: Study 40 — Custom fused OpenAI Triton GPU kernel for SubQ logarithmic wave attention ($L = 1024 \to 65536$), achieving 9.38 Million tokens/second at $<500\text{ MB}$ VRAM.
- `modal_exp_flashattn_vs_subq_triton.py`: Study 41 — Direct 3-way head-to-head benchmark (Dense FlashAttention-2 vs. PyTorch Eager SubQ vs. OpenAI Triton SubQ Kernel), proving SubQ becomes $18.7\times$ faster than FlashAttention-2 at $L=65,536$.
- `modal_exp_long_seq_accuracy_eval.py`: Study 42 — Zero-shot long-context accuracy & perplexity retention benchmark ($L = 128 \to 4096$), proving SubQ beats Dense GPT-2 by +7.39% Top-1 accuracy at $L=4096$ while evaluating only $0.20\%$ of tokens.
- `modal_exp_qwen_subq_gsm8k.py`: Study 43 — Modern 0.5B Foundation LLM (`Qwen/Qwen2.5-0.5B`, 490M params, 24 layers, GQA, RoPE, SwiGLU) SubQ transplant and GSM8K reasoning fine-tuning.
- `modal_exp_qwen_gsm8k_eval.py`: Study 44 — Rigorous quantitative GSM8K math reasoning shootout (Dense Qwen vs SubQ-Qwen), evaluating exact numerical accuracy and delimiter adherence.
- `modal_exp_qwen_subq_recurrent_gsm8k.py` & `modal_exp_qwen_subq_fast_eval.py`: Study 45 — SubQ-Qwen2.5-0.5B with Recurrent Thinking Loops ($T=6$) & Adaptive Halting on GSM8K, demonstrating the depth-multiplier effect ($24 \times 6 = 144$ layers) in deep foundation models.
- `modal_exp_qwen_subq_t4_gsm8k.py`: Study 46 — Balanced Recurrence ($K=8, T=4$) with Contraction Scaling ($1/\sqrt{T}$) on Qwen2.5-0.5B, dropping Train Loss to `1.7149` (PPL `5.56`) and reaching `88.0%` Delimiter Adherence.
- `modal_exp_gpt2_attention_distribution.py`: Study 47 — Empirical attention distance distribution profiling across all 144 heads of dense GPT-2, discovering the power-law decay curve ($P(d) \sim 1/d$) and the 68.8% Attention Sink phenomenon.
- `modal_exp_fourier_token_wave.py` & `modal_exp_fourier_global_wave.py`: Study 48 — Continuous Fourier / Harmonic Sinusoidal Attention Routing (Token-Level vs. Global Carrier Waves), outperforming standard transformers with $>4\times$ lower perplexity.
- `modal_exp_fourier_global_recurrent.py` & `modal_exp_plot_fourier_waves.py`: Study 49 — Recurrent Global Fourier Wave Attention with Iterative Dynamic Recomputation ($T=4$), generating high-resolution 4-panel harmonic wave visualizations and breaking the 6.0 PPL barrier (`5.70` PPL).
- `modal_exp_sparse_fourier_peak_subq.py`: Study 50 — Pure Sparse Harmonic Wave-Peak SubQ ($K=8$ Peaks, $T=4$ Hops, Strict $\mathcal{O}(L \cdot K)$), outperforming fixed logarithmic grids by $>2\times$ (`5.77` vs `12.71` PPL) while evaluating strictly 8 peak tokens per query.
- `modal_exp_fourier_wave_capacity_ablation.py`: Study 51 — Exact Parameter-Matched Fourier Wave Harmonic Capacity Ablation ($N = 1, 4, 8, 12, 16, 20$ Waves), proving that multi-frequency harmonic superposition monotonically improves perplexity up to $N=12$ (`5.76` PPL) with 100% identical parameter counts (256,848 params) and zero runtime overhead.
- `modal_exp_per_token_phase_shift_wave.py`: Study 52 — Per-Token Dynamic Phase-Shift Fourier Wave Routing, enabling individual tokens to customize their receptive field phase ($\phi_i$) while sharing global carrier frequencies ($\omega$), dynamically shifting peaks between dense spelling n-grams ($d=1..5$) and distant speaker tags ($d=24..43$).
- `modal_exp_plot_fourier_peak_distribution.py`: Study 53 — Empirical Profiling & Distribution of Fourier Wave Peaks vs. Fibonacci & Logarithmic Grids across 819,200 tokens, revealing the dense local syntactic well ($d \le 7$, 60.3% mass), the silent void ($d=8..35$), and harmonic verse/dialogue packets ($d \approx 58, 94$).
- `modal_exp_rnn_offset_generator.py`: Study 54 — RNN Offset Generators vs. Continuous Fourier Waves, comparing autoregressive delta-jump GRUs vs. 1D spatial curve RNN synthesizers vs. Fourier waves, confirming analytical Fourier waves achieve the best perplexity (`5.86` vs `5.96` vs `7.92`).
- `modal_exp_strictly_causal_fourier_subq.py`: Study 55 — Strictly Causal Fourier Wave SubQ (Zero Future Token Leakage), comparing pure learned static carrier waves (`5.90` PPL, zero sequence access) vs. strictly causal prefix-summed waves (`5.76` PPL, cumsum $j \le i$), verifying 100% causal mathematical integrity.
- `modal_exp_evolving_qkv_subq.py`: Study 56 — Full Evolving $Q, K, V$ Recurrent Self-Attention vs. Static $K, V$, unlocking true transitive receptive field cascades ($A \to B \to C \to D$) and setting a new record perplexity of `5.64`.
- `modal_exp_t_scaling_evolving_qkv.py`: Study 57 — Re-Evaluating Recurrent Thought Depth Scaling ($T = 1 \dots 12$) with Full Evolving $Q, K, V$, showing that perplexity scales monotonically with depth down to **`5.39` Perplexity** ($T=12$, Val Loss `1.6841`), proving the depth-equivalence theorem in 1 physical layer.
- `modal_exp_definitive_shootout.py`: Study 58 — Definitive Apples-to-Apples Shootout across all 8 architectures under 100.00% identical hyperparameters ($D=128$), seeds, and schedules, proving that Dynamic Harmonic Waves at $T=8$ achieve the lowest perplexity (`5.58` PPL) and beat both Fixed Fibonacci (`5.92` PPL) and 4-Layer Dense Transformers (`5.85` PPL) with $70\%$ fewer parameters.
- `modal_exp_hop_evolving_waves.py`: Study 59 — Hop-Evolving Harmonic Waves via Dynamical Transition Layers ($w^{(t)} = w^{(t-1)} + \Delta w$), testing state-evolving wave parameter dynamics across iterations and reaching a new record of `5.37` Perplexity (Val Loss `1.6815`) at $T=8$.
- `modal_exp_dyck4_rematch.py`: Study 60 — The Dyck-4 Deep Bracket Rematch across nesting depths up to 30+, proving that New Harmonic SubQ jumps from `75.30%` to **`83.66%`** overall and outperforms the 4-layer dense transformer on extreme nesting depths 16–30 (`86.57%` vs `86.15%`) with $72\%$ fewer parameters.
- `modal_exp_train_and_visualize_waves.py`: Study 61 — Linguistic Profiling, Checkpointing (`checkpoints/best_harmonic_subq_t8.pt`), and Wave Dynamics Visualization (`harmonic_subq_waves_and_token_routing.png`), profiling multi-head carrier interference curves and token-level receptive fields on Shakespeare.
- `modal_exp_pure_router_no_logit_bias.py`: Study 62 — Pure Wave Routing (No Logit Bias) vs. Harmonic-Biased Attention, proving that while pure wave routing alone reaches `5.85` PPL, adding the wave amplitude $W(d)$ as a learnable harmonic spatial prior logit bias drops perplexity by `+8%` down to **`5.39` PPL**.
- `modal_exp_multiplicative_wave_gating.py`: Study 63 — Multiplicative Wave Gating vs Additive Logit Bias, proving that Multiplicative Logit Scaling ($\text{Scores} \cdot 2\sigma(W(d))$) sets an all-time record perplexity of **`5.36`** (Val Loss `1.6790`) at $T=8$.
- `modal_exp_harmonic_subq_triton_benchmark.py`: Study 64 — Harmonic SubQ Speed, VRAM, and Scaling Benchmark with OpenAI Triton Kernel, proving that fused Triton Harmonic SubQ processes $L=65,536$ context in just `6.11 ms` (`10.7M tok/s`, **`12.77x faster`** than FlashAttention-2) and achieves a **`13.20x end-to-end speedup`** over a 4-layer Dense Transformer.
- `modal_exp_harmonic_subq_gpt2_transplant.py` & `modal_exp_harmonic_gpt2_refined_transplant.py`: Study 65 — Full 12-Layer Harmonic SubQ-GPT2 (124M Parameters) Transplant, adapting pre-trained GPT-2 into Harmonic SubQ with continuous learned carrier waves, dropping WikiText-2 perplexity down to **`141.45`** and beating the old fixed dyadic SubQ baseline (`151.32`).
- `modal_exp_recurrent_evolving_qkv_gpt2.py`: Study 66 — Recurrent Harmonic SubQ GPT-2 with Full Evolving Q, K, V across Thought Depths ($T=1 \dots 8$), proving that a **1-layer recurrent SubQ block (85M parameters)** initialized from pre-trained GPT-2 achieves **`88.62` Perplexity and `31.46%` Top-1 Accuracy**, beating the 12-Layer Dense GPT-2 Oracle (`104.17` PPL, `6.28%` Top-1) with **32% fewer parameters**!
- `modal_exp_cifar100_harmonic_subq_vit.py`: Study 67 — CIFAR-100 Vision Transformer Benchmark Shootout ($L=65$ tokens), proving that a **1-layer Harmonic SubQ ViT ($T=8$ hops, 490k parameters)** achieves **`50.29%` Top-1 and `79.92%` Top-5 accuracy**, beating the parameter-matched 1L Dense ViT (`37.38%`) by **`+12.91%`** and matching a 4-Layer Dense ViT (`51.87%` / `80.11%`) with **73% fewer parameters**!
- `modal_exp_cifar100_high_res_subq_vit.py`: Study 68 — High-Resolution CIFAR-100 Benchmark ($L=257$ tokens, $2 \times 2$ patches) across deep thinking iterations $T \in [4, 8, 12]$, proving strict monotonic performance gains ($T=4 \to 8 \to 12: 45.67\% \to 48.36\% \to \mathbf{49.86\%}$ Top-1), beating 1L Dense ViT by **`+10.13%`**, and beating 4L Dense ViT on Top-5 accuracy (**`79.00%` vs `77.80%`**) with **72% fewer parameters**!
- `modal_exp_subq_vit_test_time_scaling.py`: Study 69 — Zero-Shot Test-Time Thinking Compute Extrapolation ($T_{\text{train}}=4 \to T_{\text{eval}} \in [1 \dots 32]$), saving model checkpoint to Modal persistent volume and analyzing why fixed-depth training creates a sharp peak at $T=4$ (`45.69%`), revealing the exact recipe (Stochastic $T$ Training + Deep Supervision) needed for monotonic test-time compute scaling!
- `modal_exp_subq_vit_harmonic_filterbank.py`: Study 71 — Continuous Harmonic Filterbank (Harmonic Wavelet Pooling) SubQ ViT, proving that pooling 5-patch local neighborhoods around carrier wave peaks into $K=8$ **Harmonic Super-Tokens** achieves **`46.78%` Top-1 / `77.01%` Top-5 accuracy**, beating raw discrete point sampling (`45.67%` / `76.19%`) and boosting training accuracy from `48.05%` to **`52.44%`** with zero extra parameters!
- `study72_*.py`: Study 72 — Apples-to-Apples Shootout of State-Averaged Harmonic Super-Tokens vs. Discrete Points ($K=8$ and $K=40$), showing that $P=8$ State Super-Tokens (pooling 40 raw tokens) achieve **`38.47%` Top-1 / `70.95%` Top-5** (+2.83% over discrete $K=8$) with $5\times$ fewer attention dot-products than discrete $K=40$.
- `study73_1_2layer_sequential.py` & `study73_2_2layer_interleaved.py`: Study 73 — 2-Layer Harmonic SubQ ViT Benchmark (Physical Depth vs Recurrent Temporal Depth), showing that adding a second physical layer (968k params, `49.21%` Top-1 / `79.33%` Top-5) performs similarly to a single 1-layer SubQ model with $T=12$ hops (520k params, `49.86%` Top-1 / `79.00%` Top-5), proving that physical layer depth is redundant when recurrent thought depth ($T$) is available.
- `study74_radix_optimal_offsets.py` & `study74_radix_optimal_offsets_t4.py`: Study 74 — Base-$K$ (Radix-8) Mathematical Optimal Offsets Benchmark, proving that pure hardcoded geometric Base-8 strides ($\mathcal{D}_t = \{0, 1\cdot 8^{t-1}, \dots, 7\cdot 8^{t-1}\}$) provide 100% gap-free coverage ($8^3 = 512 \ge 257$) in just $T=3$ hops and beat 1-layer Dense ViT by **`+4.57%` Top-1 (`44.30%` vs `39.73%`)** with zero learned wave parameters!
- `study75_fen_subq_vit.py`: Study 75 — FEN-SubQ Vision Transformer (Channel-Roll Escrow Vault + Harmonic Waves), testing the dual-pathway Feature-Escrow Network hypothesis on High-Res CIFAR-100 ($L=257$ patches, $T=4$), achieving **`43.70%` Top-1 / `75.02%` Top-5** (+3.97% over 1L Dense ViT) and analyzing the interaction between short 4-hop relaxation graphs and circular shift escrow tapes.
- `study76_gru_subq_vit.py`: Study 76 — CuDNN Bi-GRU Scan + SubQ Wave Attention Hybrid ViT (Linear Pre-Scan + Multi-Hop Waves), testing a fast $O(L)$ bidirectional GRU patch scan followed by 1-layer SubQ wave relaxation ($T=4$) on High-Res CIFAR-100 ($L=257$), achieving **`45.95%` Top-1 / `76.04%` Top-5** (+6.22% over 1L Dense ViT) and setting an all-time record train convergence of **`51.91%`** at 31.2s/epoch.
- `study77_unidir_gru_subq_t8.py`: Study 77 — Strictly Unidirectional CuDNN GRU Scan + SubQ Wave Relaxation ($T=8$ Hops), eliminating any future look-ahead confound (`bidirectional=False`) and scaling thought depth to $T=8$, achieving **`46.92%` Top-1 / `76.93%` Top-5** (+7.19% over 1L Dense ViT) and setting a new record train convergence of **`53.13%`** and test loss of `2.0651`.
- `study78_param_matched_unidir_gru_subq_t8.py` & `study79_param_matched_bidir_gru_subq_t8.py`: Studies 78 & 79 — Parameter-Matched Down-Scaled GRU-SubQ (~520k params), revealing that forcing the GRU into a 520k budget starved the MLP to 1x/1.75x, dropping performance to 40.11% and 42.98%.
- `study80_scaled_pure_subq_t8.py`, `study81_scaled_dense_vit.py`, `study82_scaled_bidir_gru_subq_t8.py`: Studies 80, 81 & 82 — The Definitive 742k Parameter Parity Shootout. Pure SubQ without any GRU reaches **`49.74%` Top-1, `79.56%` Top-5, and `1.9126` Loss** at 742k params ($T=8$), crushing 2-Layer SubQ (`49.22%` at 968k params) with 226k fewer parameters, proving that physical layer depth in SubQ was merely a parameter inflation illusion.


