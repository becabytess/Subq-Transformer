# Season 1 Study Catalog

This catalog preserves the research questions and reported results at a usable level. The complete prose, tables, and code remain in the linked Season 1 sources.

## Origin And Gravimem Foundations

| Area | Question retained | Reported result retained | Status | Source |
|---|---|---|---|---|
| Gravimem V0 retrieval | Can a fixed semantic space be personalized by PageRank mass and query deformation without changing the canonical embeddings? | Implemented PageRank prior, active mass, query deformation, fast reinforcement, slow EMA fusion, and a 70/30 retrieval evaluator. | local-artifact; hypothesis for retrieval benefit | [archive README](../archive/gravimem_retrieval/README.md), [query engine](../archive/gravimem_retrieval/src/engine/query_deformation_engine.py) |
| Gravimem-Pro | Can learned teleportation and recurrent state accumulation improve neural sequence modeling? | Added Pre-LN blocks, step embeddings, learned routing, GRU-style accumulation, and multi-hop benchmarks. | documented-run; superseded implementation | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Positional jumps | Can a small multi-scale jump menu outperform dense attention on longer contexts? | Reported gains from 5, dyadic, and Fibonacci-style sparse jump menus, including the early L=512 long-context result. | documented-run; confounded in early comparisons | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Recurrent refinement | Can repeated updates over a sparse graph replace some physical depth? | Reported strong results on variable dependency tracking, graph navigation, and language benchmarks. | documented-run; relevant hypothesis | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Early halting | Can state velocity or prediction stability select a smaller runtime thought budget? | Reported lower average hops and apparent compute savings under velocity and top-1 stopping rules. | documented-run; implementation gap remains | [research journey](../experiments/RESEARCH_JOURNEY.md) |

## Mechanistic And Interpretability Work

| Study area | Question retained | Reported result retained | Status | Source |
|---|---|---|---|---|
| Dense approximation | Does sparse recurrent routing approximate dense attention geometry? | Measured hidden-state alignment and divergence against dense baselines. | documented-run; not a proof of equivalence | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Dynamic versus static routing | Does recomputing routing at each hop matter? | Reported similar performance for some static-graph and dynamic-routing conditions. | documented-run; task-dependent | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Attractor perturbation | Do hidden states recover after noise? | Reported perturbation recovery and basin-like behavior. | documented-run; local dynamical evidence | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Jacobian analysis | Are local transitions contractive? | Reported spectral radius below one and strong phase-space contraction for selected checkpoints. | documented-run; local result, not universal theorem | [modal_mech4_jacobian_spectral_radius.py](../experiments/modal_mech4_jacobian_spectral_radius.py) |
| Gate and trajectory probing | What information is retained across hops? | Probed gate dynamics, jump distances, linear probes, PCA trajectories, and semantic basins. | documented-run; interpretation remains provisional | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Message-passing width/depth | Can more hops compensate for fewer offsets? | Reported a K/T tradeoff and a balanced iterative-passing advantage in selected tests. | documented-run; unresolved scaling law | [research journey](../experiments/RESEARCH_JOURNEY.md) |
| Meta-optimization and continual learning | Can recurrent state dynamics perform adaptation or learning without ordinary backpropagation? | Explored Hebbian, twin, zero-backprop, streaming, and in-context update variants with mixed results. | documented-run; separate exploratory branch | [research journey](../experiments/RESEARCH_JOURNEY.md) |

## Foundation Models And Diffusion

| Study | Question retained | Reported result retained | Status | Source |
|---|---|---|---|---|
| 23-26 | Can pre-trained GPT-2 be surgically converted to sparse SubQ and adapted? | Sparse transplant and adaptation reduced perplexity substantially from the zero-shot surgery baseline; multi-domain transfer was explored. | documented-run; mixed transfer evidence | [experiments README](../experiments/README.md#4-pre-trained-dense-llm-transplant--multi-domain-adaptation-gpt-2-124m) |
| 27-31 | Can SubQ predict multiple future tokens or infill spans in parallel? | Parallel block prediction, slot embeddings, bidirectional diffusion, and wave-lattice variants were tested. | documented-run; later parallel diffusion exposed weaknesses | [experiments README](../experiments/README.md#5-non-autoregressive-attractor-diffusion--parallel-block-settling) |
| 32-36 | Can BERT be converted to sparse bidirectional wave attention? | Reported strong adapted SubQ-BERT results and multi-token infilling experiments. | documented-run; transplant result, not general proof | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 37-42 | How does sparse GPT-2 compare with dense GPT-2, and how does memory scale? | Reported long-context and Triton speed/memory advantages; controlled language quality was closer to dense than early claims implied. | documented-run; hardware claims need fresh verification | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 43-46 | Can a modern Qwen model use recurrent SubQ thought loops for GSM8K? | Explored transplant, math fine-tuning, recurrent thinking, adaptive halting, and T=4 contraction scaling. | documented-run; not the current baseline | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |

## Harmonic SubQ

| Study | Question retained | Reported result retained | Status | Source |
|---|---|---|---|---|
| 47 | What distance patterns do dense GPT-2 heads use? | Reported power-law-like distance use and attention-sink behavior. | documented-run; descriptive profiling | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 48-49 | Can continuous Fourier carriers replace fixed jump menus? | Reported improved language perplexity from continuous and recurrent harmonic waves. | documented-run; relevant routing hypothesis | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 50-54 | Do sparse wave peaks, multiple frequencies, token phase, and analytical waves help? | Reported that sparse wave peaks and multi-frequency carriers improved selected TinyShakespeare results; analytical waves beat tested RNN offset generators. | documented-run; task-specific evidence | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 55 | Can harmonic routing remain strictly causal? | Reported zero future-token leakage for static and causal-prefix wave variants. | documented-run; relevant causal requirement | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 56-57 | Does evolving Q/K/V make recurrent hops genuinely transitive, and does quality scale with T? | Reported better performance and deeper T scaling when Q/K/V were reprojected at every hop. | documented-run; central Season 2 hypothesis | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 58-63 | Which sparse routing and wave integration rule works best? | Reported dynamic harmonic waves, wave bias, and multiplicative gating reaching the best documented TinyShakespeare perplexities. | documented-run; strong but benchmark-specific | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 64 | Does a fused Triton implementation make sparse harmonic attention competitive at long context? | Reported favorable scaling and a crossover against FlashAttention at large sequence lengths. | documented-run; hardware result to reproduce | [experiments/modal_exp_harmonic_subq_triton_benchmark.py](../experiments/modal_exp_harmonic_subq_triton_benchmark.py) |
| 65-66 | Can harmonic and recurrent sparse attention be transplanted into GPT-2? | Reported improved adaptation over older sparse transplant variants and promising one-layer recurrent results. | documented-run; comparison conditions require care | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |

## Vision, Width, And Routing Geometry

| Study | Question retained | Reported result retained | Status | Source |
|---|---|---|---|---|
| 67 | Can one-layer harmonic SubQ match a deeper dense ViT on CIFAR-100? | Reported approximately 50.29% Top-1 and 79.92% Top-5 at T=8, close to the 4-layer dense reference with fewer parameters. | documented-run; relevant empirical result | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 68 | Does deeper recurrent thought improve high-resolution vision? | Reported monotonic gains from T=4 to T=12, reaching 49.86% Top-1 and 79.00% Top-5. | documented-run; relevant empirical result | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 69 | Can a checkpoint trained at T=4 extrapolate to arbitrary test-time T? | Reported a performance peak around the trained horizon and motivated stochastic-T training plus deep supervision. | documented-run; unresolved | [experiments/modal_exp_subq_vit_test_time_scaling.py](../experiments/modal_exp_subq_vit_test_time_scaling.py) |
| 71-72 | Do harmonic super-tokens or local pooling beat discrete sparse points? | Reported modest accuracy gains in some pooled settings but more gathering complexity and mixed controlled results. | documented-run; superseded for the initial baseline | [experiments README](../experiments/README.md#6-bidirectional-wave-lattices-full-12-layer-foundation-transplants--controlled-benchmarks) |
| 73 | Is physical SubQ depth useful compared with recurrent temporal depth? | Reported a 2-layer model near the single-layer T=12 result, with substantially more parameters. | documented-run; later confounded by capacity | [experiments/study73_1_2layer_sequential.py](../experiments/study73_1_2layer_sequential.py) |
| 74 | Can Base-8 offsets provide gap-free coverage without learned waves? | Reported strong CIFAR performance from fixed radix-style offsets with zero learned wave parameters. | documented-run; central evidence for simple routing | [experiments/study74_radix_optimal_offsets.py](../experiments/study74_radix_optimal_offsets.py) |
| 75 | Does a FEN channel-roll escrow improve short-horizon SubQ vision? | Reported better results than one-layer dense ViT, but weaker than pure harmonic SubQ and likely underused at T=4. | documented-run; FEN remains relevant for long horizons | [experiments/study75_fen_subq_vit.py](../experiments/study75_fen_subq_vit.py) |
| 76-77 | Does a linear GRU prefix scan improve sparse wave reasoning? | Reported gains when extra parameters were available, including 46.92% Top-1 for the unidirectional T=8 hybrid. | documented-run; capacity and task dependent | [experiments/study77_unidir_gru_subq_t8.py](../experiments/study77_unidir_gru_subq_t8.py) |
| 78-79 | Does the GRU scan still help under strict parameter matching? | Reported that shrinking the MLP to pay for the GRU reduced performance substantially. | documented-run; useful negative result | [experiments/study78_param_matched_unidir_gru_subq_t8.py](../experiments/study78_param_matched_unidir_gru_subq_t8.py) |
| 80-82 | Was the apparent physical-depth gain actually a capacity effect? | Reported scaled pure SubQ at about 742k parameters reaching 49.74% Top-1, beating the documented 2-layer and GRU hybrids in that setup. | documented-run; strong but setup-specific | [experiments/study80_scaled_pure_subq_t8.py](../experiments/study80_scaled_pure_subq_t8.py), [STATE_OF_SUBQ](../STATE_OF_SUBQ.md) |

## Current Uncommitted Queue

These scripts are present locally but are not committed, and their raw outputs are not present in this checkout. Their questions are retained here without treating intended output as evidence.

| Script family | Question retained | Recorded state | Status |
|---|---|---|---|
| 75 parallel diffusion, 75b-c | Can a SubQ model trained or adapted for autoregressive prediction perform parallel block diffusion? | The script comments record failures and increasingly targeted fixes: long bidirectional offsets, longer training, slot embeddings, and a corrected masked objective. | uncommitted; results not locally verified |
| 75d | Can the saved SubQ-BERT checkpoint support parallel infilling without starting from scratch? | Rebuilds the Study 35 architecture, checks checkpoint behavior, and tests MaskGIT-style decoding plus light tuning. | uncommitted; results not locally verified |
| 75e-j | Can sparse SubQ-BERT perform long-distance, open-vocabulary needle copying and generalize beyond templates? | Comments record the progression: distant recall failure, weak tuning, template success, poor unseen generalization, then a pointer-head hypothesis. | uncommitted; prior comments are not raw evidence |
| 83 | Under strictly causal conditions, which is strongest: dense attention, pure harmonic SubQ, or causal GRU plus SubQ? | Seven-model TinyShakespeare shootout with matched minibatches and several parameter scales. | uncommitted; results not locally verified |
| 84 | Can GRU-derived content salience replace harmonic routing or combine with it? | Five-model causal language shootout. The content-salience implementation expands over query/key positions and should be treated as potentially quadratic until redesigned. | uncommitted; results not locally verified |
| 85a-c | What happens when W_v, W_k, or W_q is removed? | Three TinyShakespeare ablations at T=8. | uncommitted; results not locally verified |
| 86a-b | Which component is doing the work: GRU, sparse SubQ, Q/K/V, or the MLP? | Component isolation and no-GRU Q/K/V ablations. | uncommitted; results not locally verified |
| 86c | Can the candidate architectures solve a stricter MQAR task? | MQAR stress test with pure GRU, dense attention, pure SubQ, GRU plus SubQ, and reduced-Q/V variants. | uncommitted; results not locally verified |
| 87 | How do canonical Gravimem, multi-scale SubQ, and harmonic SubQ compare on MQAR? | Full MQAR verification at L=512 with 16 key/value pairs and 8 queries. | uncommitted; results not locally verified |
| 88a-d | Are Q, K, and V individually necessary on MQAR, and can a pure wave router work? | Separate removal of Q, K, V, Q+V, and all dot products. | uncommitted; results not locally verified |

## Important Reading Rule

The full Season 1 tables remain authoritative for what was reported. This catalog is intentionally more cautious about what those results prove.
