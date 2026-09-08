# Season 1 Core Result Matrix

These are the numerical results most likely to matter for Season 2. They are copied as reported in the Season 1 documents, not independently revalidated here.

| Area | Reference | Reported comparison | Why retain it | Evidence status |
|---|---|---|---|---|
| Early long-context language | Gravimem 1L T=4 vs dense Transformer | Reported 6.17 PPL versus 10.00 PPL for a 4-layer dense reference at L=512. | First strong evidence that sparse recurrent refinement can beat a deeper dense baseline. | documented-run; early controls need review |
| Causal language | Harmonic SubQ T=8 | Reported best TinyShakespeare PPL around 5.36 with wave gating. | Shows the effect is not limited to classification. | documented-run; benchmark-specific |
| Deep bracket reasoning | Harmonic SubQ T=8 vs 4L dense | Reported 86.57% versus 86.15% on the deepest Dyck-4 tier. | Direct multi-hop stress result. | documented-run; task-specific |
| CIFAR-100, L=65 | Harmonic SubQ T=8 vs 1L dense | Reported 50.29% versus 37.38% Top-1; close to 4L dense at 51.87%. | Strong same-task evidence for iterative sparse refinement. | documented-run |
| CIFAR-100, L=257 | Harmonic SubQ T=4/8/12 | Reported 45.67% -> 48.36% -> 49.86% Top-1 as T increased. | Most direct evidence for iterative improvement over a local view. | documented-run; reproduce in Season 2 |
| CIFAR parameter parity | Pure SubQ vs GRU hybrids | Reported strict parameter matching reduced the GRU hybrid advantage; scaled pure SubQ reached 49.74% Top-1. | Prevents us from assuming the GRU is essential. | documented-run; setup-specific |
| BERT transplant | Adapted 12L SubQ-BERT vs dense BERT | Reported 58.94% vs 52.42% Top-1 and masked PPL 8.49 vs 15.24. | Demonstrates that sparse replacement can adapt from a foundation checkpoint. | documented-run; needs exact reproducibility audit |
| GPT-2 transplant | Recurrent 1L SubQ-GPT2 vs dense GPT-2 | Reported 88.62 vs 104.17 PPL in one adapted setup. | Interesting evidence for parameter-tied recurrence. | documented-run; comparison conditions need care |
| Hardware scaling | Triton SubQ vs FlashAttention | Reported a large-context crossover and approximately 12.77x speedup at L=65,536 in one benchmark. | Relevant only after correctness and kernel methodology are rechecked. | documented-run; hardware result to reproduce |
| Long needle | Sparse BERT vs dense BERT | The 75e script records 0/24 distant recall for SubQ before tuning versus approximately 20/24 for dense BERT. | Essential negative result: low PPL does not guarantee arbitrary copying. | script-recorded; raw output not local |

## Interpretation Policy

The matrix preserves the numbers. Season 2 should rerun the most important comparisons under one protocol before using them as baselines.
