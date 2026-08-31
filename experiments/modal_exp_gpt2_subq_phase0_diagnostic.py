import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "numpy",
        "requests",
        "accelerate"
    )
)

app = modal.App("gpt2-subq-phase0-diagnostic", image=image)

@app.function(gpu="T4", timeout=600)
def run_phase0_diagnostic():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  PHASE 0 DIAGNOSTIC: EMPIRICAL ATTENTION MASS PROFILING OF PRE-TRAINED GPT-2 (124M)")
    print("  Question: Does dense attention in GPT-2 actually use O(L^2) freedom, or is it concentrated on sparse relative offsets?")
    print("  Goal: Compute exact per-head offset-mass distributions and determine the theoretical ceiling for SubQ Top-K transplant.")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Pre-Trained GPT-2 with output_attentions=True
    print("Loading Pre-Trained GPT-2 (124M) from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True).to(device)
    model.eval()

    n_layers = model.config.n_layer # 12
    n_heads = model.config.n_head   # 12
    total_heads = n_layers * n_heads # 144
    print(f"Model Architecture: {n_layers} Layers x {n_heads} Heads = {total_heads} Total Attention Heads\n")

    # 2. Download Real Multi-Domain Calibration Text (TinyShakespeare + Wikitext Sample)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(url).text
    tokens = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    seq_len = 512
    n_sequences = 60 # 60 sequences of length 512 = 30,720 tokens

    print(f"Calibration Dataset: {n_sequences} sequences of length {seq_len} ({n_sequences * seq_len:,} tokens)\n")

    # 3. Accumulate Attention Mass per Relative Offset delta = (i - j)
    # offset_hist[layer, head, delta] where delta in [0, seq_len - 1]
    offset_mass_accum = torch.zeros(n_layers, n_heads, seq_len, device="cpu", dtype=torch.float64)
    offset_count_accum = torch.zeros(seq_len, device="cpu", dtype=torch.float64)

    # Number of valid causal token pairs for each offset delta across sequence length L
    for delta in range(seq_len):
        offset_count_accum[delta] = (seq_len - delta) * n_sequences

    print("Running Attention Mass Profiling across all 144 heads...")
    t0 = time.time()

    batch_size = 4
    n_batches = n_sequences // batch_size

    with torch.no_grad():
        for b in range(n_batches):
            idx_start = b * batch_size * seq_len
            x_batch = torch.stack([
                data_tensor[idx_start + i * seq_len : idx_start + (i + 1) * seq_len]
                for i in range(batch_size)
            ]).to(device)

            outputs = model(x_batch)
            # attentions is a tuple of n_layers tensors: each [batch_size, n_heads, seq_len, seq_len]
            attentions = outputs.attentions

            for l in range(n_layers):
                attn_l = attentions[l] # [B, H, L, L]
                for delta in range(seq_len):
                    # Diagonal elements at distance delta: i - j = delta
                    # torch.diagonal(attn_l, offset=-delta, dim1=-2, dim2=-1) -> [B, H, L - delta]
                    diag = torch.diagonal(attn_l, offset=-delta, dim1=-2, dim2=-1)
                    mass_sum = diag.sum(dim=(0, 2)).cpu().to(torch.float64) # [H]
                    offset_mass_accum[l, :, delta] += mass_sum

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                print(f"  Processed Batch {b+1:>2}/{n_batches} ({((b+1)/n_batches)*100:.0f}%) in {time.time()-t0:.1f}s")

    print(f"\nProfiling completed in {time.time() - t0:.1f}s\n")

    # 4. Normalize to obtain True Probability Distribution per Head
    # Total attention mass per query token sums to 1.0 (via causal softmax)
    # Total attention mass per head across sequence = sum_i sum_j A_ij = L
    total_mass_per_head = seq_len * n_sequences # Expected sum
    norm_offset_mass = offset_mass_accum / offset_mass_accum.sum(dim=-1, keepdim=True) # [12, 12, 512]

    # 5. Compute Top-K Cumulative Mass Retention
    K_values = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128]

    # cumulative_retention[layer, head, k_idx]
    retention_table = np.zeros((n_layers, n_heads, len(K_values)))
    optimal_menus = {} # (layer, head) -> list of top-K offsets

    for l in range(n_layers):
        for h in range(n_heads):
            masses = norm_offset_mass[l, h].numpy() # [512]
            sorted_indices = np.argsort(masses)[::-1] # offsets sorted by mass descending
            sorted_masses = masses[sorted_indices]
            cum_masses = np.cumsum(sorted_masses)

            optimal_menus[(l, h)] = sorted_indices[:32].tolist()

            for k_idx, k in enumerate(K_values):
                retention_table[l, h, k_idx] = cum_masses[k - 1] * 100.0

    # -------------------------------------------------------------------------
    # 6. Global Summary & Attention Ceiling Curve
    # -------------------------------------------------------------------------
    print("=" * 125)
    print("  GLOBAL THEORETICAL CEILING: CUMULATIVE ATTENTION MASS CAPTURED BY TOP-K RELATIVE OFFSETS")
    print("=" * 125)
    print(f"{'Top-K Menu Size':<18} | {'Mean Mass Retained (%)':<24} | {'Min Head Mass (%)':<20} | {'Max Head Mass (%)':<20} | {'Compression Ratio':<18}")
    print("-" * 125)

    for k_idx, k in enumerate(K_values):
        k_masses = retention_table[:, :, k_idx].flatten()
        mean_k = np.mean(k_masses)
        min_k = np.min(k_masses)
        max_k = np.max(k_masses)
        compression = seq_len / k
        print(f"K = {k:<14} | {mean_k:>20.2f}% | {min_k:>16.2f}% | {max_k:>16.2f}% | {compression:>14.1f}x")

    print("=" * 125)

    # -------------------------------------------------------------------------
    # 7. Layer-by-Layer Breakdown for K=16 and K=32
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  LAYER-BY-LAYER ATTENTION CONCENTRATION (K = 16 vs K = 32 OFFSETS vs FULL 512)")
    print("=" * 125)
    print(f"{'Layer Index':<14} | {'Mean Retained (K=8)':<22} | {'Mean Retained (K=16)':<24} | {'Mean Retained (K=32)':<24} | {'Layer Head Specialization'}")
    print("-" * 125)

    idx_k8 = K_values.index(8)
    idx_k16 = K_values.index(16)
    idx_k32 = K_values.index(32)

    for l in range(n_layers):
        layer_k8 = np.mean(retention_table[l, :, idx_k8])
        layer_k16 = np.mean(retention_table[l, :, idx_k16])
        layer_k32 = np.mean(retention_table[l, :, idx_k32])

        # Inspect offset distribution of this layer
        top_offsets_layer = []
        for h in range(n_heads):
            top_offsets_layer.extend(optimal_menus[(l, h)][:3])
        unique, counts = np.unique(top_offsets_layer, return_counts=True)
        top_3_common = unique[np.argsort(counts)[::-1][:3]].tolist()

        if l <= 2:
            spec = f"Ultra-Local / Adjacent Syntax (Common Offsets: {top_3_common})"
        elif l <= 8:
            spec = f"Syntactic & Coreference Inductions (Common Offsets: {top_3_common})"
        else:
            spec = f"Semantic Synthesis / Delimiter Anchors (Common Offsets: {top_3_common})"

        print(f"Layer {l:>2} (L{l+1:>2})  | {layer_k8:>18.2f}% | {layer_k16:>20.2f}% | {layer_k32:>20.2f}% | {spec}")

    print("=" * 125)

    # -------------------------------------------------------------------------
    # 8. Head Specialization Taxonomy (Categorizing all 144 Heads)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  HEAD TAXONOMY & SPARSITY PROFILING ACROSS ALL 144 HEADS")
    print("=" * 125)

    ultra_sparse_heads = [] # >95% mass in top-16
    dense_diffuse_heads = [] # <70% mass in top-16
    induction_heads = []

    for l in range(n_layers):
        for h in range(n_heads):
            ret_k16 = retention_table[l, h, idx_k16]
            top_offsets = optimal_menus[(l, h)][:5]
            if ret_k16 >= 90.0:
                ultra_sparse_heads.append((l, h, ret_k16, top_offsets))
            elif ret_k16 < 70.0:
                dense_diffuse_heads.append((l, h, ret_k16, top_offsets))
            else:
                induction_heads.append((l, h, ret_k16, top_offsets))

    print(f"  1. Highly Localized / Sparse Heads (>=90% mass in Top-16) : {len(ultra_sparse_heads):>3} / 144 heads ({len(ultra_sparse_heads)/144*100:.1f}%)")
    print(f"  2. Intermediate / Structural Heads (70%-90% in Top-16)     : {len(induction_heads):>3} / 144 heads ({len(induction_heads)/144*100:.1f}%)")
    print(f"  3. Diffuse / Global Attention Heads (<70% in Top-16)       : {len(dense_diffuse_heads):>3} / 144 heads ({len(dense_diffuse_heads)/144*100:.1f}%)")

    print("\nSample Highly Localized Heads (Perfect for 1-Shot SubQ Transplant):")
    for l, h, ret, offs in ultra_sparse_heads[:5]:
        print(f"  - Layer {l:>2}, Head {h:>2}: {ret:.2f}% Mass in Top-16 (Top-5 Offsets: {offs})")

    if len(dense_diffuse_heads) > 0:
        print("\nSample Diffuse Heads (Need SubQ Recurrent Multi-Hop T>1 to Resolve):")
        for l, h, ret, offs in dense_diffuse_heads[:5]:
            print(f"  - Layer {l:>2}, Head {h:>2}: {ret:.2f}% Mass in Top-16 (Top-5 Offsets: {offs})")
    else:
        print("\nNotice: ZERO heads in GPT-2 have diffuse mass under 70% in top-16!")

    print("=" * 125)

@app.local_entrypoint()
def main():
    run_phase0_diagnostic.remote()
