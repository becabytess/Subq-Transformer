import modal
import os
import base64

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "numpy",
        "matplotlib",
        "seaborn"
    )
)

app = modal.App("exp-gpt2-attention-distribution", image=image)

@app.function(gpu="A10G", timeout=600)
def profile_gpt2_attention_distribution():
    import json
    import math
    import urllib.request
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  EMPIRICAL ATTENTION DISTANCE DISTRIBUTION PROFILING ON GPT-2 (12 Layers x 12 Heads = 144 Heads)")
    print("  Mapping Spatial Mass P(d = |i - j|) Across Natural Language Sequences (L=512)")
    print("=" * 120)

    # 1. Download WikiText-2 Sample Text
    print("\n[1/4] Fetching Natural Language Evaluation Corpus...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    req = urllib.request.urlopen(url)
    raw_text = req.read().decode('utf-8')
    
    # 2. Load Pretrained GPT-2
    print("\n[2/4] Loading Pretrained GPT-2...")
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        output_attentions=True,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()

    all_tokens = tokenizer.encode(raw_text)
    seq_len = 512
    num_eval_sequences = 60
    token_chunks = [all_tokens[i * seq_len : (i + 1) * seq_len] for i in range(num_eval_sequences)]
    print(f"Prepared {len(token_chunks)} evaluation chunks of length L={seq_len} ({len(token_chunks)*seq_len:,} total tokens).")

    num_layers = model.config.n_layer   # 12
    num_heads = model.config.n_head     # 12
    total_heads = num_layers * num_heads # 144
    seq_len = 512
    max_dist = seq_len

    # Accumulator tensors
    # dist_mass_by_head: [num_layers, num_heads, max_dist]
    dist_mass_by_head = np.zeros((num_layers, num_heads, max_dist), dtype=np.float64)
    bos_mass_by_head = np.zeros((num_layers, num_heads), dtype=np.float64)
    total_queries_by_head = np.zeros((num_layers, num_heads), dtype=np.float64)

    # 3. Profile Attention Matrices across Sequences
    print("\n[3/4] Extracting Attention Matrices across 50 Sequences (L=512)...")
    num_eval_sequences = 50
    curr_idx = 0
    seqs_processed = 0

    with torch.no_grad():
        for chunk in token_chunks:
            if seqs_processed >= num_eval_sequences:
                break
            
            input_ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
            outputs = model(input_ids)
            # outputs.attentions is a tuple of 12 tensors, each [1, 12, 512, 512]
            attentions = outputs.attentions

            for l_idx, layer_attn in enumerate(attentions):
                # layer_attn: [1, 12, 512, 512]
                attn_np = layer_attn[0].cpu().numpy() # [12, 512, 512]

                for h_idx in range(num_heads):
                    head_mat = attn_np[h_idx] # [512, 512] (causal lower triangular)

                    for i in range(seq_len):
                        row = head_mat[i, : i + 1] # Attention from query i to all past tokens j <= i
                        # Distance d = i - j
                        distances = i - np.arange(i + 1) # [0, 1, 2, ..., i]
                        for d_val, mass in zip(distances, row):
                            dist_mass_by_head[l_idx, h_idx, d_val] += mass
                        bos_mass_by_head[l_idx, h_idx] += row[0] # token 0 (BOS / Anchor)
                        total_queries_by_head[l_idx, h_idx] += 1.0

            seqs_processed += 1
            if seqs_processed % 10 == 0:
                print(f"  Processed {seqs_processed:>2}/{num_eval_sequences} sequences...")

    # Normalize by total queries to get probability distributions P(d)
    # P_head[l, h, d]
    prob_by_head = np.zeros_like(dist_mass_by_head)
    for l in range(num_layers):
        for h in range(num_heads):
            q_count = total_queries_by_head[l, h]
            if q_count > 0:
                prob_by_head[l, h] = dist_mass_by_head[l, h] / q_count
                bos_mass_by_head[l, h] /= q_count

    # Global Mean Distance Distribution \bar{P}(d)
    global_p_dist = prob_by_head.mean(axis=(0, 1)) # [512]
    # Cumulative Mass C(d)
    cumulative_p = np.cumsum(global_p_dist)

    # 4. Generate Comprehensive Visualizations
    print("\n[4/4] Generating High-Resolution Analysis Plots...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), dpi=150)

    # Plot 1: Global Distance Distribution \bar{P}(d) vs Distance d (Log-Log and Log-X)
    ax1 = axs[0, 0]
    distances = np.arange(seq_len)
    ax1.plot(distances[:128], global_p_dist[:128], color="#1f77b4", lw=2.5, label=r"Empirical Global $\bar{P}(d)$")
    subq_offsets = [0, 1, 2, 4, 8, 16, 32, 64]
    ax1.scatter(subq_offsets, [global_p_dist[d] for d in subq_offsets], color="#d62728", s=80, zorder=5, label="SubQ Offsets Menu")
    for d in subq_offsets:
        ax1.annotate(f"d={d}\n({global_p_dist[d]*100:.1f}%)", (d, global_p_dist[d]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, fontweight='bold', color="#d62728")
    ax1.set_title("1. Empirical Attention Mass vs. Distance (d = |i - j|)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Relative Distance (d tokens back)", fontsize=12)
    ax1.set_ylabel("Average Attention Probability P(d)", fontsize=12)
    ax1.set_xlim(-2, 70)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Cumulative Attention Mass vs Distance & SubQ Coverage
    ax2 = axs[0, 1]
    ax2.plot(distances[:256], cumulative_p[:256] * 100.0, color="#2ca02c", lw=2.5, label="Cumulative Mass C(d)")
    ax2.axhline(50, color="gray", linestyle=":", label="50% Threshold")
    ax2.axhline(80, color="orange", linestyle=":", label="80% Threshold")
    ax2.axhline(90, color="red", linestyle=":", label="90% Threshold")
    for d in [1, 2, 4, 8, 16, 32, 64, 128]:
        ax2.scatter([d], [cumulative_p[d] * 100.0], color="#2ca02c", s=60)
        ax2.annotate(f"{cumulative_p[d]*100:.1f}%", (d, cumulative_p[d]*100), textcoords="offset points", xytext=(5, -12), fontsize=8, fontweight="bold")
    ax2.set_title("2. Cumulative Attention Mass Captured by Distance Horizon", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Distance Horizon d (tokens)", fontsize=12)
    ax2.set_ylabel("Cumulative Mass (%)", fontsize=12)
    ax2.set_xlim(0, 130)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Layer-by-Layer Attention Mass Heatmap (Layers 0..11 vs Distance Scales)
    ax3 = axs[1, 0]
    distance_bins = ["d=0 (Self)", "d=1 (Prev)", "d=2..3", "d=4..7", "d=8..15", "d=16..31", "d=32..63", "d>=64", "BOS (j=0)"]
    layer_binned_mass = np.zeros((num_layers, len(distance_bins)))

    for l in range(num_layers):
        l_prob = prob_by_head[l].mean(axis=0) # [512]
        layer_binned_mass[l, 0] = l_prob[0]
        layer_binned_mass[l, 1] = l_prob[1]
        layer_binned_mass[l, 2] = l_prob[2:4].sum()
        layer_binned_mass[l, 3] = l_prob[4:8].sum()
        layer_binned_mass[l, 4] = l_prob[8:16].sum()
        layer_binned_mass[l, 5] = l_prob[16:32].sum()
        layer_binned_mass[l, 6] = l_prob[32:64].sum()
        layer_binned_mass[l, 7] = l_prob[64:].sum()
        layer_binned_mass[l, 8] = bos_mass_by_head[l].mean()

    sns.heatmap(layer_binned_mass * 100.0, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=distance_bins, yticklabels=[f"Layer {l}" for l in range(num_layers)], ax=ax3, cbar_kws={'label': 'Attention Mass (%)'})
    ax3.set_title("3. Layer-by-Layer Attention Distribution Profile (Depth Evolution)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Distance Category", fontsize=12)
    ax3.set_ylabel("Layer Depth", fontsize=12)

    # Plot 4: Head Clustering & Taxonomy across all 144 Heads
    ax4 = axs[1, 1]
    head_types = {"Ultra-Local (d<=2 > 50%)": 0, "Syntactic (d=3..16 > 30%)": 0, "BOS/Anchor (j=0 > 25%)": 0, "Broad/Long-Range (d>32 > 30%)": 0, "Mixed / Balanced": 0}
    
    head_scatter_x = []
    head_scatter_y = []
    head_colors = []

    for l in range(num_layers):
        for h in range(num_heads):
            hp = prob_by_head[l, h]
            local_mass = hp[:3].sum()
            bos_mass = bos_mass_by_head[l, h]
            syntactic_mass = hp[3:17].sum()
            long_mass = hp[32:].sum()

            head_scatter_x.append(local_mass * 100.0)
            head_scatter_y.append(long_mass * 100.0)

            if local_mass > 0.50:
                head_types["Ultra-Local (d<=2 > 50%)"] += 1
                head_colors.append("#1f77b4")
            elif bos_mass > 0.25:
                head_types["BOS/Anchor (j=0 > 25%)"] += 1
                head_colors.append("#d62728")
            elif syntactic_mass > 0.30:
                head_types["Syntactic (d=3..16 > 30%)"] += 1
                head_colors.append("#2ca02c")
            elif long_mass > 0.30:
                head_types["Broad/Long-Range (d>32 > 30%)"] += 1
                head_colors.append("#9467bd")
            else:
                head_types["Mixed / Balanced"] += 1
                head_colors.append("#ff7f0e")

    ax4.scatter(head_scatter_x, head_scatter_y, c=head_colors, s=70, alpha=0.85, edgecolors="k", lw=0.5)
    ax4.set_title("4. Head Taxonomy Across All 144 Attention Heads", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Local Mass (%) [d <= 2]", fontsize=12)
    ax4.set_ylabel("Long-Range Mass (%) [d >= 32]", fontsize=12)
    ax4.grid(True, linestyle="--", alpha=0.6)

    # Add text summary box in ax4
    tax_text = "\n".join([f"• {k}: {v} heads ({v/total_heads*100:.1f}%)" for k, v in head_types.items()])
    ax4.text(0.40, 0.65, tax_text, transform=ax4.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plot_path = "/tmp/gpt2_attention_distance_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    with open(plot_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 5. Print Quantitative Terminal Report
    print("\n" + "=" * 120)
    print("  QUANTITATIVE ATTENTION DISTANCE REPORT (GPT-2 144 HEADS)")
    print("=" * 120)
    print(f"Top 10 Exact Distances by Probability Mass:")
    for d in range(10):
        print(f"  Distance d={d:>2} (t-{d:>2}) : {global_p_dist[d]*100:>6.2f}% | Cumulative: {cumulative_p[d]*100:>6.2f}%")

    print(f"\nSubQ Logarithmic Grid Coverage:")
    for d in subq_offsets:
        print(f"  Offset d={d:>2} : Probability = {global_p_dist[d]*100:>6.2f}%")

    print(f"\nHead Taxonomy Summary (144 Heads):")
    for k, v in head_types.items():
        print(f"  {k:<35} : {v:>3} heads ({v/total_heads*100:>5.1f}%)")

    return {
        "global_p_dist": global_p_dist[:128].tolist(),
        "cumulative_p": cumulative_p[:128].tolist(),
        "head_types": head_types,
        "image_b64": img_b64
    }

@app.local_entrypoint()
def main():
    res = profile_gpt2_attention_distribution.remote()
    
    # Save image locally
    out_dir = r"C:\Users\beca\.gemini\antigravity\brain\87f12cc9-4463-4972-82d9-e63736b3613e"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gpt2_attention_distance_distribution.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(res["image_b64"]))
    print(f"\n✅ Plot saved locally to: {out_path}")
