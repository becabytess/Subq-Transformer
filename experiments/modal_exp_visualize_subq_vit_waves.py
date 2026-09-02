import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib",
        "seaborn"
    )
)

app = modal.App("exp-visualize-subq-vit-waves", image=image)
volume = modal.Volume.from_name("subq-models-vol")

@app.function(gpu="A10G", timeout=600, volumes={"/models": volume})
def extract_and_plot_learned_waves():
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import io

    # Model Architecture Specifications
    d_model = 192
    n_heads = 6
    head_dim = d_model // n_heads # 32
    seq_len = 257 # 256 patches + 1 [CLS]
    num_classes = 100
    K_peaks = 8
    num_waves = 12
    T_train = 4

    # 1. Define Architecture
    class HighResPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

    class HarmonicSubQViT(nn.Module):
        def __init__(self, T_train=4):
            super().__init__()
            self.T_train = T_train
            self.patch_embed = HighResPatchEmbed()
            self.ln_1 = nn.LayerNorm(d_model)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            init_latents = torch.zeros(n_heads, num_waves, 4)
            init_latents[..., 0] = 0.5
            init_latents[..., 3] = 0.1
            self.init_wave_latent = nn.Parameter(init_latents.view(n_heads, num_waves * 4))
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )
            log_freqs = torch.linspace(math.log10(math.pi / 1.0), math.log10(math.pi / 64.0), num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.max_d = seq_len
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, self.max_d - 1, 1))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

    print("Loading checkpoint from /models/harmonic_subq_vit_T4.pt...")
    model = HarmonicSubQViT(T_train=T_train)
    checkpoint_path = "/models/harmonic_subq_vit_T4.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Successfully loaded Harmonic SubQ ViT Checkpoint!")

    # 2. Unroll Wave Evolution Across T = 1, 2, 3, 4
    waves_by_t = [] # [T, n_heads, seq_len - 1]
    offsets_by_t = [] # [T, n_heads, K_peaks]
    biases_by_t = [] # [T, n_heads, K_peaks]

    curr_wave = model.init_wave_latent
    d_grid = model.d_grid # [1, 1, 256, 1]
    base_freqs = model.base_freqs # [1, 1, 1, 12]

    with torch.no_grad():
        for t in range(T_train):
            params = curr_wave.view(n_heads, num_waves, 4)
            amp = torch.tanh(params[..., 0]).view(1, n_heads, 1, num_waves)
            omega = (F.softplus(params[..., 1]).view(1, n_heads, 1, num_waves) * base_freqs)
            phi = (params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
            decay = (F.softplus(params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

            wave_comps = amp * torch.cos(omega * d_grid + phi) * torch.exp(-decay * d_grid)
            wave_1d = wave_comps.sum(dim=-1).squeeze(0) # [n_heads, 256]

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1
            zero_offset = torch.zeros((n_heads, 1), dtype=torch.long)
            zero_val = torch.zeros((n_heads, 1), dtype=torch.float)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            waves_by_t.append(wave_1d.numpy())
            offsets_by_t.append(peak_offsets.numpy())
            biases_by_t.append(peak_vals.numpy())

            if t < T_train - 1:
                curr_wave = curr_wave + 0.1 * model.wave_transition(curr_wave)

    # 3. Create High-Quality Multi-Panel Visualization
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 16), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1.2, 1.2], hspace=0.35, wspace=0.25)

    head_colors = ['#00FFCC', '#FF3366', '#3399FF', '#FFCC00', '#CC66FF', '#00FF66']
    t_colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B']

    # --- PANEL 1: Continuous Wave Profiles by Head Across Thinking Hops (T=1 vs T=4) ---
    ax1 = fig.add_subplot(gs[0, :2])
    d_axis = np.arange(1, seq_len)
    for h in range(n_heads):
        # Plot T=1 (initial) and T=4 (deep thought)
        ax1.plot(d_axis, waves_by_t[0][h], label=f"Head {h+1} (t=1)", color=head_colors[h], alpha=0.4, linestyle='--')
        ax1.plot(d_axis, waves_by_t[3][h], label=f"Head {h+1} (t=4)", color=head_colors[h], linewidth=2.0)

    ax1.set_title("Learned Continuous Harmonic Waveforms Across Attention Heads (Distance d in [1, 256])\nSolid = Final Thinking Hop (t=4), Dashed = Initial Hop (t=1)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Token Offset Distance d (16x16 2D Patch Grid)", fontsize=11)
    ax1.set_ylabel("Wave Interference Amplitude W(d)", fontsize=11)
    ax1.set_xlim(1, 256)
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.legend(loc='upper right', ncol=3, fontsize=9, framealpha=0.7)

    # Mark spatial grid frequencies (row jumps: 16, 32, 48, 64)
    for row_step in [16, 32, 48, 64, 128]:
        ax1.axvline(row_step, color='#888888', linestyle=':', alpha=0.5)
        ax1.text(row_step + 1, ax1.get_ylim()[1]*0.85, f"d={row_step}\n(row {row_step//16})", color='#AAAAAA', fontsize=7)

    # --- PANEL 2: Dynamic Wave Evolution for a Specialized Head Across t=1..4 ---
    ax2 = fig.add_subplot(gs[0, 2])
    target_head = 0 # Head 1
    for t in range(T_train):
        ax2.plot(d_axis[:64], waves_by_t[t][target_head][:64], label=f"Hop t={t+1}", color=t_colors[t], linewidth=2.2)
    ax2.set_title(f"Head {target_head+1} Carrier Wave Transition (t=1 -> 4)\n(Zoom: d in [1, 64])", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Distance d (First 4 Rows of Patches)", fontsize=10)
    ax2.set_ylabel("Amplitude", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    # --- PANEL 3: Active Discrete Peak Offsets Pattern (Heatmap of Offsets by Head & Step) ---
    ax3 = fig.add_subplot(gs[1, :])
    # Build offset table for display
    offset_matrix = np.zeros((n_heads * T_train, K_peaks))
    y_labels = []
    for h in range(n_heads):
        for t in range(T_train):
            row_idx = h * T_train + t
            sorted_offsets = np.sort(offsets_by_t[t][h])
            offset_matrix[row_idx, :] = sorted_offsets
            y_labels.append(f"H{h+1}, t={t+1}")

    im3 = ax3.imshow(offset_matrix, aspect='auto', cmap='magma', interpolation='nearest')
    ax3.set_title("Active Learned Sparse Peak Offsets (K=8 per Token) Across All Heads & Thinking Hops t=1..4", fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel("Peak Rank (k = 0 to 7)", fontsize=11)
    ax3.set_ylabel("Head & Thinking Hop", fontsize=11)
    ax3.set_yticks(np.arange(len(y_labels)))
    ax3.set_yticklabels(y_labels, fontsize=8)
    ax3.set_xticks(np.arange(K_peaks))
    ax3.set_xticklabels([f"Offset {k+1}" for k in range(K_peaks)], fontsize=10)

    # Overlay numeric values on heatmap
    for r in range(offset_matrix.shape[0]):
        for c in range(offset_matrix.shape[1]):
            val = int(offset_matrix[r, c])
            ax3.text(c, r, f"{val}", ha="center", va="center", color="white" if val < 150 else "black", fontsize=8, fontweight='bold')

    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.02)
    cbar3.set_label("Spatial Offset Distance (0 = Self)", fontsize=10)

    # --- PANEL 4: 2D Spatial Receptive Field Projection on 16x16 Patch Grid ---
    # Show where a center patch (e.g. (8, 8) = index 136) routes its 8 attention peaks
    ax4 = fig.add_subplot(gs[2, 0])
    center_idx = 136 # Center patch at (row 8, col 8)
    grid_2d_t1 = np.zeros((16, 16))
    grid_2d_t4 = np.zeros((16, 16))

    # Center token position in 2D
    c_row, c_col = center_idx // 16, center_idx % 16

    for h in range(n_heads):
        offs_t1 = offsets_by_t[0][h]
        offs_t4 = offsets_by_t[3][h]
        for off in offs_t1:
            target_idx = (center_idx - off) % 256
            grid_2d_t1[target_idx // 16, target_idx % 16] += 1
        for off in offs_t4:
            target_idx = (center_idx - off) % 256
            grid_2d_t4[target_idx // 16, target_idx % 16] += 1

    im4 = ax4.imshow(grid_2d_t1, cmap='viridis', interpolation='nearest')
    ax4.scatter([c_col], [c_row], color='red', s=100, marker='*', label='Center Query Token')
    ax4.set_title("2D Spatial Routing at Hop t=1\n(Initial Local & Lattice Offsets)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Patch Column (0..15)", fontsize=9)
    ax4.set_ylabel("Patch Row (0..15)", fontsize=9)
    ax4.legend(loc='upper right', fontsize=8)

    ax5 = fig.add_subplot(gs[2, 1])
    im5 = ax5.imshow(grid_2d_t4, cmap='viridis', interpolation='nearest')
    ax5.scatter([c_col], [c_row], color='red', s=100, marker='*', label='Center Query Token')
    ax5.set_title("2D Spatial Routing at Hop t=4\n(Evolved Multi-Scale Resonances)", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Patch Column (0..15)", fontsize=9)
    ax5.set_ylabel("Patch Row (0..15)", fontsize=9)
    ax5.legend(loc='upper right', fontsize=8)

    # --- PANEL 6: Offset Distribution Histogram (Local vs Vertical Grid vs Global) ---
    ax6 = fig.add_subplot(gs[2, 2])
    all_offsets_t1 = np.concatenate([offsets_by_t[0][h] for h in range(n_heads)])
    all_offsets_t4 = np.concatenate([offsets_by_t[3][h] for h in range(n_heads)])

    ax6.hist(all_offsets_t1, bins=32, range=(0, 256), alpha=0.5, color='#00FFCC', label='t=1 Offsets')
    ax6.hist(all_offsets_t4, bins=32, range=(0, 256), alpha=0.5, color='#FF3366', label='t=4 Offsets')
    ax6.axvline(1, color='yellow', linestyle='--', label='d=1 (Horizontal Neighbor)')
    ax6.axvline(16, color='orange', linestyle='--', label='d=16 (Vertical Neighbor)')
    ax6.axvline(32, color='magenta', linestyle='--', label='d=32 (2-Row Jump)')
    ax6.set_title("Spatial Distance Histogram of Learned Peaks\nLocal + 2D Row Jumps + Global Resonance", fontsize=11, fontweight='bold')
    ax6.set_xlabel("Offset Distance d", fontsize=9)
    ax6.set_ylabel("Peak Count Across Heads", fontsize=9)
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, linestyle=':', alpha=0.3)

    plt.suptitle("HARMONIC SUBQ VISION TRANSFORMER: LEARNED 2D SPATIAL WAVES & RECURRENT HOP EVOLUTION\nLoaded from Saved Model Checkpoint (/models/harmonic_subq_vit_T4.pt)", fontsize=16, fontweight='bold', y=0.99)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.getvalue()

    # Also prepare textual summary of offsets by head
    summary_text = []
    summary_text.append("=" * 90)
    summary_text.append("  LEARNED ACTIVE PEAK OFFSETS PER ATTENTION HEAD (K=8 Peaks per Token)")
    summary_text.append("=" * 90)
    for h in range(n_heads):
        summary_text.append(f"\n--- ATTENTION HEAD {h+1} ---")
        for t in range(T_train):
            offs = np.sort(offsets_by_t[t][h])
            biases = [f"{b:.2f}" for b in biases_by_t[t][h]]
            summary_text.append(f"  Hop t={t+1}: Offsets = {offs.tolist()} | Logit Biases = {biases}")

    return png_bytes, "\n".join(summary_text)

@app.local_entrypoint()
def main():
    png_bytes, summary_str = extract_and_plot_learned_waves.remote()
    print(summary_str)

    # Save to local brain artifacts folder
    artifact_path = "C:/Users/beca/.gemini/antigravity/brain/87f12cc9-4463-4972-82d9-e63736b3613e/harmonic_subq_vit_learned_waves_and_2d_offsets.png"
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "wb") as f:
        f.write(png_bytes)
    print(f"\n✓ Saved high-resolution visualization to {artifact_path}")
