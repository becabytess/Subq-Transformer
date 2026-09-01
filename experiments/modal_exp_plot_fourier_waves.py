import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib",
        "seaborn"
    )
)

app = modal.App("exp-plot-fourier-waves", image=image)

@app.function(gpu="A10G", timeout=900)
def train_and_plot_fourier_waves():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 49 VISUALIZER: EXTRACTING & PLOTTING DYNAMIC RECURRENT FOURIER WAVES (T=4)")
    print("  Visualizing How the Model Synthesizes Frequencies, Phase Shifts, and Interference Patterns Across Hops")
    print("=" * 120)

    # 1. Download TinyShakespeare Dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    num_waves = 4
    T_hops = 4

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Model Definition
    class RecurrentGlobalFourierWaveLM(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, num_waves=4, T=4):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.num_waves = num_waves
            self.T = T

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.wave_synth = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_heads * num_waves * 4)
            )

            base_freqs = torch.tensor([1.0, 0.25, 0.0625, 0.015625]).view(1, 1, 1, 1, num_waves)
            self.register_buffer("base_freqs", base_freqs)

            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            q_idx = torch.arange(seq_len).unsqueeze(1)
            k_idx = torch.arange(seq_len).unsqueeze(0)
            d_matrix = torch.clamp(q_idx - k_idx, min=0).float()
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            self.register_buffer("d_matrix", d_matrix)
            self.register_buffer("causal_mask", causal_mask)

        def forward(self, idx, return_wave_diagnostics=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            d_mat = self.d_matrix[:L, :L].view(1, 1, L, L, 1)

            s = x
            diagnostics = []

            for step in range(self.T):
                pool_weights = F.softmax(self.summary_pool(s), dim=1)
                seq_summary = (s * pool_weights).sum(dim=1)

                wave_params = self.wave_synth(seq_summary).view(B, self.n_heads, self.num_waves, 4)
                
                amp = torch.tanh(wave_params[..., 0]).view(B, self.n_heads, 1, 1, self.num_waves)
                omega = (F.softplus(wave_params[..., 1]).view(B, self.n_heads, 1, 1, self.num_waves) * self.base_freqs)
                phi = (wave_params[..., 2] * math.pi).view(B, self.n_heads, 1, 1, self.num_waves)
                decay = (F.softplus(wave_params[..., 3]) * 0.05).view(B, self.n_heads, 1, 1, self.num_waves)

                wave_components = amp * torch.cos(omega * d_mat + phi) * torch.exp(-decay * d_mat)
                global_wave_bias = wave_components.sum(dim=-1)

                if return_wave_diagnostics:
                    diagnostics.append({
                        "step": step + 1,
                        "amp": amp.detach().cpu().numpy(),
                        "omega": omega.detach().cpu().numpy(),
                        "phi": phi.detach().cpu().numpy(),
                        "decay": decay.detach().cpu().numpy(),
                        "wave_components": wave_components.detach().cpu().numpy(),
                        "global_wave_bias": global_wave_bias.detach().cpu().numpy()
                    })

                Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                content_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
                total_scores = content_scores + global_wave_bias
                total_scores = total_scores.masked_fill(~self.causal_mask[:L, :L], float("-inf"))
                attn_weights = F.softmax(total_scores, dim=-1)

                H = (attn_weights @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                s = s + (1.0 / math.sqrt(self.T)) * self.out_proj(H)
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            logits = self.head(s)
            if return_wave_diagnostics:
                return logits, diagnostics
            return logits

    # 3. Train Model
    model = RecurrentGlobalFourierWaveLM(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_mlp=512, num_waves=num_waves, T=T_hops).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

    print("Training 2,000 steps on TinyShakespeare...")
    t0 = time.time()
    for step in range(2000):
        model.train()
        bx, by = get_batch('train')
        optimizer.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        if (step + 1) % 500 == 0 or (step + 1) == 2000:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):
                    vx, vy = get_batch('val')
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(v_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"  Step {step+1:>4}/2000 | Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | PPL: {math.exp(avg_val_loss):.2f} | Time: {time.time()-t0:.1f}s")

    # 4. Extract Wave Diagnostics on a Test Paragraph
    print("\n[Extracting Wave Parameters Across All T=4 Thinking Hops]...")
    model.eval()
    sample_text = "KING RICHARD:\nWhat is the matter, my lord? Why are you so sad?"
    sample_tokens = torch.tensor([char_to_ix.get(c, 0) for c in sample_text], dtype=torch.long, device=device).unsqueeze(0)
    # Pad to 256
    padded_tokens = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    padded_tokens[0, :len(sample_tokens[0])] = sample_tokens[0]

    with torch.no_grad():
        _, diagnostics = model(padded_tokens, return_wave_diagnostics=True)

    # 5. High-Resolution 4-Panel Visualization
    print("\n[Generating High-Resolution Wave Evolution Visualization]...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    d_range = np.arange(0, 64) # Look at distances d = 0 to 63

    # Panel A: Evolution of Total Interference Wave W^(t)(d) Across Hops (Head 1)
    ax_a = axes[0, 0]
    for step_idx, diag in enumerate(diagnostics):
        # global_wave_bias: [1, H, L, L] -> take row 63, cols 63-d
        # or reconstruct from parameters directly:
        h_idx = 0 # Head 1
        amp = diag["amp"][0, h_idx, 0, 0] # [N]
        omega = diag["omega"][0, h_idx, 0, 0] # [N]
        phi = diag["phi"][0, h_idx, 0, 0] # [N]
        decay = diag["decay"][0, h_idx, 0, 0] # [N]

        # W(d) = sum_m amp[m] * cos(omega[m]*d + phi[m]) * exp(-decay[m]*d)
        w_d = np.zeros(len(d_range))
        for m in range(num_waves):
            w_d += amp[m] * np.cos(omega[m] * d_range + phi[m]) * np.exp(-decay[m] * d_range)

        ax_a.plot(d_range, w_d, label=f"Hop t={step_idx+1}", color=palette[step_idx], linewidth=2.5)

    ax_a.set_title("A. Recurrent Wave Evolution Across Thinking Hops (Head 1)\nHow the Interference Pattern Reshapes from t=1 to t=4", fontsize=13, fontweight="bold")
    ax_a.set_xlabel("Relative Distance Lag d = |i - j| (Tokens Back)", fontsize=11)
    ax_a.set_ylabel("Harmonic Attention Bias W(d)", fontsize=11)
    ax_a.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=11, frameon=True)

    # Panel B: Decomposition of Constituent Harmonic Waves at Hop t=1 (Head 1)
    ax_b = axes[0, 1]
    diag_1 = diagnostics[0]
    h_idx = 0
    amp_1 = diag_1["amp"][0, h_idx, 0, 0]
    omega_1 = diag_1["omega"][0, h_idx, 0, 0]
    phi_1 = diag_1["phi"][0, h_idx, 0, 0]
    decay_1 = diag_1["decay"][0, h_idx, 0, 0]
    wave_names = ["Wave 1 (Local Syntax, High Freq)", "Wave 2 (Phrase Rhythm, Med Freq)", "Wave 3 (Clause Stride, Low Freq)", "Wave 4 (Global Envelope, DC Carrier)"]
    wave_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    w_sum = np.zeros(len(d_range))
    for m in range(num_waves):
        comp = amp_1[m] * np.cos(omega_1[m] * d_range + phi_1[m]) * np.exp(-decay_1[m] * d_range)
        w_sum += comp
        ax_b.plot(d_range, comp, label=f"{wave_names[m]} [ω={omega_1[m]:.3f}]", color=wave_colors[m], linestyle=":", linewidth=2.0)
    ax_b.plot(d_range, w_sum, label="TOTAL Interference Sum W(d)", color="black", linewidth=3.0)

    ax_b.set_title("B. Fourier Harmonic Decomposition (Hop t=1, Head 1)\n4 Pure Sinusoidal Oscillators Summed via Superposition", fontsize=13, fontweight="bold")
    ax_b.set_xlabel("Relative Distance Lag d = |i - j| (Tokens Back)", fontsize=11)
    ax_b.set_ylabel("Constituent Wave Amplitude", fontsize=11)
    ax_b.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=9, frameon=True)

    # Panel C: Multi-Head Functional Specialization at Hop t=4 (All 4 Heads)
    ax_c = axes[1, 0]
    diag_4 = diagnostics[-1] # Hop 4
    head_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for h in range(n_heads):
        amp_h = diag_4["amp"][0, h, 0, 0]
        omega_h = diag_4["omega"][0, h, 0, 0]
        phi_h = diag_4["phi"][0, h, 0, 0]
        decay_h = diag_4["decay"][0, h, 0, 0]

        w_h = np.zeros(len(d_range))
        for m in range(num_waves):
            w_h += amp_h[m] * np.cos(omega_h[m] * d_range + phi_h[m]) * np.exp(-decay_h[m] * d_range)
        ax_c.plot(d_range, w_h, label=f"Head {h+1}", color=head_colors[h], linewidth=2.5)

    ax_c.set_title("C. Multi-Head Functional Specialization (Final Hop t=4)\nEach Attention Head Establishes a Distinct Spatial Harmonic Highway", fontsize=13, fontweight="bold")
    ax_c.set_xlabel("Relative Distance Lag d = |i - j| (Tokens Back)", fontsize=11)
    ax_c.set_ylabel("Harmonic Attention Bias W(d)", fontsize=11)
    ax_c.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=11, frameon=True)

    # Panel D: 2D Spatial Attention Matrix Bias Heatmap (Head 1, Hop 4, L=64)
    ax_d = axes[1, 1]
    bias_2d = diag_4["global_wave_bias"][0, 0, :64, :64] # [64, 64]
    im = ax_d.imshow(bias_2d, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax_d, label="Wave Bias Amplitude")
    ax_d.set_title("D. 2D Continuous Harmonic Toeplitz Matrix (Head 1, Hop t=4)\nSmooth Diagonal Wave Fronts Propagating Across Sequence (L=64)", fontsize=13, fontweight="bold")
    ax_d.set_xlabel("Key Position j", fontsize=11)
    ax_d.set_ylabel("Query Position i", fontsize=11)

    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.getvalue()
    plt.close()

    return img_bytes

@app.local_entrypoint()
def main():
    import os
    print("Launching Wave Visualization on Modal...")
    img_bytes = train_and_plot_fourier_waves.remote()
    local_path = r"C:\Users\beca\.gemini\antigravity\brain\87f12cc9-4463-4972-82d9-e63736b3613e\recurrent_fourier_waves_evolution.png"
    with open(local_path, "wb") as f:
        f.write(img_bytes)
    print(f"✅ Plot successfully saved to: {local_path}")
