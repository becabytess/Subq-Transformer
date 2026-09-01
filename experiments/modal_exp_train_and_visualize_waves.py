import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-train-and-visualize-waves", image=image)

@app.function(gpu="A10G", timeout=2400)
def train_and_visualize():
    import math
    import time
    import urllib.request
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print("  STUDY 61: TRAINING CHAMPION HARMONIC SUBQ, CHECKPOINTING & PROFILING WAVE DYNAMICS")
    print("  Visualizing Continuous Waves, Top-K Peak Selections, Hop Evolution, and Token Receptive Fields")
    print("=" * 100)

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
    print(f"TinyShakespeare: {len(data):,} Characters | Vocab: {vocab_size}")

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    d_mlp = 512
    num_waves = 12
    K_peaks = 8
    max_d = 128
    T_hops = 8

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # Champion Model Architecture: Dynamical Wave Transition + Evolving Q,K,V
    # -------------------------------------------------------------------------
    class DynamicalWaveAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.num_waves, self.K_peaks, self.max_d = num_waves, K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.init_wave_latent = nn.Parameter(torch.randn(n_heads, num_waves * 4) * 0.1)

            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 64),
                nn.GELU(),
                nn.Linear(64, num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def get_wave_and_peaks(self, wave_latent):
            curr_params = wave_latent.view(self.n_heads, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, self.n_heads, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1) # [1, n_heads, max_d - 1]
            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1
            return wave_1d.squeeze(0), past_peak_offsets.squeeze(0), topk_vals.squeeze(0)

        def forward(self, x, s, wave_latent):
            B, L, D = x.shape
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            curr_params = wave_latent.view(self.n_heads, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, self.n_heads, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1

            zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=x.device)
            zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=x.device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets.unsqueeze(2)
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k) + peak_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)

            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return self.out_proj(out), next_wave_latent, attn_weights, peak_offsets

    class HarmonicSubQLM(nn.Module):
        def __init__(self, vocab_size, T=8):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = DynamicalWaveAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.T = T

        def forward(self, idx, return_traces=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x

            w = self.attn.init_wave_latent
            traces = []

            for step in range(self.T):
                if return_traces:
                    w_curr = w.clone()
                attn_out, w, weights, offsets = self.attn(x, s, wave_latent=w)
                if return_traces:
                    traces.append({
                        "step": step + 1,
                        "wave_latent": w_curr.detach().cpu(),
                        "attn_weights": weights.detach().cpu(),
                        "peak_offsets": offsets.detach().cpu()
                    })
                s = s + (1.0 / math.sqrt(self.T)) * attn_out
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            logits = self.head(s)
            if return_traces:
                return logits, traces
            return logits

    # -------------------------------------------------------------------------
    # Train the Model
    # -------------------------------------------------------------------------
    torch.manual_seed(42)
    model = HarmonicSubQLM(vocab_size=vocab_size, T=T_hops).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nTraining Harmonic SubQ (T={T_hops} Hops, Params: {param_count:,})...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

    t0 = time.time()
    for step in range(2000):
        model.train()
        bx, by = get_batch('train', step_seed=10000 + step)
        optimizer.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1:4d}/2000 | Train Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        val_losses = []
        for v_step in range(30):
            vx, vy = get_batch('val', step_seed=90000 + v_step)
            v_logits = model(vx)
            v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
            val_losses.append(v_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        ppl = math.exp(avg_val_loss)

    print(f"\n--> Training Complete! Val Loss: {avg_val_loss:.4f} | Val PPL: {ppl:.2f} | Time: {elapsed:.1f}s")

    # Serialize Checkpoint
    checkpoint_buffer = io.BytesIO()
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "char_to_ix": char_to_ix,
        "ix_to_char": ix_to_char,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_waves": num_waves,
        "K_peaks": K_peaks,
        "T_hops": T_hops,
        "val_loss": avg_val_loss,
        "val_ppl": ppl
    }, checkpoint_buffer)
    checkpoint_bytes = checkpoint_buffer.getvalue()
    print(f"Checkpoint Serialized: {len(checkpoint_bytes):,} Bytes (~{len(checkpoint_bytes)/(1024*1024):.2f} MB)")

    # -------------------------------------------------------------------------
    # Profile Wave Curves and Token Selection on Validation Sequence
    # -------------------------------------------------------------------------
    print("\nExtracting Wave Curves and Token-Level Receptive Field...")
    torch.manual_seed(999)
    sample_x, sample_y = get_batch('val', step_seed=42)
    sample_text = "".join([ix_to_char[idx.item()] for idx in sample_x[0]])
    
    with torch.no_grad():
        _, traces = model(sample_x[:1], return_traces=True)

    # Visualization Setup
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    fig.suptitle(f"Harmonic SubQ (T={T_hops} Hops, K={K_peaks} Peaks, Val PPL: {ppl:.2f})\nLearned Continuous Wave Curves, Peak Extraction & Token Receptive Field", fontsize=15, fontweight='bold', y=0.98)

    d_vals = np.arange(1, max_d)
    head_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    head_names = ['Head 0 (Local Bigram)', 'Head 1 (Clause Cadence)', 'Head 2 (Meter/Verse)', 'Head 3 (Discourse Anchor)']

    # Panel 1: Multi-Head Continuous Wave Curves at Hop t=1
    ax1 = axes[0, 0]
    w1_latent = traces[0]["wave_latent"].to(device)
    wave_1d_h, past_peaks_h, topk_vals_h = model.attn.get_wave_and_peaks(w1_latent)
    wave_1d_np = wave_1d_h.cpu().numpy()
    past_peaks_np = past_peaks_h.cpu().numpy()

    for h in range(n_heads):
        ax1.plot(d_vals, wave_1d_np[h], color=head_colors[h], linewidth=2.0, label=f"{head_names[h]}")
        # Mark the extracted peaks on the wave
        peaks = past_peaks_np[h]
        for p in peaks:
            ax1.scatter(p, wave_1d_np[h, p - 1], color=head_colors[h], s=45, zorder=5, edgecolors='black', linewidth=0.8)

    ax1.set_title("A. Continuous Harmonic Interference Waves $W_h(d)$ (Hop $t=1$)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Relative Distance $d$ (Tokens Backward)", fontsize=11)
    ax1.set_ylabel("Wave Energy Amplitude $W(d)$", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Panel 2: Selected Top-K Discrete Offsets per Head
    ax2 = axes[0, 1]
    for h in range(n_heads):
        full_offsets = [0] + sorted(past_peaks_np[h].tolist())
        ax2.scatter(full_offsets, [h] * len(full_offsets), color=head_colors[h], s=120, edgecolors='black', linewidth=1.2, label=f"Head {h}")
        for off in full_offsets:
            ax2.text(off, h + 0.15, f"d={off}", ha='center', fontsize=8, color=head_colors[h], fontweight='bold')

    ax2.set_title("B. Top-$K$ Extracted Peak Offset Grid per Head ($K=8$ Tokens)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Selected Offset Distance $d$", fontsize=11)
    ax2.set_yticks(range(n_heads))
    ax2.set_yticklabels([f"Head {h}" for h in range(n_heads)], fontsize=10, fontweight='bold')
    ax2.set_xlim(-2, 130)
    ax2.set_ylim(-0.5, 3.8)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')

    # Panel 3: Dynamical Wave Evolution Across Thinking Hops t = 1, 2, 4, 8 (Head 0)
    ax3 = axes[1, 0]
    hop_indices = [1, 2, 4, 8]
    hop_colors = ['#4575b4', '#74add1', '#f46d43', '#d73027']
    for idx_hop, t_step in enumerate(hop_indices):
        w_t_latent = traces[t_step - 1]["wave_latent"].to(device)
        w_t_1d, _, _ = model.attn.get_wave_and_peaks(w_t_latent)
        w_t_np = w_t_1d.cpu().numpy()[0] # Head 0
        ax3.plot(d_vals, w_t_np, color=hop_colors[idx_hop], linewidth=2.0, label=f"Hop $t={t_step}$")

    ax3.set_title("C. Dynamical Wave Evolution Across Hops $t=1 \\to 2 \\to 4 \\to 8$ (Head 0)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Relative Distance $d$ (Tokens Backward)", fontsize=11)
    ax3.set_ylabel("Wave Energy Amplitude $W(d)$", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Panel 4: Concrete Token Receptive Field Map on Shakespeare Sentence
    ax4 = axes[1, 1]
    target_pos = 115
    query_char = sample_text[target_pos]
    ax4.axis('off')

    # Prepare token routing text summary
    snippet_start = max(0, target_pos - 45)
    snippet = sample_text[snippet_start:target_pos+1].replace('\n', '↵')
    
    info_text = f"D. Concrete Linguistic Token Receptive Field\n"
    info_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    info_text += f"Query Position {target_pos}: '{query_char}' (in context: \"...{snippet}\")\n\n"

    for h in range(n_heads):
        peaks = [0] + sorted(past_peaks_np[h].tolist())
        target_tokens = []
        for d_off in peaks:
            src_idx = target_pos - d_off
            if src_idx >= 0:
                c = sample_text[src_idx].replace('\n', '↵')
                target_tokens.append(f"d={d_off:<2}: '{c}'")
        tokens_str = ",  ".join(target_tokens[:4]) + "\n       " + ",  ".join(target_tokens[4:])
        info_text += f"• {head_names[h]}:\n   {tokens_str}\n\n"

    ax4.text(0.02, 0.95, info_text, transform=ax4.transAxes, fontsize=9.5, fontfamily='monospace', va='top', bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', linewidth=1.5))

    plt.tight_layout()
    plot_buffer = io.BytesIO()
    plt.savefig(plot_buffer, format='png', bbox_inches='tight')
    plt.close()
    plot_bytes = plot_buffer.getvalue()

    return {
        "val_loss": avg_val_loss,
        "val_ppl": ppl,
        "elapsed": elapsed,
        "checkpoint_bytes": checkpoint_bytes,
        "plot_bytes": plot_bytes,
        "sample_text_snippet": sample_text[target_pos-50:target_pos+10]
    }

@app.local_entrypoint()
def main():
    res = train_and_visualize.remote()
    
    # Save checkpoint locally
    ckpt_path = "checkpoints/best_harmonic_subq_t8.pt"
    with open(ckpt_path, "wb") as f:
        f.write(res["checkpoint_bytes"])
    print(f"--> Saved checkpoint locally to {ckpt_path} ({len(res['checkpoint_bytes']):,} bytes)")

    # Save artifact plot locally
    artifact_plot_path = r"C:\Users\beca\.gemini\antigravity\brain\87f12cc9-4463-4972-82d9-e63736b3613e\harmonic_subq_waves_and_token_routing.png"
    with open(artifact_plot_path, "wb") as f:
        f.write(res["plot_bytes"])
    print(f"--> Saved publication figure to {artifact_plot_path}")
