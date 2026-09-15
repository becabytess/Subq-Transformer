"""S3-023: Deep Iterative Thought Scaling at T=12 Hops on TinyShakespeare.

Tests whether pushing thinking depth from T=8 to T=12 hops in Inverted Query-Escrow FEN-SubQ
continues to improve validation loss and perplexity (beyond the 5.16 PPL record in S3-022),
or whether representations and routing saturate.

Diagnostics Tracked Across All 12 Hops:
1. Token State Velocity: v(t) = ||s^(t) - s^(t-1)|| / ||s^(t)||
2. State Cosine Alignment: cos(s^(t), s^(t-1))
3. Wave Spatial Decay & Reach: lambda(t) and R(t) = 1 / lambda(t)
4. Waveform Acceleration & Energy Delta: ||W^(t) - W^(t-1)|| and cos(W^(t), W^(t-1))
5. Weighted Mean Attended Distance: d_bar(t)
6. Attention Softmax Entropy: H(t)
Generates a publication-quality 4-panel matplotlib visualization.
"""

import json
import math
import os
import time
import urllib.request
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy", "matplotlib"
)
app = modal.App("season3-s3-023-t12-scaling")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_t12_experiment(
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 500,
    val_batches: int = 30,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    n_hops: int = 12,
    k_peaks: int = 8,
    num_waves: int = 12,
    mlp_ratio: int = 2,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load TinyShakespeare
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    print("=" * 105)
    print(f"  S3-023: DEEP ITERATIVE THOUGHT SCALING AT T = {n_hops} HOPS (INVERTED QUERY-ESCROW)")
    print(f"  TinyShakespeare: {len(data):,} chars | Vocab: {vocab_size} | Hops T = {n_hops} | K = {k_peaks}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

    def get_batch(split="train", gen=None):
        src = train_data if split == "train" else val_data
        max_idx = len(src) - seq_len - 1
        if gen is not None:
            ix = torch.randint(0, max_idx, (batch_size,), generator=gen)
        else:
            ix = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([src[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([src[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # 2. Inverted Query-Escrow Model
    class InvertedQueryEscrowLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_hops = n_hops
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = seq_len

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Phase 1: Causal cuDNN GRU Scanner + FEN Query Extractor
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.q_proj = nn.Linear(d_model, d_model)

            # Phase 2: Discrete Key and Value Projections
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            # Harmonic Wave Router
            self.init_wave_latent = nn.Parameter(torch.randn(num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4),
            )

            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, self.max_d - 1, 1))

            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            mlp_dim = d_model * mlp_ratio
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model),
            )

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, return_dynamics: bool = False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Phase 1: Causal Scan -> Query Escrow Vault
            H, _ = self.scanner(x)                     # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            Q_raw = self.q_proj(G * H)                 # [B, L, D]
            E_q_all = torch.cumsum(Q_raw, dim=1)       # [B, L, D]

            # Phase 2: Multi-Hop Retrieval over Discrete Tokens
            state = H
            curr_wave = self.init_wave_latent
            pos_grid = torch.arange(L, device=idx.device).unsqueeze(1)

            diagnostics = {
                "state_velocities": [],
                "state_cosine_alignments": [],
                "wave_decays": [],
                "wave_reaches": [],
                "wave_energy_deltas": [],
                "wave_cosines": [],
                "mean_attended_distances": [],
                "hop_entropies": [],
                "waveforms": [],
                "active_offsets": [],
            }
            prev_state = state
            prev_wave_1d = None

            for hop in range(self.n_hops):
                params = curr_wave.view(self.num_waves, 4)
                amp = torch.tanh(params[:, 0]).view(1, 1, self.num_waves)
                omega = (F.softplus(params[:, 1]).view(1, 1, self.num_waves) * self.base_freqs)
                phi = (params[:, 2] * math.pi).view(1, 1, self.num_waves)
                decay = (F.softplus(params[:, 3]) * 0.05).view(1, 1, self.num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).squeeze(0)  # [max_d - 1]

                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros(1, dtype=torch.long, device=idx.device)
                zero_val = torch.zeros(1, dtype=torch.float, device=idx.device)

                active_offsets = torch.cat([zero_off, past_offsets])
                peak_vals = torch.cat([zero_val, topk_vals])

                targets = pos_grid - active_offsets.unsqueeze(0)
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                K_discrete = self.k_proj(state)
                V_discrete = self.v_proj(state)

                K_cand = K_discrete[:, targets_clamped, :]
                V_cand = V_discrete[:, targets_clamped, :]

                q = self.ln_q(E_q_all + state)
                k = self.ln_k(K_cand)
                v = V_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_peaks)
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.unsqueeze(0).float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                next_state = state + self.scale * context
                next_state = next_state + self.scale * self.mlp(self.ln_mlp(next_state))

                if return_dynamics:
                    diff = torch.norm(next_state - prev_state, dim=-1)
                    denom = torch.norm(next_state, dim=-1) + 1e-6
                    v_t = (diff / denom).mean().item()
                    diagnostics["state_velocities"].append(v_t)

                    cos_s = F.cosine_similarity(next_state.flatten(0, 1), prev_state.flatten(0, 1), dim=-1).mean().item()
                    diagnostics["state_cosine_alignments"].append(cos_s)

                    mean_decay = decay.mean().item()
                    diagnostics["wave_decays"].append(mean_decay)
                    diagnostics["wave_reaches"].append(1.0 / max(mean_decay, 1e-5))

                    if prev_wave_1d is not None:
                        w_delta = torch.norm(wave_1d - prev_wave_1d).item()
                        w_cos = F.cosine_similarity(wave_1d.unsqueeze(0), prev_wave_1d.unsqueeze(0)).item()
                    else:
                        w_delta = 0.0
                        w_cos = 1.0
                    diagnostics["wave_energy_deltas"].append(w_delta)
                    diagnostics["wave_cosines"].append(w_cos)

                    offsets_expanded = active_offsets.view(1, 1, self.k_peaks).float()
                    mean_dist = (weights * offsets_expanded).sum(dim=-1).mean().item()
                    diagnostics["mean_attended_distances"].append(mean_dist)

                    ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
                    diagnostics["hop_entropies"].append(ent)

                    diagnostics["waveforms"].append(wave_1d.detach().cpu().tolist())
                    diagnostics["active_offsets"].append(active_offsets.detach().cpu().tolist())

                    prev_state = next_state
                    prev_wave_1d = wave_1d.detach()

                state = next_state
                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            logits = self.head(self.ln_f(state))
            if return_dynamics:
                return logits, diagnostics
            return logits

    # Train Model
    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed + 1000)

    model = InvertedQueryEscrowLM().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def get_lr(step):
        if step < 100:
            return 1e-3 * (step + 1) / 100
        progress = (step - 100) / max(1, total_steps - 100)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch("train", train_gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_batches):
                    vx, vy = get_batch("val", val_gen)
                    v_logits = model(vx)
                    val_loss += F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1)).item()
            val_loss /= val_batches
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - t0
            print(f"  Step {step:4d}/{total_steps} | Train: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Time: {elapsed:.1f}s", flush=True)

    total_time = time.time() - t0

    # Extract High-Resolution Dynamics over 30 Validation Batches
    print("\nExtracting hop-by-hop dynamical trajectory across validation set...")
    model.eval()
    val_gen_eval = torch.Generator().manual_seed(seed + 9999)

    accumulated = {
        "velocities": [0.0] * n_hops,
        "state_cosines": [0.0] * n_hops,
        "decays": [0.0] * n_hops,
        "reaches": [0.0] * n_hops,
        "wave_deltas": [0.0] * n_hops,
        "wave_cosines": [0.0] * n_hops,
        "mean_distances": [0.0] * n_hops,
        "entropies": [0.0] * n_hops,
    }
    last_waveforms = None
    last_active_offsets = None

    with torch.no_grad():
        for _ in range(val_batches):
            vx, vy = get_batch("val", val_gen_eval)
            _, d = model(vx, return_dynamics=True)
            for h in range(n_hops):
                accumulated["velocities"][h] += d["state_velocities"][h]
                accumulated["state_cosines"][h] += d["state_cosine_alignments"][h]
                accumulated["decays"][h] += d["wave_decays"][h]
                accumulated["reaches"][h] += d["wave_reaches"][h]
                accumulated["wave_deltas"][h] += d["wave_energy_deltas"][h]
                accumulated["wave_cosines"][h] += d["wave_cosines"][h]
                accumulated["mean_distances"][h] += d["mean_attended_distances"][h]
                accumulated["entropies"][h] += d["hop_entropies"][h]
            last_waveforms = d["waveforms"]
            last_active_offsets = d["active_offsets"]

    for k in accumulated:
        accumulated[k] = [v / val_batches for v in accumulated[k]]

    # Pearson correlation r(v, R)
    v_arr = accumulated["velocities"]
    r_arr = accumulated["reaches"]
    mean_v = sum(v_arr) / len(v_arr)
    mean_r = sum(r_arr) / len(r_arr)
    cov = sum((vi - mean_v) * (ri - mean_r) for vi, ri in zip(v_arr, r_arr))
    var_v = sum((vi - mean_v) ** 2 for vi in v_arr)
    var_r = sum((ri - mean_r) ** 2 for ri in r_arr)
    r_corr = cov / (math.sqrt(var_v * var_r) + 1e-12)

    print("=" * 105)
    print(f"  S3-023: T={n_hops} ATTRACTOR DYNAMICS DUAL CONTRACTION SCORECARD")
    print("=" * 105)
    print(f"{'Hop':<5} | {'Velocity v(t)':<15} | {'State Cosine':<14} | {'Wave Reach R':<14} | {'Attn Entropy H':<16} | {'Mean Dist d':<12} | {'Active Offsets'}")
    print("-" * 105)
    for h in range(n_hops):
        offs = last_active_offsets[h][:5]
        print(f"{h+1:<5} | {accumulated['velocities'][h]:<15.4f} | {accumulated['state_cosines'][h]:<14.4f} | {accumulated['reaches'][h]:<14.2f} | {accumulated['entropies'][h]:<16.4f} | {accumulated['mean_distances'][h]:<12.2f} | {offs}")
    print("-" * 105)
    print(f"Final Val Loss: {val_loss:.4f} | Final Val PPL: {val_ppl:.2f} | Total Time: {total_time:.1f}s")
    print(f"Pearson Correlation r(v, R): {r_corr:+.4f}")
    print("=" * 105)

    return {
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "n_hops": n_hops,
        "hops": list(range(1, n_hops + 1)),
        "velocities": accumulated["velocities"],
        "state_cosines": accumulated["state_cosines"],
        "decays": accumulated["decays"],
        "reaches": accumulated["reaches"],
        "wave_deltas": accumulated["wave_deltas"],
        "wave_cosines": accumulated["wave_cosines"],
        "mean_distances": accumulated["mean_distances"],
        "entropies": accumulated["entropies"],
        "correlation_v_r": r_corr,
        "waveforms": last_waveforms,
        "active_offsets": last_active_offsets,
        "total_time": total_time,
    }


@app.local_entrypoint()
def main():
    import matplotlib.pyplot as plt
    import numpy as np

    print("Launching S3-023 Deep Iterative Thought Scaling (T=12 Hops) on Modal A10G...")
    result = run_t12_experiment.remote()

    os.makedirs("season3/results", exist_ok=True)
    out_path = "season3/results/s3_023_tinyshakespeare_t12_scaling.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] Raw dynamical trajectory saved to {out_path}")

    # Generate Publication-Quality 4-Panel Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

    hops = result["hops"]
    n_hops = len(hops)
    v = result["velocities"]
    r = result["reaches"]
    mean_dists = result["mean_distances"]
    entropies = result["entropies"]
    waveforms = result["waveforms"]

    # Panel 1: Token State Velocity Contraction
    ax1 = axes[0, 0]
    ax1.set_facecolor('#181b24')
    ax1.plot(hops, v, 'o-', color='#38bdf8', linewidth=2.5, markersize=8, label=r'State Velocity $v(t) = \|\Delta s\| / \|s\|$')
    ax1.fill_between(hops, v, alpha=0.2, color='#38bdf8')
    ax1.set_title(f'1. Token State Velocity (Hops 1 to {n_hops}): Attractor Convergence', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax1.set_xlabel('Thinking Hop ($t$)', fontsize=10, color='#94a3b8')
    ax1.set_ylabel('Relative State Velocity $v(t)$', fontsize=10, color='#38bdf8')
    ax1.grid(True, linestyle='--', alpha=0.25)
    ax1.set_xticks(hops)
    decel_pct = (1.0 - v[-1] / v[0]) * 100.0
    summary_text = f"Initial: {v[0]:.3f}\nHop 8: {v[7]:.3f}\nFinal (Hop 12): {v[-1]:.3f}\nTotal Deceleration: -{decel_pct:.1f}%"
    ax1.text(0.05, 0.15, summary_text, transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='#1e293b', edgecolor='#475569', alpha=0.9))

    # Panel 2: Spatial Reach & Attended Distance
    ax2 = axes[0, 1]
    ax2.set_facecolor('#181b24')
    ax2.plot(hops, r, 's-', color='#f59e0b', linewidth=2.5, markersize=8, label=r'Spatial Reach $R(t) = 1/\lambda$')
    ax2.plot(hops, mean_dists, '^-', color='#10b981', linewidth=2.0, markersize=7, label=r'Attended Distance $\bar{d}(t)$')
    ax2.fill_between(hops, r, alpha=0.15, color='#f59e0b')
    ax2.set_title(f'2. Spatial Wave Bus Across {n_hops} Hops (R ~ {np.mean(r):.1f} tokens)', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax2.set_xlabel('Thinking Hop ($t$)', fontsize=10, color='#94a3b8')
    ax2.set_ylabel('Tokens in Past', fontsize=10, color='#f59e0b')
    ax2.legend(loc='upper right', framealpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.25)
    ax2.set_xticks(hops)

    # Panel 3: Attention Entropy & State Cosine Alignment
    ax3 = axes[1, 0]
    ax3.set_facecolor('#181b24')
    ax3_twin = ax3.twinx()
    p1 = ax3.plot(hops, entropies, 'd-', color='#ec4899', linewidth=2.2, markersize=7, label=r'Attention Entropy $H(t)$')
    p2 = ax3_twin.plot(hops, result["state_cosines"], 'x--', color='#a855f7', linewidth=2.0, markersize=7, label=r'State Cosine $\cos(s^{(t)}, s^{(t-1)})$')
    ax3.set_title('3. Compilation Phase Dynamics: Entropy Focus & Emission', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax3.set_xlabel('Thinking Hop ($t$)', fontsize=10, color='#94a3b8')
    ax3.set_ylabel('Attention Softmax Entropy $H(t)$', fontsize=10, color='#ec4899')
    ax3_twin.set_ylabel('State Cosine Alignment', fontsize=10, color='#a855f7')
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center left', framealpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.25)
    ax3.set_xticks(hops)

    # Panel 4: Wave Carrier Overlays
    ax4 = axes[1, 1]
    ax4.set_facecolor('#181b24')
    d_axis = np.arange(1, 256)
    cmap = plt.cm.magma(np.linspace(0.3, 0.95, n_hops))
    for h in range(n_hops):
        alpha = 0.4 + 0.6 * (h / (n_hops - 1))
        ax4.plot(d_axis, waveforms[h], label=f'Hop {h+1}', color=cmap[h], alpha=alpha, linewidth=1.8 if h in [0, n_hops-1] else 1.0)
    ax4.set_title(f'4. Continuous Harmonic Carrier Waves Across {n_hops} Hops', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax4.set_xlabel('Past Distance $d$ (Tokens)', fontsize=10, color='#94a3b8')
    ax4.set_ylabel('Wave Carrier Amplitude $W(d)$', fontsize=10, color='#94a3b8')
    ax4.legend(ncol=3, loc='upper right', framealpha=0.6, fontsize=7)
    ax4.grid(True, linestyle='--', alpha=0.25)

    plt.suptitle(f'Study S3-023: Deep Thought Scaling at T={n_hops} Hops (Val PPL: {result["final_val_ppl"]:.2f} | Loss: {result["final_val_loss"]:.4f})', 
                 fontsize=15, fontweight='heavy', color='#ffffff', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    plot_path = "season3/results/s3_023_tinyshakespeare_t12_scaling.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    artifact_path = "C:/Users/beca/.gemini/antigravity/brain/dbfc076f-5303-41c8-803d-5cf988d61a5e/s3_023_tinyshakespeare_t12_scaling.png"
    plt.savefig(artifact_path, bbox_inches='tight', dpi=200)
    print(f"[PLOT] Saved attractor dynamics figure to {plot_path} and {artifact_path}")


