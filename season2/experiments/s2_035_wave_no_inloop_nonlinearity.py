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

app = modal.App("exp-s2-035-wave-no-inloop", image=image)

@app.function(gpu="A10G", timeout=1800)
def run_wave_inloop_investigation():
    import math
    import time
    import urllib.request
    import io
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 110)
    print("  STUDY S2-035: HARMONIC WAVE DYNAMICS UNDER PURE LINEAR RECURRENCE (NO IN-LOOP NON-LINEARITY)")
    print("  Testing if Macro-to-Micro Frequency Shift & Spatial Decay Funnel Persist Without Per-Hop GRU / MLP")
    print(f"  Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")
    print("=" * 110)

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
    n_steps = 2000

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # Shared Dynamical Wave Attention Module
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

        def get_wave_diagnostics(self, wave_latent):
            curr_params = wave_latent.view(self.n_heads, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]) # [n_heads, num_waves]
            omega_mult = F.softplus(curr_params[..., 1])
            effective_freqs = (omega_mult.view(1, self.n_heads, 1, self.num_waves) * self.base_freqs).view(self.n_heads, self.num_waves)
            periods = 2 * math.pi / (effective_freqs + 1e-6)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(self.n_heads, self.num_waves)

            amp_view = amp.view(1, self.n_heads, 1, self.num_waves)
            omega_view = effective_freqs.view(1, self.n_heads, 1, self.num_waves)
            phi_view = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
            decay_view = decay.view(1, self.n_heads, 1, self.num_waves)

            wave_comps = amp_view * torch.cos(omega_view * self.d_grid + phi_view) * torch.exp(-decay_view * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).squeeze(0) # [n_heads, max_d - 1]

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1
            return wave_1d, past_peak_offsets, amp, periods, decay

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
            return self.out_proj(out), next_wave_latent

    # -------------------------------------------------------------------------
    # Model 1: Season 1 Baseline (Per-Hop In-Loop MLP + LN)
    # -------------------------------------------------------------------------
    class SubQ_PerHopNonLinearity(nn.Module):
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

        def forward(self, idx, return_wave_trace=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x

            w = self.attn.init_wave_latent
            wave_trace = []

            for step in range(self.T):
                if return_wave_trace:
                    wave_trace.append(w.clone())
                attn_out, w = self.attn(x, s, wave_latent=w)
                s = s + (1.0 / math.sqrt(self.T)) * attn_out
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            logits = self.head(s)
            if return_wave_trace:
                return logits, wave_trace
            return logits

    # -------------------------------------------------------------------------
    # Model 2: Season 2 Canonical (NO In-Loop Non-Linearity - Pure Linear Accumulation)
    # -------------------------------------------------------------------------
    class SubQ_NoInLoopNonLinearity(nn.Module):
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

        def forward(self, idx, return_wave_trace=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x

            w = self.attn.init_wave_latent
            wave_trace = []

            # Pure linear attention accumulation: NO per-hop MLP, NO per-hop LN!
            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            for step in range(self.T):
                if return_wave_trace:
                    wave_trace.append(w.clone())
                z = self.ln_attn(s)
                attn_out, w = self.attn(x, z, wave_latent=w)
                s = s + inv_sqrt_T * attn_out  # Pure linear residual accumulation

            # Single non-linearity at the end of the block
            s = s + self.mlp(self.ln_mlp(s))
            s = self.ln_f(s)
            logits = self.head(s)
            if return_wave_trace:
                return logits, wave_trace
            return logits

    # -------------------------------------------------------------------------
    # Training & Evaluation Helper
    # -------------------------------------------------------------------------
    def train_model(model_cls, name):
        torch.manual_seed(42)
        model = model_cls(vocab_size, T=T_hops).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n---> Training: {name} (Params: {param_count:,}, T={T_hops} hops, Steps: {n_steps})...")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

        t0 = time.time()
        for step in range(n_steps):
            model.train()
            bx, by = get_batch('train', step_seed=10000 + step)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 500 == 0:
                print(f"      Step {step+1:4d}/{n_steps} | Train Loss: {loss.item():.4f}")

        elapsed = time.time() - t0

        # Evaluate on validation split
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vstep in range(30):
                vx, vy = get_batch('val', step_seed=50000 + vstep)
                v_logits = model(vx)
                v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                val_losses.append(v_loss.item())

        avg_val_loss = float(np.mean(val_losses))
        ppl = float(math.exp(avg_val_loss))
        print(f"  [Result] {name}: Val Loss: {avg_val_loss:.4f} | Val PPL: {ppl:.2f} | Time: {elapsed:.1f}s")

        # Extract Wave Dynamics across hops
        with torch.no_grad():
            sample_x, _ = get_batch('val', step_seed=42)
            _, wave_trace = model(sample_x[:1], return_wave_trace=True)

        hop_diagnostics = []
        for h_idx, w_lat in enumerate(wave_trace):
            wave_1d, past_peaks, amp, periods, decay = model.attn.get_wave_diagnostics(w_lat)
            hop_data = {"hop": h_idx + 1, "heads": []}
            for head in range(n_heads):
                w_np = wave_1d[head].cpu().numpy()
                amp_np = amp[head].abs().cpu().numpy()
                periods_np = periods[head].cpu().numpy()
                decay_np = decay[head].cpu().numpy()
                peaks = [0] + sorted(past_peaks[head].cpu().tolist())

                hi = float(amp_np[periods_np < 8].mean()) if (periods_np < 8).any() else 0.0
                mid = float(amp_np[(periods_np >= 8) & (periods_np <= 32)].mean()) if ((periods_np >= 8) & (periods_np <= 32)).any() else 0.0
                low = float(amp_np[periods_np > 32].mean()) if (periods_np > 32).any() else 0.0
                rms = float(math.sqrt((w_np**2).mean()))

                hop_data["heads"].append({
                    "head": head,
                    "rms_amp": rms,
                    "mean_decay": float(decay_np.mean()),
                    "offsets": peaks,
                    "freq_bands": {"high": hi, "mid": mid, "low": low}
                })
            hop_diagnostics.append(hop_data)

        return {
            "name": name,
            "params": param_count,
            "val_loss": avg_val_loss,
            "val_ppl": ppl,
            "elapsed": elapsed,
            "diagnostics": hop_diagnostics
        }

    # Run Both Models Under Strict Identical Conditions
    res_per_hop = train_model(SubQ_PerHopNonLinearity, "1. Season 1 Baseline (Per-Hop In-Loop MLP)")
    res_no_inloop = train_model(SubQ_NoInLoopNonLinearity, "2. Season 2 Canonical (NO In-Loop MLP - Pure Linear Acc)")

    # Print Comparative Diagnostic Table
    print("\n" + "=" * 110)
    print("  FINAL COMPARISON: WAVE DYNAMICS WITH vs WITHOUT IN-LOOP NON-LINEARITY")
    print("=" * 110)
    print(f"{'Model':<48} | {'Val Loss':<10} | {'Val PPL':<10} | {'Training Time':<12}")
    print("-" * 110)
    print(f"{res_per_hop['name']:<48} | {res_per_hop['val_loss']:<10.4f} | {res_per_hop['val_ppl']:<10.2f} | {res_per_hop['elapsed']:<10.1f}s")
    print(f"{res_no_inloop['name']:<48} | {res_no_inloop['val_loss']:<10.4f} | {res_no_inloop['val_ppl']:<10.2f} | {res_no_inloop['elapsed']:<10.1f}s")
    print("=" * 110)

    # Detailed Hop-by-Hop Wave Frequency and Decay Analysis
    print("\nHOP-BY-HOP FREQUENCY & DECAY COMPARISON (Head 1 & Head 0):")
    for hop_idx in [0, 1, 3, 7]:
        h_s1 = res_per_hop["diagnostics"][hop_idx]
        h_s2 = res_no_inloop["diagnostics"][hop_idx]
        t = hop_idx + 1
        print(f"\n--- Hop t={t} ---")
        print(f"  [Season 1 Per-Hop MLP]  Head 1 RMS: {h_s1['heads'][1]['rms_amp']:.3f} | Decay: {h_s1['heads'][1]['mean_decay']:.4f} | Bands (H/M/L): ({h_s1['heads'][1]['freq_bands']['high']:.2f}/{h_s1['heads'][1]['freq_bands']['mid']:.2f}/{h_s1['heads'][1]['freq_bands']['low']:.2f}) | Offsets: {h_s1['heads'][1]['offsets']}")
        print(f"  [Season 2 Pure Linear]  Head 1 RMS: {h_s2['heads'][1]['rms_amp']:.3f} | Decay: {h_s2['heads'][1]['mean_decay']:.4f} | Bands (H/M/L): ({h_s2['heads'][1]['freq_bands']['high']:.2f}/{h_s2['heads'][1]['freq_bands']['mid']:.2f}/{h_s2['heads'][1]['freq_bands']['low']:.2f}) | Offsets: {h_s2['heads'][1]['offsets']}")

    return {
        "res_per_hop": res_per_hop,
        "res_no_inloop": res_no_inloop
    }

@app.local_entrypoint()
def main():
    import json
    results = run_wave_inloop_investigation.remote()
    os.makedirs("season2/results", exist_ok=True)
    out_file = "season2/results/s2_035_wave_no_inloop_nonlinearity.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n--> Successfully saved results to {out_file}")
