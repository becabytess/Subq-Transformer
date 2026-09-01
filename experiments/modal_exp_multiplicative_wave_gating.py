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

app = modal.App("exp-multiplicative-wave-gating", image=image)

@app.function(gpu="A10G", timeout=2400)
def run_multiplicative_gating_ablation():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 63: MULTIPLICATIVE WAVE GATING VS ADDITIVE LOGIT BIAS")
    print("  Testing 1) Additive Bias, 2) Multiplicative Logit Scaling, 3) Multiplicative Value Output Gating")
    print("  Budget Strictly Held Constant: K = 8 Tokens | L = 256 | D = 128 | 2,000 Steps | Evolving Q,K,V")
    print("=" * 120)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Download TinyShakespeare Dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
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

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # Attention Layer with Switchable Wave Modulation Modes
    # -------------------------------------------------------------------------
    class FlexibleWaveGatedAttention(nn.Module):
        def __init__(self, mode="additive", d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.mode = mode # 'additive', 'mult_logit', 'mult_value'
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
            raw_scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k) # [B, H, L, K]

            # -----------------------------------------------------------------
            # 3 Distinct Integration Modes
            # -----------------------------------------------------------------
            if self.mode == "additive":
                # Mode 1: Additive Logit Bias (Scores + W(d))
                scores = raw_scores + peak_vals.unsqueeze(2)
                scores = scores.masked_fill(~valid_mask, -1e4)
                attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
                out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)

            elif self.mode == "mult_logit":
                # Mode 2: Multiplicative Logit Scaling (Scores * (1 + 2 * sigmoid(W(d))))
                # Sigmoid scales gain smoothly between 0.0 and 2.0
                wave_scale = 2.0 * torch.sigmoid(peak_vals.unsqueeze(2))
                scores = raw_scores * wave_scale
                scores = scores.masked_fill(~valid_mask, -1e4)
                attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
                out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)

            elif self.mode == "mult_value":
                # Mode 3: Multiplicative Spatial Value Gate (Pure Q.K attention, Wave gates the V output)
                scores = raw_scores.masked_fill(~valid_mask, -1e4)
                attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Wave acts as an explicit spatial gating mask on gathered V values
                spatial_gate = 2.0 * torch.sigmoid(peak_vals.unsqueeze(2)).unsqueeze(-1) # [B, H, L, K, 1]
                V_gated = V_gathered * spatial_gate
                out = (attn_weights.unsqueeze(-1) * V_gated).sum(dim=3)

            out = out.transpose(1, 2).contiguous().view(B, L, D)
            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return self.out_proj(out), next_wave_latent

    class HarmonicSubQLM(nn.Module):
        def __init__(self, vocab_size, mode="additive", T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = FlexibleWaveGatedAttention(mode=mode, d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
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

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x

            w = self.attn.init_wave_latent
            for step in range(self.T):
                attn_out, w = self.attn(x, s, wave_latent=w)
                s = s + (1.0 / math.sqrt(self.T)) * attn_out
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            return self.head(s)

    # -------------------------------------------------------------------------
    # Benchmark Suite
    # -------------------------------------------------------------------------
    experiments = [
        ("1. Additive Logit Bias (Scores + W(d), T=4)", "additive", 4),
        ("2. Multiplicative Logit Scaling (Scores * σ(W), T=4)", "mult_logit", 4),
        ("3. Multiplicative Value Output Gate (Attn * σ(W) * V, T=4)", "mult_value", 4),
        ("4. Additive Logit Bias (Scores + W(d), T=8)", "additive", 8),
        ("5. Multiplicative Logit Scaling (Scores * σ(W), T=8)", "mult_logit", 8),
        ("6. Multiplicative Value Output Gate (Attn * σ(W) * V, T=8)", "mult_value", 8),
    ]

    results = []
    print("\n" + "=" * 100)
    print("  RUNNING BENCHMARK: ADDITIVE VS MULTIPLICATIVE LOGIT VS MULTIPLICATIVE VALUE GATING")
    print("=" * 100)

    for name, mode, T_val in experiments:
        torch.manual_seed(42)
        model = HarmonicSubQLM(vocab_size=vocab_size, mode=mode, T=T_val).to(device)
        param_count = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

        print(f"\n[Training {name}] (Parameters: {param_count:,})...")
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

        print(f"  --> Result: Val Loss = {avg_val_loss:.4f} | Val PPL = {ppl:.2f} | Time = {elapsed:.1f}s")
        results.append({
            "name": name,
            "params": param_count,
            "train_loss": loss.item(),
            "val_loss": avg_val_loss,
            "val_ppl": ppl,
            "elapsed": elapsed
        })

    # Summary Table
    print("\n" + "=" * 120)
    print("  STUDY 63 FINAL SUMMARY: MULTIPLICATIVE GATING VS ADDITIVE LOGIT BIAS")
    print("=" * 120)
    print(f"{'Architecture':<65} | {'Parameters':<12} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Speed (s)':<10}")
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<65} | {r['params']:<12,} | {r['train_loss']:<12.4f} | {r['val_loss']:<10.4f} | {r['val_ppl']:<10.2f} | {r['elapsed']:<10.1f}")

    return results

@app.local_entrypoint()
def main():
    run_multiplicative_gating_ablation.remote()
