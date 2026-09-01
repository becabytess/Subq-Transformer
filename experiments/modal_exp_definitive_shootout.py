import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-definitive-apples-to-apples-shootout", image=image)

@app.function(gpu="A10G", timeout=1800)
def run_definitive_shootout():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  DEFINITIVE APPLES-TO-APPLES BENCHMARK SHOOTOUT (STUDY 58)")
    print("  ALL Models Trained Under 100% Identical Conditions, Dimensions (D=128), Seeds, and Optimizer Schedules")
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
    # 1. Standard Dense Transformer (1 Layer & 4 Layer)
    # -------------------------------------------------------------------------
    class StandardDenseAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        def forward(self, x):
            B, L, D = x.shape
            Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out)

    class DenseTransformerBlock(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.attn = StandardDenseAttention(d_model, n_heads)
            self.ln_attn = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_mlp = nn.LayerNorm(d_model)

        def forward(self, x):
            x = x + self.attn(self.ln_attn(x))
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class StandardDenseTransformerLM(nn.Module):
        def __init__(self, vocab_size, num_layers=1):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList([DenseTransformerBlock(d_model, n_heads, d_mlp) for _ in range(num_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    # -------------------------------------------------------------------------
    # 2. Fixed Offset Sparse SubQ Model (Dyadic / Fibonacci)
    # -------------------------------------------------------------------------
    class FixedOffsetAttention(nn.Module):
        def __init__(self, offsets_list, d_model=128, n_heads=4):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.offsets_list = offsets_list
            self.K_peaks = len(offsets_list)
            self.register_buffer("offsets", torch.tensor(offsets_list, dtype=torch.long).view(1, 1, 1, self.K_peaks))

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, s):
            B, L, D = x.shape
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            target_indices = q_pos - self.offsets # [1, 1, L, K]
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            idx_expanded = target_indices_clamped.expand(B, self.n_heads, L, self.K_peaks).unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out)

    # -------------------------------------------------------------------------
    # 3. Dynamic Fourier Wave Peak SubQ Model (Full Evolving Q, K, V)
    # -------------------------------------------------------------------------
    class DynamicFourierWaveAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.num_waves, self.K_peaks, self.max_d = num_waves, K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def forward(self, x, s):
            B, L, D = x.shape
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            amp = torch.tanh(self.raw_wave_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
            omega = (F.softplus(self.raw_wave_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
            phi = (self.raw_wave_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
            decay = (F.softplus(self.raw_wave_params[..., 3]) * 0.05).view(1, self.n_heads, 1, self.num_waves)

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
            return self.out_proj(out)

    # -------------------------------------------------------------------------
    # Recurrent Wrapper for SubQ
    # -------------------------------------------------------------------------
    class RecurrentSubQLM(nn.Module):
        def __init__(self, vocab_size, attn_module, T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = attn_module
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

            for step in range(self.T):
                attn_out = self.attn(x, s)
                s = s + (1.0 / math.sqrt(self.T)) * attn_out
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            return self.head(s)

    # -------------------------------------------------------------------------
    # 4. Comprehensive Shootout Suite
    # -------------------------------------------------------------------------
    benchmark_models = [
        ("1. Standard 1L Dense Transformer", StandardDenseTransformerLM(vocab_size=vocab_size, num_layers=1)),
        ("2. Standard 4L Dense Transformer", StandardDenseTransformerLM(vocab_size=vocab_size, num_layers=4)),
        ("3. Fixed Dyadic 8 Jumps (K=8, T=4)", RecurrentSubQLM(vocab_size, FixedOffsetAttention([0, 1, 2, 4, 8, 16, 32, 64], d_model, n_heads), T=4)),
        ("4. Fixed Fibonacci 8 Jumps (K=8, T=4)", RecurrentSubQLM(vocab_size, FixedOffsetAttention([0, 1, 2, 3, 5, 8, 13, 21], d_model, n_heads), T=4)),
        ("5. Fixed Fibonacci 12 Jumps (K=12, T=4)", RecurrentSubQLM(vocab_size, FixedOffsetAttention([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127], d_model, n_heads), T=4)),
        ("6. Dynamic Fourier Wave Peaks (K=8, T=4, Evolving QKV)", RecurrentSubQLM(vocab_size, DynamicFourierWaveAttention(d_model, n_heads, num_waves=12, K_peaks=8), T=4)),
        ("7. Dynamic Fourier Wave Peaks (K=8, T=8, Evolving QKV)", RecurrentSubQLM(vocab_size, DynamicFourierWaveAttention(d_model, n_heads, num_waves=12, K_peaks=8), T=8)),
        ("8. Dynamic Fourier Wave Peaks (K=8, T=12, Evolving QKV)", RecurrentSubQLM(vocab_size, DynamicFourierWaveAttention(d_model, n_heads, num_waves=12, K_peaks=8), T=12)),
    ]

    results = []
    print("\n" + "=" * 100)
    print(f"  RUNNING COMPLETE RIGOROUS SHOOTOUT ACROSS ALL {len(benchmark_models)} MODELS")
    print("=" * 100)

    for name, model in benchmark_models:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters())

        # Exact matched AdamW and Cosine schedule
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

    # Summary Report Table
    print("\n" + "=" * 120)
    print("  STUDY 58 FINAL DEFINITIVE SUMMARY: RIGOROUS APPLES-TO-APPLES BENCHMARK")
    print("=" * 120)
    print(f"{'Architecture':<56} | {'Parameters':<12} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Speed (s)':<10}")
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<56} | {r['params']:<12,} | {r['train_loss']:<12.4f} | {r['val_loss']:<10.4f} | {r['val_ppl']:<10.2f} | {r['elapsed']:<10.1f}")

    return results

@app.local_entrypoint()
def main():
    run_definitive_shootout.remote()
