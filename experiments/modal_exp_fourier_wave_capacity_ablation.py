import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-fourier-wave-capacity-ablation", image=image)

@app.function(gpu="A10G", timeout=1200)
def run_wave_capacity_ablation():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 51: EXACT PARAMETER-MATCHED FOURIER WAVE CAPACITY ABLATION (N = 1, 4, 8, 12, 16, 20 Waves)")
    print("  All Models Have 100.00% IDENTICAL Parameter Counts (Zero Parameter Drift Across Wave Counts)")
    print("  Budget Held Strictly Constant: K = 8 Tokens per Query | T = 4 Thinking Hops | L = 256")
    print("=" * 120)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

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
    K_peaks = 8  # Strictly evaluate ONLY K=8 tokens per query!
    T_hops = 4   # T=4 recurrent thinking hops
    max_d = 128  # Search space for wave peaks (0 to 127 distance)

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # 2. Exact Parameter-Matched Sparse Wave Peak Module (N waves)
    # -------------------------------------------------------------------------
    class FixedParamSparseFourierPeakAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=4, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.num_waves = num_waves
            self.K_peaks = K_peaks
            self.max_d = max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            # Sequence summary & fixed-parameter controller (always exactly 16 outputs = 4 scalar latents per head)
            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.wave_synth = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, n_heads * 4)
            )

            # Non-trainable harmonic basis matrices (zero parameter drift!)
            if num_waves == 1:
                base_freqs = torch.tensor([0.25]).view(1, 1, 1, 1)
                phase_ladders = torch.tensor([0.0]).view(1, 1, 1, 1)
            else:
                log_freqs = torch.linspace(0.0, -2.0, num_waves) # 10^0 = 1.0 down to 10^-2 = 0.01
                base_freqs = (10.0 ** log_freqs).view(1, 1, 1, num_waves)
                phase_ladders = torch.linspace(0.0, 1.0, num_waves).view(1, 1, 1, num_waves)

            self.register_buffer("base_freqs", base_freqs)
            self.register_buffer("phase_ladders", phase_ladders)

        def forward(self, x, s):
            B, L, D = x.shape

            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            pool_weights = F.softmax(self.summary_pool(s), dim=1)
            seq_summary = (s * pool_weights).sum(dim=1)
            
            # Base controllers: [B, H, 4] (Constant across any N!)
            base_params = self.wave_synth(seq_summary).view(B, self.n_heads, 4)

            amp = torch.tanh(base_params[..., 0:1]).unsqueeze(-1) # [B, H, 1, 1]
            omega = (F.softplus(base_params[..., 1:2]).unsqueeze(-1) * self.base_freqs) # [B, H, 1, N]
            phi = ((base_params[..., 2:3].unsqueeze(-1) + self.phase_ladders) * math.pi) # [B, H, 1, N]
            decay = (F.softplus(base_params[..., 3:4]).unsqueeze(-1) * 0.05) # [B, H, 1, 1]

            # 1D Distance search grid for past offsets: d = 1, 2, ..., max_d-1
            d_grid = torch.arange(1, self.max_d, device=x.device).float().view(1, 1, self.max_d - 1, 1)
            wave_comps = amp * torch.cos(omega * d_grid + phi) * torch.exp(-decay * d_grid)
            wave_1d = wave_comps.sum(dim=-1) # [B, H, max_d - 1]

            # Extract Top-(K-1) Peak Distances
            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1 # shift back to 1-indexed

            # Prepend offset d=0 (self-attention)
            zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=x.device)
            zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=x.device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            # Build Sparse Target Gathering Indices
            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            offsets = peak_offsets.unsqueeze(2)
            target_indices = q_pos - offsets
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            # Gather K Peak Key/Value Tokens (Strict O(L * K))
            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded)

            # Dot-Product Attention on K Peak Tokens
            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k)
            scores = scores + peak_vals.unsqueeze(2)

            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = attn_weights * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, L, D)

            return self.out_proj(out), peak_offsets

    class FixedParamSparseFourierPeakLM(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, num_waves=4, K_peaks=8, T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = FixedParamSparseFourierPeakAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
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
                attn_out, _ = self.attn(x, s)
                s = s + (1.0 / math.sqrt(self.T)) * attn_out
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            return self.head(s)

    # -------------------------------------------------------------------------
    # 3. Benchmark Suite: Sweep Over N = [1, 4, 8, 12, 16, 20] Waves
    # -------------------------------------------------------------------------
    wave_configs = [1, 4, 8, 12, 16, 20]
    results = []

    print("\n" + "=" * 100)
    print(f"  STARTING EXACT PARAMETER-MATCHED HARMONIC CAPACITY BENCHMARK ACROSS {len(wave_configs)} CONFIGURATIONS")
    print("=" * 100)

    for num_w in wave_configs:
        torch.manual_seed(42)
        model = FixedParamSparseFourierPeakLM(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_mlp=512, num_waves=num_w, K_peaks=K_peaks, T=T_hops).to(device)
        param_count = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

        print(f"\n[Running N={num_w:>2} Waves] (Params: {param_count:,})...")
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

        print(f"  --> N={num_w:>2} Waves Result: Val Loss = {avg_val_loss:.4f} | Val PPL = {ppl:.2f} | Time = {elapsed:.1f}s")
        results.append({
            "num_waves": num_w,
            "params": param_count,
            "train_loss": loss.item(),
            "val_loss": avg_val_loss,
            "val_ppl": ppl,
            "elapsed": elapsed
        })

    # Summary Report Table
    print("\n" + "=" * 120)
    print("  FINAL EXACT PARAMETER-MATCHED HARMONIC CAPACITY BENCHMARK SUMMARY")
    print("=" * 120)
    print(f"{'Wave Count (N)':<16} | {'Parameters':<12} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Speed (s)':<10} | {'Relative Gain vs N=1'}")
    print("-" * 120)
    base_ppl = results[0]["val_ppl"]
    for r in results:
        gain = ((base_ppl - r["val_ppl"]) / base_ppl) * 100
        gain_str = f"+{gain:.1f}% PPL drop" if gain > 0 else "Baseline (N=1)"
        print(f"N = {r['num_waves']:<12} | {r['params']:<12,} | {r['train_loss']:<12.4f} | {r['val_loss']:<10.4f} | {r['val_ppl']:<10.2f} | {r['elapsed']:<10.1f} | {gain_str}")

    return results

@app.local_entrypoint()
def main():
    run_wave_capacity_ablation.remote()
