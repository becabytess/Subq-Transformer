import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-rnn-offset-generator", image=image)

@app.function(gpu="A10G", timeout=1200)
def run_rnn_vs_fourier_ablation():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 54: RNN OFFSET GENERATORS VS CONTINUOUS FOURIER WAVES")
    print("  Can an Iterative Recurrent Neural Network (RNN/GRU) Learn Higher-Resolution Routing Than Fourier Waves?")
    print("  Budget Strictly Held Constant: K = 8 Tokens per Query | T = 4 Thinking Hops | L = 256")
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
    K_peaks = 8
    T_hops = 4
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
    # Baseline 1: Fourier Harmonic Wave Synthesizer (N=12)
    # -------------------------------------------------------------------------
    class FourierWavePeakAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.num_waves, self.K_peaks, self.max_d = num_waves, K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.wave_synth = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, n_heads * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("phase_ladders", torch.linspace(0.0, 1.0, num_waves).view(1, 1, 1, num_waves))

        def forward(self, x, s):
            B, L, D = x.shape
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            pool_weights = F.softmax(self.summary_pool(s), dim=1)
            seq_summary = (s * pool_weights).sum(dim=1)
            base_params = self.wave_synth(seq_summary).view(B, self.n_heads, 4)

            amp = torch.tanh(base_params[..., 0:1]).unsqueeze(-1)
            omega = (F.softplus(base_params[..., 1:2]).unsqueeze(-1) * self.base_freqs)
            phi = ((base_params[..., 2:3].unsqueeze(-1) + self.phase_ladders) * math.pi)
            decay = (F.softplus(base_params[..., 3:4]).unsqueeze(-1) * 0.05)

            d_grid = torch.arange(1, self.max_d, device=x.device).float().view(1, 1, self.max_d - 1, 1)
            wave_comps = amp * torch.cos(omega * d_grid + phi) * torch.exp(-decay * d_grid)
            wave_1d = wave_comps.sum(dim=-1)

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
    # Model 1: Autoregressive Delta Jump RNN (Direct Step-by-Step Offset Emitter)
    # -------------------------------------------------------------------------
    class AutoregressiveDeltaJumpRNNAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.K_peaks, self.max_d = K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.init_proj = nn.Linear(d_model, n_heads * 32)
            # GRU Cell to emit step-by-step delta jump increments
            self.gru_cell = nn.GRUCell(input_size=1, hidden_size=32)
            self.delta_head = nn.Linear(32, 1)

        def forward(self, x, s):
            B, L, D = x.shape
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            pool_weights = F.softmax(self.summary_pool(s), dim=1)
            seq_summary = (s * pool_weights).sum(dim=1)

            # Reshape initial GRU hidden states: [B * H, 32]
            h_gru = self.init_proj(seq_summary).view(B * self.n_heads, 32)
            current_d = torch.zeros((B * self.n_heads, 1), device=x.device)
            offset_list = [torch.zeros((B * self.n_heads, 1), dtype=torch.long, device=x.device)]

            # Autoregressively step the RNN K-1 times to generate delta jumps
            for step_k in range(self.K_peaks - 1):
                h_gru = self.gru_cell(current_d / float(self.max_d), h_gru)
                # Output positive jump increment (minimum jump = +1)
                delta_d = F.softplus(self.delta_head(h_gru)) + 1.0
                current_d = current_d + delta_d
                clamped_d = torch.clamp(current_d.round().long(), max=self.max_d - 1)
                offset_list.append(clamped_d)

            # peak_offsets: [B, H, K]
            peak_offsets = torch.cat(offset_list, dim=-1).view(B, self.n_heads, self.K_peaks)

            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets.unsqueeze(2)
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
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
    # Model 2: Continuous 1D Spatial Curve RNN Synthesizer (Generates W(d) via RNN)
    # -------------------------------------------------------------------------
    class SpatialCurveRNNAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.K_peaks, self.max_d = K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.init_proj = nn.Linear(d_model, n_heads * 32)
            # GRU that unrolls along the distance dimension d = 1..127 to synthesize W(d)
            self.spatial_gru = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
            self.energy_head = nn.Linear(32, 1)

            # Pre-allocated 1D distance step input [1, max_d-1, 1]
            d_steps = (torch.arange(1, max_d).float() / float(max_d)).view(1, max_d - 1, 1)
            self.register_buffer("d_steps", d_steps)

        def forward(self, x, s):
            B, L, D = x.shape
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            pool_weights = F.softmax(self.summary_pool(s), dim=1)
            seq_summary = (s * pool_weights).sum(dim=1)

            # h0 for spatial GRU: [1, B * H, 32]
            h0 = self.init_proj(seq_summary).view(B * self.n_heads, 32).unsqueeze(0)
            d_input = self.d_steps.expand(B * self.n_heads, self.max_d - 1, 1)
            
            gru_out, _ = self.spatial_gru(d_input, h0) # [B*H, max_d-1, 32]
            wave_1d = self.energy_head(gru_out).squeeze(-1).view(B, self.n_heads, self.max_d - 1)

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
    # General LM Wrapper
    # -------------------------------------------------------------------------
    class SubQLM(nn.Module):
        def __init__(self, vocab_size, attn_module, d_model=128, d_mlp=512, T=4):
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
    # 3. Train & Benchmark All 3 Paradigms Under Exact Matched Conditions
    # -------------------------------------------------------------------------
    models_to_test = [
        ("1. Fourier Harmonic Waves (N=12)", FourierWavePeakAttention(d_model=d_model, n_heads=n_heads, num_waves=12, K_peaks=K_peaks, max_d=max_d)),
        ("2. Autoregressive Delta Jump RNN", AutoregressiveDeltaJumpRNNAttention(d_model=d_model, n_heads=n_heads, K_peaks=K_peaks, max_d=max_d)),
        ("3. Spatial Curve 1D RNN Synthesizer", SpatialCurveRNNAttention(d_model=d_model, n_heads=n_heads, K_peaks=K_peaks, max_d=max_d))
    ]

    results = []
    print("\n" + "=" * 100)
    print("  RUNNING PARADIGM COMPARISON: FOURIER WAVES VS RNN OFFSET GENERATORS")
    print("=" * 100)

    for name, attn_layer in models_to_test:
        torch.manual_seed(42)
        model = SubQLM(vocab_size=vocab_size, attn_module=attn_layer, d_model=d_model, d_mlp=512, T=T_hops).to(device)
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
    print("  STUDY 54 FINAL SUMMARY: FOURIER HARMONIC WAVES VS RNN OFFSET GENERATORS")
    print("=" * 120)
    print(f"{'Offset Generation Mechanism':<42} | {'Parameters':<12} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Speed (s)':<10}")
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<42} | {r['params']:<12,} | {r['train_loss']:<12.4f} | {r['val_loss']:<10.4f} | {r['val_ppl']:<10.2f} | {r['elapsed']:<10.1f}")

    return results

@app.local_entrypoint()
def main():
    run_rnn_vs_fourier_ablation.remote()
