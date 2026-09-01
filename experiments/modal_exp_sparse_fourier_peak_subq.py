import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-sparse-fourier-peak-subq", image=image)

@app.function(gpu="A10G", timeout=900)
def run_sparse_fourier_peak_subq():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 50: PURE SPARSE HARMONIC WAVE PEAK ROUTER (Strict O(L * K) Linear Compute)")
    print("  The 4-Wave Interference Curve Selects the Top-K Peaks -> Attention Evaluates ONLY Those K Tokens!")
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
    num_waves = 4
    K_peaks = 8  # Strictly evaluate ONLY K=8 tokens per query!
    T_hops = 4   # T=4 recurrent thinking hops
    max_d = 128  # Search space for wave peaks (0 to 127 distance)

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # 2. Pure Sparse Wave-Peak Attention Module (O(L * K) Only!)
    # -------------------------------------------------------------------------
    class SparseFourierPeakAttention(nn.Module):
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

            # Sequence summary & wave parameter generator
            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.wave_synth = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_heads * num_waves * 4)
            )

            # Base multi-scale frequencies
            base_freqs = torch.tensor([1.0, 0.25, 0.0625, 0.015625]).view(1, 1, 1, num_waves)
            self.register_buffer("base_freqs", base_freqs)

            # 1D Distance search grid: d = 0, 1, 2, ..., max_d-1
            d_grid = torch.arange(max_d).float().view(1, 1, max_d, 1) # [1, 1, max_d, 1]
            self.register_buffer("d_grid", d_grid)

        def forward(self, x, s):
            # x: original token embeddings (for K, V)
            # s: current evolving recurrent state (for Q & Wave Synthesizer)
            B, L, D = x.shape

            # 1. Project Q, K, V
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]

            # 2. Extract sequence summary and synthesize 4-wave parameters
            pool_weights = F.softmax(self.summary_pool(s), dim=1) # [B, L, 1]
            seq_summary = (s * pool_weights).sum(dim=1)           # [B, D]
            wave_params = self.wave_synth(seq_summary).view(B, self.n_heads, self.num_waves, 4) # [B, H, N, 4]

            amp = torch.tanh(wave_params[..., 0]).view(B, self.n_heads, 1, self.num_waves)
            omega = (F.softplus(wave_params[..., 1]).view(B, self.n_heads, 1, self.num_waves) * self.base_freqs)
            phi = (wave_params[..., 2] * math.pi).view(B, self.n_heads, 1, self.num_waves)
            decay = (F.softplus(wave_params[..., 3]) * 0.05).view(B, self.n_heads, 1, self.num_waves)

            # 1D Distance search grid for past offsets: d = 1, 2, ..., max_d-1
            d_grid = torch.arange(1, self.max_d, device=x.device).float().view(1, 1, self.max_d - 1, 1) # [1, 1, max_d - 1, 1]

            # 3. Compute 1D Continuous Wave Curve W(d) for d in [1 ... max_d-1]
            wave_comps = amp * torch.cos(omega * d_grid + phi) * torch.exp(-decay * d_grid)
            wave_1d = wave_comps.sum(dim=-1) # [B, H, max_d - 1]

            # 4. Extract Top-(K-1) Peak Distances from the wave
            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1) # [B, H, K-1]
            past_peak_offsets = past_peak_offsets + 1 # shift back to 1-indexed

            # Prepend offset d=0 (self-attention) so every token always has at least 1 valid candidate
            zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=x.device)
            zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=x.device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1) # [B, H, K]
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)             # [B, H, K]

            # 5. Build Sparse Target Gathering Indices: target = clamp(i - d, min=0)
            # q_pos: [1, 1, L, 1]
            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            # offsets: [B, H, 1, K]
            offsets = peak_offsets.unsqueeze(2)
            
            # target_indices: [B, H, L, K]
            target_indices = q_pos - offsets
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            # 6. GATHER Key and Value for ONLY the K peak tokens! (Strict O(L * K))
            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded) # [B, H, L, K, d_k]
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded) # [B, H, L, K, d_k]

            # 7. Compute Dot-Product Attention on ONLY the K peak tokens!
            Q_exp = Q.unsqueeze(3) # [B, H, L, 1, d_k]
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k) # [B, H, L, K]
            scores = scores + peak_vals.unsqueeze(2) # add wave energy prior

            # Safe Masked Softmax (Zero NaNs guaranteed!)
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = attn_weights * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 8. Gather output context: sum_k attn[k] * V_gathered[k]
            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3) # [B, H, L, d_k]
            out = out.transpose(1, 2).contiguous().view(B, L, D)

            return self.out_proj(out), peak_offsets

    # -------------------------------------------------------------------------
    # 3. Recurrent Sparse Wave Peak Model (T=4 iterations)
    # -------------------------------------------------------------------------
    class SparseFourierPeakLM(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, num_waves=4, K_peaks=8, T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = SparseFourierPeakAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
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

        def forward(self, idx, return_peaks=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            s = x
            all_peaks = []

            for step in range(self.T):
                attn_out, peaks = self.attn(x, s)
                if return_peaks:
                    all_peaks.append(peaks.detach().cpu().numpy())

                s = s + (1.0 / math.sqrt(self.T)) * attn_out
                s = self.ln_attn(s)
                s = s + (1.0 / math.sqrt(self.T)) * self.mlp(self.ln_mlp(s))

            s = self.ln_f(s)
            logits = self.head(s)
            if return_peaks:
                return logits, all_peaks
            return logits

    # -------------------------------------------------------------------------
    # 4. Train Model
    # -------------------------------------------------------------------------
    model = SparseFourierPeakLM(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_mlp=512, num_waves=num_waves, K_peaks=K_peaks, T=T_hops).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: Sparse Fourier Wave-Peak SubQ (K={K_peaks} Peaks, T={T_hops} Hops) | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

    print(f"\nTraining Pure Sparse Wave Peak SubQ (K={K_peaks} tokens per query) on TinyShakespeare (2,000 steps)...")
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

        if (step + 1) % 400 == 0 or (step + 1) == 2000:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    vx, vy = get_batch('val')
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(v_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                ppl = math.exp(avg_val_loss)
                elapsed = time.time() - t0
                print(f"  Step {step+1:>4}/2000 | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {ppl:>6.2f} | Time: {elapsed:.1f}s")

    # Diagnostic: Inspect the exact K=8 peak distances chosen at each hop
    print("\n[Inspecting the Exact K=8 Wave Peaks Chosen Across Hops t=1..4]...")
    model.eval()
    sample_text = "KING RICHARD:\nWhat is the matter, my lord? Why are you so sad?"
    sample_tokens = torch.tensor([char_to_ix.get(c, 0) for c in sample_text], dtype=torch.long, device=device).unsqueeze(0)
    padded_tokens = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    padded_tokens[0, :len(sample_tokens[0])] = sample_tokens[0]

    with torch.no_grad():
        _, all_peaks = model(padded_tokens, return_peaks=True)

    for step_idx, p in enumerate(all_peaks):
        print(f"\nHop t={step_idx+1} Peak Offsets Selected by the 4 Attention Heads (K={K_peaks} tokens):")
        for h in range(n_heads):
            offsets_h = sorted(list(p[0, h]))
            print(f"  Head {h+1}: Offsets = {offsets_h}")

    # Sample Generation
    context = torch.tensor([char_to_ix[c] for c in "KING RICHARD:\n"], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(120):
        logits = model(context[:, -seq_len:])
        next_tok = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, dim=-1), num_samples=1)
        context = torch.cat([context, next_tok], dim=1)
    generated_sample = "".join([ix_to_char[i] for i in context[0].cpu().numpy()])
    print(f"\nSample Generation (Sparse K={K_peaks} Peaks):\n{'-'*60}\n{generated_sample}\n{'-'*60}")

    return {"name": f"Sparse Wave-Peak SubQ (K={K_peaks}, T={T_hops})", "val_loss": avg_val_loss, "val_ppl": ppl}

@app.local_entrypoint()
def main():
    run_sparse_fourier_peak_subq.remote()
