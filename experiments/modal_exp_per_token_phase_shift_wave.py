import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-per-token-phase-shift-wave", image=image)

@app.function(gpu="A10G", timeout=900)
def run_per_token_phase_shift():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 52: PER-TOKEN DYNAMIC PHASE-SHIFT FOURIER WAVE ROUTING")
    print("  Global Harmonic Carrier Frequencies (ω) + Token-Specific Phase Shifts (φ_i) for Customized Wave Peaks")
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
    num_waves = 8 # N=8 harmonic waves
    K_peaks = 8   # Strictly evaluate ONLY K=8 tokens per query!
    T_hops = 4    # T=4 recurrent thinking hops
    max_d = 128   # Search space for wave peaks (0 to 127 distance)

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # 2. Per-Token Phase-Shifted Sparse Wave Peak Module
    # -------------------------------------------------------------------------
    class PerTokenPhaseShiftWaveAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=8, K_peaks=8, max_d=128):
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

            # Global Sequence Summarizer: outputs global carrier parameters (A, omega, phi_global, decay)
            self.summary_pool = nn.Linear(d_model, 1, bias=False)
            self.global_synth = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, n_heads * 4) # 4 global scalar controllers per head
            )

            # Per-Token Phase Projector: each token i generates its personal phase offset delta_phi_i
            self.token_phase_proj = nn.Linear(d_model, n_heads * 4, bias=False)

            # Non-trainable harmonic basis matrices
            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            base_freqs = (10.0 ** log_freqs).view(1, 1, 1, 1, num_waves)
            phase_ladders = torch.linspace(0.0, 1.0, num_waves).view(1, 1, 1, 1, num_waves)

            self.register_buffer("base_freqs", base_freqs)
            self.register_buffer("phase_ladders", phase_ladders)

        def forward(self, x, s):
            B, L, D = x.shape

            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]

            # 1. Global carrier parameters from sequence summary
            pool_weights = F.softmax(self.summary_pool(s), dim=1)
            seq_summary = (s * pool_weights).sum(dim=1)
            global_params = self.global_synth(seq_summary).view(B, self.n_heads, 4)

            amp = torch.tanh(global_params[..., 0:1]).view(B, self.n_heads, 1, 1, 1) # [B, H, 1, 1, 1]
            omega = (F.softplus(global_params[..., 1:2]).view(B, self.n_heads, 1, 1, 1) * self.base_freqs) # [B, H, 1, 1, N]
            phi_global = global_params[..., 2:3].view(B, self.n_heads, 1, 1, 1)
            decay = (F.softplus(global_params[..., 3:4]).view(B, self.n_heads, 1, 1, 1) * 0.05) # [B, H, 1, 1, 1]

            # 2. Per-Token Dynamic Phase Shift: Delta phi_i computed from token state s_i
            # delta_phi: [B, L, H, 4] -> permute to [B, H, L, 1, 1]
            delta_phi = self.token_phase_proj(s).view(B, L, self.n_heads, 4).permute(0, 2, 1, 3)[..., 2:3].unsqueeze(-1) # [B, H, L, 1, 1]
            phi_token = ((phi_global + delta_phi + self.phase_ladders) * math.pi) # [B, H, L, 1, N]

            # 3. Compute Per-Token 1D Wave Curve: W_i(d) for each token i across d in [1 ... max_d-1]
            # d_grid: [1, 1, 1, max_d-1, 1]
            d_grid = torch.arange(1, self.max_d, device=x.device).float().view(1, 1, 1, self.max_d - 1, 1)
            wave_comps = amp * torch.cos(omega * d_grid + phi_token) * torch.exp(-decay * d_grid) # [B, H, L, max_d-1, N]
            wave_per_token = wave_comps.sum(dim=-1) # [B, H, L, max_d-1]

            # 4. Extract Per-Token Top-(K-1) Peak Distances: each token gets its personal optimal offsets!
            topk_vals, past_peak_offsets = torch.topk(wave_per_token, k=self.K_peaks - 1, dim=-1) # [B, H, L, K-1]
            past_peak_offsets = past_peak_offsets + 1 # shift back to 1-indexed

            # Prepend offset d=0 (self-attention) for every token
            zero_offset = torch.zeros((B, self.n_heads, L, 1), dtype=torch.long, device=x.device)
            zero_val = torch.zeros((B, self.n_heads, L, 1), dtype=torch.float, device=x.device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1) # [B, H, L, K]
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)             # [B, H, L, K]

            # 5. Build Sparse Target Gathering Indices
            # q_pos: [1, 1, L, 1]
            q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets # [B, H, L, K]
            valid_mask = target_indices >= 0
            target_indices_clamped = torch.clamp(target_indices, min=0)

            # 6. Gather K Peak Key/Value Tokens (Strict O(L * K))
            idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, self.K_peaks, self.d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded) # [B, H, L, K, d_k]
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, self.n_heads, L, self.K_peaks, self.d_k), dim=2, index=idx_expanded) # [B, H, L, K, d_k]

            # 7. Dot-Product Attention on ONLY the K Peak Tokens
            Q_exp = Q.unsqueeze(3) # [B, H, L, 1, d_k]
            scores = (Q_exp * K_gathered).sum(dim=-1) / math.sqrt(self.d_k) # [B, H, L, K]
            scores = scores + peak_vals # add token wave energy prior

            # Safe Masked Softmax
            scores = scores.masked_fill(~valid_mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = attn_weights * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 8. Gather output context: sum_k attn[k] * V_gathered[k]
            out = (attn_weights.unsqueeze(-1) * V_gathered).sum(dim=3) # [B, H, L, d_k]
            out = out.transpose(1, 2).contiguous().view(B, L, D)

            return self.out_proj(out), peak_offsets

    # -------------------------------------------------------------------------
    # 3. Model & Training Loop
    # -------------------------------------------------------------------------
    class PerTokenPhaseShiftLM(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, num_waves=8, K_peaks=8, T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = PerTokenPhaseShiftWaveAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
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

    model = PerTokenPhaseShiftLM(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_mlp=512, num_waves=num_waves, K_peaks=K_peaks, T=T_hops).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: Per-Token Phase-Shift Fourier Wave SubQ (K={K_peaks}, T={T_hops}) | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

    print(f"\nTraining Per-Token Phase-Shift Wave SubQ on TinyShakespeare (2,000 steps)...")
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

        if (step + 1) % 400 == 0 or (step + 1) == 2000:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for v_step in range(20):
                    vx, vy = get_batch('val', step_seed=90000 + v_step)
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(v_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                ppl = math.exp(avg_val_loss)
                elapsed = time.time() - t0
                print(f"  Step {step+1:>4}/2000 | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {ppl:>6.2f} | Time: {elapsed:.1f}s")

    # Diagnostic: Inspect how different tokens within the SAME sequence pick different wave peaks!
    print("\n[Inspecting Per-Token Phase-Shifted Wave Peaks Across Different Tokens in Sequence]...")
    model.eval()
    sample_text = "KING RICHARD:\nWhat is the matter, my lord? Why are you so sad?"
    sample_tokens = torch.tensor([char_to_ix.get(c, 0) for c in sample_text], dtype=torch.long, device=device).unsqueeze(0)
    padded_tokens = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    padded_tokens[0, :len(sample_tokens[0])] = sample_tokens[0]

    with torch.no_grad():
        _, all_peaks = model(padded_tokens, return_peaks=True)

    # Inspect final hop peaks for 3 different tokens: Token 5 ('R'), Token 14 ('\n'), Token 30 ('m')
    final_peaks = all_peaks[-1] # [B, H, L, K]
    inspect_tokens = [5, 14, 30]
    print(f"\nFinal Hop (t=4) Wave Peak Offsets for Head 1 Across Different Token Positions:")
    for pos_i in inspect_tokens:
        char_i = repr(sample_text[pos_i]) if pos_i < len(sample_text) else 'PAD'
        offsets_i = sorted(list(final_peaks[0, 0, pos_i]))
        print(f"  Token Position i={pos_i:>2} (Char: {char_i:<6}): Peak Offsets = {offsets_i}")

    # Sample Generation
    context = torch.tensor([char_to_ix[c] for c in "KING RICHARD:\n"], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(120):
        logits = model(context[:, -seq_len:])
        next_tok = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, dim=-1), num_samples=1)
        context = torch.cat([context, next_tok], dim=1)
    generated_sample = "".join([ix_to_char[i] for i in context[0].cpu().numpy()])
    print(f"\nSample Generation (Per-Token Phase Shifted):\n{'-'*60}\n{generated_sample}\n{'-'*60}")

    return {"name": "Per-Token Phase-Shift Fourier Wave SubQ", "val_loss": avg_val_loss, "val_ppl": ppl}

@app.local_entrypoint()
def main():
    run_per_token_phase_shift.remote()
