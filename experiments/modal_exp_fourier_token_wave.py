import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-fourier-token-wave", image=image)

@app.function(gpu="A10G", timeout=900)
def run_fourier_token_wave():
    import math
    import time
    import urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 120)
    print("  STUDY 48A: TOKEN-LEVEL DYNAMIC FOURIER WAVE ATTENTION (Per-Token Sinusoidal Synthesis)")
    print("  Each Token Dynamically Generates Frequency (ω), Phase Shift (φ), Amplitude (A), & Decay (λ)")
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
    num_waves = 4 # N=4 harmonic waves per token

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # 2. Token-Level Fourier Wave Attention Model
    # -------------------------------------------------------------------------
    class TokenFourierWaveAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=4):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.num_waves = num_waves

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            # Per-token wave parameter synthesizer: generates (A, omega, phi, lambda) per head per wave
            # Total parameters per token: n_heads * num_waves * 4
            self.wave_synth = nn.Linear(d_model, n_heads * num_waves * 4, bias=True)
            
            # Learnable base frequency centers (log-spaced from high freq to low freq)
            base_freqs = torch.tensor([1.0, 0.25, 0.0625, 0.015625]).view(1, 1, num_waves, 1) # [1, 1, N, 1]
            self.register_buffer("base_freqs", base_freqs)

            # Distance matrix: row i (query), col j (key) -> distance d = i - j
            q_idx = torch.arange(seq_len).unsqueeze(1) # [L, 1]
            k_idx = torch.arange(seq_len).unsqueeze(0) # [1, L]
            d_matrix = torch.clamp(q_idx - k_idx, min=0).float() # [L, L] (i - j >= 0)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)) # [L, L] strictly past tokens j <= i
            self.register_buffer("d_matrix", d_matrix)
            self.register_buffer("causal_mask", causal_mask)

        def forward(self, x):
            B, L, D = x.shape

            Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, d_k]

            # 1. Content dot-product affinity
            content_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) # [B, H, L, L]

            # 2. Synthesize sinusoidal wave parameters per token
            # params: [B, L, H, N, 4]
            wave_raw = self.wave_synth(x).view(B, L, self.n_heads, self.num_waves, 4).permute(0, 2, 1, 3, 4) # [B, H, L, N, 4]
            
            amp = torch.tanh(wave_raw[..., 0:1]) # [-1, 1] amplitude
            omega = F.softplus(wave_raw[..., 1:2]) * self.base_freqs # Frequency
            phi = wave_raw[..., 2:3] * math.pi # Phase shift [-pi, pi]
            decay = F.softplus(wave_raw[..., 3:4]) * 0.05 # Decay factor

            # 3. Compute continuous harmonic interference pattern: W(i, j) for all j <= i
            # d_mat: [1, 1, L, L, 1]
            d_mat = self.d_matrix[:L, :L].view(1, 1, L, L, 1)
            
            # Superposition over N waves: sum_m A_m * cos(omega_m * d + phi_m) * exp(-decay_m * d)
            # amp, omega, phi, decay are [B, H, L, N, 1] -> unsqueeze to [B, H, L, 1, N]
            amp = amp.permute(0, 1, 2, 4, 3)     # [B, H, L, 1, N]
            omega = omega.permute(0, 1, 2, 4, 3) # [B, H, L, 1, N]
            phi = phi.permute(0, 1, 2, 4, 3)     # [B, H, L, 1, N]
            decay = decay.permute(0, 1, 2, 4, 3) # [B, H, L, 1, N]

            wave_components = amp * torch.cos(omega * d_mat + phi) * torch.exp(-decay * d_mat) # [B, H, L, L, N]
            wave_bias = wave_components.sum(dim=-1) # [B, H, L, L]

            # 4. Modulate content attention by harmonic interference
            total_scores = content_scores + wave_bias
            total_scores = total_scores.masked_fill(~self.causal_mask[:L, :L], float("-inf"))
            attn_weights = F.softmax(total_scores, dim=-1)

            # 5. Gather values
            out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out)

    class TokenFourierTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, num_waves=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = TokenFourierWaveAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            x = self.ln_f(x)
            return self.head(x)

    # -------------------------------------------------------------------------
    # 3. Train Token Fourier Transformer
    # -------------------------------------------------------------------------
    model = TokenFourierTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_mlp=512, num_waves=num_waves).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: Token-Level Fourier Wave Transformer | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

    print("\nTraining Token Fourier Wave Transformer on TinyShakespeare (2,000 steps)...")
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

    # Sample Generation
    model.eval()
    context = torch.tensor([char_to_ix[c] for c in "KING RICHARD:\n"], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(120):
        logits = model(context[:, -seq_len:])
        next_tok = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, dim=-1), num_samples=1)
        context = torch.cat([context, next_tok], dim=1)
    generated_sample = "".join([ix_to_char[i] for i in context[0].cpu().numpy()])
    print(f"\nSample Generation:\n{'-'*60}\n{generated_sample}\n{'-'*60}")

    return {"name": "Token Fourier Wave (Per-Token Synthesizer)", "val_loss": avg_val_loss, "val_ppl": ppl}

@app.local_entrypoint()
def main():
    run_fourier_token_wave.remote()
