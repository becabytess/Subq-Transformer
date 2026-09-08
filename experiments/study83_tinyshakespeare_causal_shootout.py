import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("study83-tinyshakespeare-causal-shootout", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_causal_shootout():
    import math, time, json, urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 83: DEFINITIVE CAUSAL SHOOTOUT ON TINYSHAKESPEARE")
    print("  100% Strictly Causal (Zero Future Peeking Anywhere: Causal GRU, Causal SubQ, Causal Dense)")
    print("  Evaluating Causal GRU Prefix Scan vs Pure Harmonic SubQ vs Causal Dense Transformer (250k & 350k Parity)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # 1. Dataset Setup
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"Dataset: TinyShakespeare ({len(data):,} chars, Vocab={vocab_size}, Train={len(train_data):,}, Val={len(val_data):,})")

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads # 32
    num_waves = 12
    K_peaks = 8
    max_d = 128
    total_steps = 2000
    eval_interval = 250
    val_batches = 30

    def get_batch(split, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix])
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # Architecture 1: Standard 1L Causal Dense Transformer
    # -------------------------------------------------------------------------
    class CausalDenseTransformer(nn.Module):
        def __init__(self, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Causal Dense Attention
            x_norm = self.ln_1(x)
            q = self.q_proj(x_norm).view(B, L, n_heads, d_k).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L, n_heads, d_k).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L, n_heads, d_k).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
            scores = scores.masked_fill(self.causal_mask[:, :, :L, :L] == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, L, d_model)
            x = x + self.out_proj(attn_out)

            x = x + self.mlp(self.ln_2(x))
            return self.head(self.ln_f(x))

    # -------------------------------------------------------------------------
    # Architecture 2: Pure Harmonic Recurrent SubQ LM (Strictly Causal)
    # -------------------------------------------------------------------------
    class PureHarmonicSubQLM(nn.Module):
        def __init__(self, T=4, d_mlp=512):
            super().__init__()
            self.T = T
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            self.ln_1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )

            # Carrier wave parameters
            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x
            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            for t in range(self.T):
                s_norm = self.ln_1(s)
                # Evolving Q, K, V
                q = self.q_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)
                k = self.k_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)
                v = self.v_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)

                # Continuous spatial carrier wave
                amp = torch.tanh(curr_params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(curr_params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (curr_params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                # Top-K positive backward peak offsets
                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=idx.device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=idx.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1) # [B, n_heads, 8]
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                # Strictly Causal Gathering: target_indices = q_pos - peak_offsets (<= q_pos)
                target_indices = q_pos - peak_offsets.unsqueeze(2) # [B, n_heads, L, K]
                valid_mask = target_indices >= 0
                target_indices_clamped = torch.clamp(target_indices, min=0)

                idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, n_heads, L, K_peaks, d_k)
                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, d_k), dim=2, index=idx_expanded)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, d_k), dim=2, index=idx_expanded)

                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(d_k) + peak_vals.unsqueeze(2)
                scores = scores.masked_fill(~valid_mask, -1e4)
                attn = F.softmax(scores, dim=-1) * valid_mask.float()
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.out_proj(attn_out)

                s = s + inv_sqrt_T * attn_out
                s = s + inv_sqrt_T * self.mlp(self.ln_2(s))

                if t < self.T - 1:
                    flat_p = curr_params.view(n_heads, num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(n_heads, num_waves, 4)

            return self.head(self.ln_f(s))

    # -------------------------------------------------------------------------
    # Architecture 3: Causal GRU + SubQ Wave Attention Hybrid LM
    # -------------------------------------------------------------------------
    class CausalGRUSubQLM(nn.Module):
        def __init__(self, T=4, d_mlp=512):
            super().__init__()
            self.T = T
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Strictly Causal Unidirectional GRU Scan (0 -> L, Zero Future Access)
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            self.ln_gru = nn.LayerNorm(d_model)

            self.ln_1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )

            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Phase 1: Fast O(L) Strictly Causal Forward Scan
            gru_out, _ = self.gru(x)
            s = self.ln_gru(x + gru_out)

            # Phase 2: Causal SubQ Wave Relaxation
            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            for t in range(self.T):
                s_norm = self.ln_1(s)
                q = self.q_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)
                k = self.k_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)
                v = self.v_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)

                amp = torch.tanh(curr_params[..., 0]).view(1, n_heads, 1, num_waves)
                omega = (F.softplus(curr_params[..., 1]).view(1, n_heads, 1, num_waves) * self.base_freqs)
                phi = (curr_params[..., 2] * math.pi).view(1, n_heads, 1, num_waves)
                decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, n_heads, 1, num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=K_peaks - 1, dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, n_heads, 1), dtype=torch.long, device=idx.device)
                zero_val = torch.zeros((B, n_heads, 1), dtype=torch.float, device=idx.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                target_indices = q_pos - peak_offsets.unsqueeze(2)
                valid_mask = target_indices >= 0
                target_indices_clamped = torch.clamp(target_indices, min=0)

                idx_expanded = target_indices_clamped.unsqueeze(-1).expand(B, n_heads, L, K_peaks, d_k)
                K_g = torch.gather(k.unsqueeze(3).expand(B, n_heads, L, K_peaks, d_k), dim=2, index=idx_expanded)
                V_g = torch.gather(v.unsqueeze(3).expand(B, n_heads, L, K_peaks, d_k), dim=2, index=idx_expanded)

                scores = (q.unsqueeze(3) * K_g).sum(dim=-1) / math.sqrt(d_k) + peak_vals.unsqueeze(2)
                scores = scores.masked_fill(~valid_mask, -1e4)
                attn = F.softmax(scores, dim=-1) * valid_mask.float()
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

                attn_out = (attn.unsqueeze(-1) * V_g).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                attn_out = self.out_proj(attn_out)

                s = s + inv_sqrt_T * attn_out
                s = s + inv_sqrt_T * self.mlp(self.ln_2(s))

                if t < self.T - 1:
                    flat_p = curr_params.view(n_heads, num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(n_heads, num_waves, 4)

            return self.head(self.ln_f(s))

    # -------------------------------------------------------------------------
    # Training & Evaluation Engine (100% Parameter & Minibatch Controlled)
    # -------------------------------------------------------------------------
    def train_and_eval(model, model_name):
        model.to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n" + "-" * 115)
        print(f"  TRAINING: {model_name} ({params:,} parameters)")
        print("-" * 115)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')

        t0 = time.time()
        for step in range(1, total_steps + 1):
            model.train()
            # Identical minibatches via fixed deterministic seed per step
            x, y = get_batch('train', seed=step * 1000 + 42)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % eval_interval == 0 or step == total_steps:
                elapsed = time.time() - t0
                print(f"    Step {step:>4d}/{total_steps} ({elapsed:.1f}s) | Train Loss: {loss.item():.4f}")

        train_time = time.time() - t0

        # Deterministic Validation Evaluation (30 minibatches)
        model.eval()
        total_val_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for v_step in range(val_batches):
                x_val, y_val = get_batch('val', seed=v_step * 5000 + 1337)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x_val)
                    loss = criterion(logits.view(-1, vocab_size), y_val.view(-1))
                total_val_loss += loss.item() * y_val.numel()
                total_tokens += y_val.numel()

        mean_val_loss = total_val_loss / total_tokens
        val_ppl = math.exp(mean_val_loss)
        print(f"  --> {model_name}: Val Loss: {mean_val_loss:.4f} | Val PPL: {val_ppl:.2f} | Time: {train_time:.1f}s")
        return {
            "name": model_name,
            "params": params,
            "val_loss": round(mean_val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "time": round(train_time, 1)
        }

    # -------------------------------------------------------------------------
    # The 7 Causal Benchmark Models
    # -------------------------------------------------------------------------
    models_to_test = [
        # --- Baseline Standard ~250k Scale ---
        (CausalDenseTransformer(d_mlp=512), "1. Causal Dense 1L Transformer (Standard, d_mlp=512)"),
        (PureHarmonicSubQLM(T=4, d_mlp=512), "2. Pure Harmonic SubQ (T=4, Standard, d_mlp=512)"),
        (PureHarmonicSubQLM(T=8, d_mlp=512), "3. Pure Harmonic SubQ (T=8, Standard, d_mlp=512)"),

        # --- Hybrid Causal GRU + SubQ (~350k Scale) ---
        (CausalGRUSubQLM(T=4, d_mlp=512), "4. Causal GRU + SubQ (T=4, Hybrid, d_mlp=512)"),
        (CausalGRUSubQLM(T=8, d_mlp=512), "5. Causal GRU + SubQ (T=8, Hybrid, d_mlp=512)"),

        # --- Parameter-Matched Baselines (~350k Scale) ---
        (PureHarmonicSubQLM(T=8, d_mlp=898), "6. Scaled Pure Harmonic SubQ (T=8, Matched ~350k, d_mlp=898)"),
        (CausalDenseTransformer(d_mlp=912), "7. Scaled Causal Dense 1L Transformer (Matched ~350k, d_mlp=912)")
    ]

    results = []
    for model_inst, name in models_to_test:
        res = train_and_eval(model_inst, name)
        results.append(res)

    print("\n" + "=" * 125)
    print("  STUDY 83 FINAL SHOOTOUT RESULTS (TINYSHAKESPEARE, 100% CAUSAL)")
    print("=" * 125)
    print(f"{'Model Architecture':<55} | {'Params':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Train Time':<10}")
    print("-" * 125)
    for r in results:
        print(f"{r['name']:<55} | {r['params']:<10,d} | {r['val_loss']:<10.4f} | {r['val_ppl']:<10.2f} | {r['time']:<10.1f}s")
    print("=" * 125)

    with open("/models/study83_tinyshakespeare_causal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results

@app.local_entrypoint()
def main():
    run_causal_shootout.remote()
