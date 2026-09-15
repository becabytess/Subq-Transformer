import json
import os
import pathlib
import time
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "matplotlib"
    )
)

app = modal.App("exp-s2-037-shared-wave-standard-heads", image=image)

@app.function(image=image, gpu="A10G", timeout=2400)
def train_one(k_peaks: int, name: str):
    import math
    import time
    import urllib.request
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100, flush=True)
    print(f"  STARTING PARALLEL RUN: {name} (K_peaks = {k_peaks}) on {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 100, flush=True)

    # 1. Download TinyShakespeare Dataset (Identical to S2-035 & S2-036)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    d_mlp = 512
    num_waves = 12
    max_d = 128
    T_hops = 8      # Matched strictly to S2-035 & S2-036
    n_steps = 3000  # Strictly 3,000 steps

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # Shared Wave Standard MHA
    class SharedWaveStandardMHA(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.num_waves, self.K_peaks, self.max_d = num_waves, K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            # 1 SHARED Wave Latent across all heads: shape (1, num_waves * 4)
            self.init_wave_latent = nn.Parameter(torch.randn(1, num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 64),
                nn.GELU(),
                nn.Linear(64, num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def get_offsets_and_bias(self, wave_latent, B, device):
            curr_params = wave_latent.view(1, self.num_waves, 4)
            amp = torch.tanh(curr_params[..., 0]).view(1, 1, 1, self.num_waves)
            omega = (F.softplus(curr_params[..., 1]).view(1, 1, 1, self.num_waves) * self.base_freqs)
            phi = (curr_params[..., 2] * math.pi).view(1, 1, 1, self.num_waves)
            decay = (F.softplus(curr_params[..., 3]) * 0.05).view(1, 1, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, 1, -1)

            topk_vals, past_peak_offsets = torch.topk(wave_1d, k=self.K_peaks - 1, dim=-1)
            past_peak_offsets = past_peak_offsets + 1

            zero_offset = torch.zeros((B, 1, 1), dtype=torch.long, device=device)
            zero_val = torch.zeros((B, 1, 1), dtype=torch.float, device=device)
            peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
            peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

            # Broadcast identical offsets to all heads
            peak_offsets = peak_offsets.expand(B, self.n_heads, self.K_peaks)
            peak_vals = peak_vals.expand(B, self.n_heads, self.K_peaks)

            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return peak_offsets, peak_vals, next_wave_latent

        def forward(self, x, s, wave_latent=None):
            B, L, D = x.shape
            device = x.device
            if wave_latent is None:
                wave_latent = self.init_wave_latent
            peak_offsets, peak_vals, next_wave_latent = self.get_offsets_and_bias(wave_latent, B, device)

            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
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
            return self.out_proj(out), next_wave_latent

    class SubQLM(nn.Module):
        def __init__(self, K_peaks):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = SharedWaveStandardMHA(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.T = T_hops

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s = x

            w = getattr(self.attn, "init_wave_latent", None)
            inv_sqrt_T = 1.0 / math.sqrt(self.T)
            for step in range(self.T):
                z = self.ln_attn(s)
                attn_out, w = self.attn(x, z, wave_latent=w)
                s = s + inv_sqrt_T * attn_out

            s = s + self.mlp(self.ln_mlp(s))
            s = self.ln_f(s)
            logits = self.head(s)
            return logits

    torch.cuda.reset_peak_memory_stats()
    model = SubQLM(K_peaks=k_peaks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Total Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

    t0 = time.time()
    eval_history = []

    for step in range(1, n_steps + 1):
        model.train()
        bx, by = get_batch('train', step_seed=10000 + step)
        optimizer.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        if step % 500 == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for v_step in range(20):
                    vx, vy = get_batch('val', step_seed=90000 + v_step)
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(v_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(avg_val_loss)
                elapsed = time.time() - t0
                print(f"[{name}] Step {step:>4}/{n_steps} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:>6.2f} | Time: {elapsed:.1f}s", flush=True)
                eval_history.append({
                    "step": step,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(avg_val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                    "elapsed_s": round(elapsed, 1)
                })

    total_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Patterns
    patterns = {}
    with torch.no_grad():
        w = model.attn.init_wave_latent
        for hop in range(1, T_hops + 1):
            curr_offsets, curr_biases, w = model.attn.get_offsets_and_bias(w, 1, device)
            if hop in [1, 2, 4, 8]:
                hop_key = f"hop_{hop}"
                patterns[hop_key] = {
                    "integer_offsets": curr_offsets[0, 0].cpu().tolist(),
                    "prior_biases": [round(float(b), 3) for b in curr_biases[0, 0].cpu().tolist()]
                }

    return {
        "name": name,
        "k_peaks": k_peaks,
        "final_val_loss": eval_history[-1]["val_loss"],
        "final_val_ppl": eval_history[-1]["val_ppl"],
        "eval_history": eval_history,
        "total_time_s": round(total_time, 1),
        "throughput_tok_s": round((n_steps * batch_size * seq_len) / total_time, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "patterns": patterns
    }

@app.local_entrypoint()
def main():
    print("Launching Study S2-037 on Modal in PARALLEL across separate A10G GPUs...", flush=True)
    configs = [
        (32, "shared_wave_k32"),
        (8, "shared_wave_k8"),
    ]

    t0 = time.time()
    # Spawn both jobs simultaneously on Modal
    calls = [train_one.spawn(k, name) for k, name in configs]

    results = {}
    for (k, name), call in zip(configs, calls):
        print(f"Waiting for {name} (K={k})...", flush=True)
        res = call.get()
        results[name] = res
        print(f"Finished {name}: Val Loss = {res['final_val_loss']:.4f}, Val PPL = {res['final_val_ppl']:.2f}", flush=True)

    total_parallel_time = time.time() - t0

    print("\n" + "=" * 115)
    print("  STUDY S2-037 FINAL HEAD STANDARDIZATION SCORECARD (PARALLEL RUN)")
    print("=" * 115)
    print(f"{'Condition':<42} | {'Tokens Touched':<15} | {'Val Loss':<10} | {'Val PPL':<10} | {'Time (s)':<10}")
    print("-" * 115)
    print(f"{'S2-036 Baseline (Per-Head Waves K=8)':<42} | {'<= 32 tokens':<15} | {'1.6828':<10} | {'5.38':<10} | {'87.8s':<10}")
    res_k32 = results["shared_wave_k32"]
    res_k8 = results["shared_wave_k8"]
    print(f"{'Version 1: Standard MHA Shared Wave (K=32)':<42} | {'32 tokens':<15} | {res_k32['final_val_loss']:<10.4f} | {res_k32['final_val_ppl']:<10.2f} | {res_k32['total_time_s']:<10.1f}")
    print(f"{'Version 2: Standard MHA Shared Wave (K=8)':<42} | {'8 tokens (3.1%)':<15} | {res_k8['final_val_loss']:<10.4f} | {res_k8['final_val_ppl']:<10.2f} | {res_k8['total_time_s']:<10.1f}")
    print(f"\nTotal Wall-Clock Parallel Runtime: {total_parallel_time:.1f}s")
    print("=" * 115)

    os.makedirs("season2/results", exist_ok=True)
    out_json = "season2/results/s2_037_shared_wave_standard_heads.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved S2-037 results to {out_json}]")
