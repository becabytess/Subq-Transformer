"""S4-001: Attention-Driven Parallel GRU Cell (The Spatiotemporal Lattice).

Tests the foundational Season 4 hypothesis:
- Attention is purely the dynamic input selector (c_i = sum_k alpha_k V_{i - Delta_k}).
- The current token state s_i is the recurrent hidden state.
- The state update is a parallel GRU cell transition: s_i^(t) = GRUCell(input=c_i, hidden=s_i^(t-1)).

Compared against:
1. canonical_subq_final_mlp: Season 2 baseline (T=4, linear attention accumulation + 1 final MLP).
2. canonical_subq_per_hop_mlp: S2-025 baseline (T=4, per-hop MLP inside the loop).
3. attention_driven_gru_cell: Season 4 architecture (T=4, attention context as input to GRUCell).
"""

import json
import math
import os
import time
import urllib.request
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("season4-s4-001-gru-cell", image=image)


@app.function(image=image, gpu="A10G", timeout=2400)
def train_model(model_type: str, n_hops: int = 8, seed: int = 42, n_steps: int = 2000):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100, flush=True)
    print(f"  STARTING RUN: {model_type} (T={n_hops} Hops, Seed={seed}) on {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 100, flush=True)

    # 1. Download TinyShakespeare Dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode("utf-8")
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
    K_peaks = 8
    max_d = 128

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------------------
    # Shared Wave Routing Engine (Strict K=8 budget broadcast to all heads)
    # -------------------------------------------------------------------------
    class SharedWaveRouter(nn.Module):
        def __init__(self, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.num_waves = num_waves
            self.K_peaks = K_peaks
            self.max_d = max_d

            self.init_wave_latent = nn.Parameter(torch.randn(1, num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 64),
                nn.GELU(),
                nn.Linear(64, num_waves * 4)
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def forward(self, wave_latent, B, device):
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
            peak_offsets = peak_offsets.expand(B, n_heads, self.K_peaks)
            peak_vals = peak_vals.expand(B, n_heads, self.K_peaks)

            next_wave_latent = wave_latent + 0.1 * self.wave_transition(wave_latent)
            return peak_offsets, peak_vals, next_wave_latent

    # -------------------------------------------------------------------------
    # Attention Sensory Antenna (Computes gathered context c_i)
    # -------------------------------------------------------------------------
    class AttentionAntenna(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.router = SharedWaveRouter(num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
            self.scale = 1.0 / math.sqrt(d_k)

        def forward(self, s, wave_latent):
            B, L, D = s.shape
            device = s.device

            peak_offsets, peak_vals, next_wave = self.router(wave_latent, B, device)

            Q = self.q_proj(s).view(B, L, n_heads, d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, n_heads, d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, n_heads, d_k).transpose(1, 2)

            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            target_indices = q_pos - peak_offsets.unsqueeze(2)
            valid_mask = target_indices >= 0
            target_clamped = torch.clamp(target_indices, min=0)

            idx_exp = target_clamped.unsqueeze(-1).expand(B, n_heads, L, K_peaks, d_k)
            K_gathered = torch.gather(K.unsqueeze(3).expand(B, n_heads, L, K_peaks, d_k), dim=2, index=idx_exp)
            V_gathered = torch.gather(V.unsqueeze(3).expand(B, n_heads, L, K_peaks, d_k), dim=2, index=idx_exp)

            Q_exp = Q.unsqueeze(3)
            scores = (Q_exp * K_gathered).sum(dim=-1) * self.scale + peak_vals.unsqueeze(2)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1) * valid_mask.float()
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            context = (weights.unsqueeze(-1) * V_gathered).sum(dim=3)
            context = context.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(context), next_wave

    # -------------------------------------------------------------------------
    # Parallel GRU Cell Transition Engine
    # -------------------------------------------------------------------------
    class ParallelGRUCell(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.w_ih = nn.Linear(d_model, 3 * d_model)
            self.w_hh = nn.Linear(d_model, 3 * d_model)

        def forward(self, x, h):
            # x: incoming context [B, L, D]
            # h: previous token state [B, L, D]
            gates_i = self.w_ih(x)
            gates_h = self.w_hh(h)
            i_r, i_z, i_n = gates_i.chunk(3, dim=-1)
            h_r, h_z, h_n = gates_h.chunk(3, dim=-1)

            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            # PyTorch convention: z is gate on old state, (1 - z) on new candidate
            h_next = (1.0 - z) * n + z * h
            return h_next, z.detach().mean().item(), r.detach().mean().item()

    # -------------------------------------------------------------------------
    # 3 Model Variants
    # -------------------------------------------------------------------------
    class Model(nn.Module):
        def __init__(self, mode: str):
            super().__init__()
            self.mode = mode
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.antenna = AttentionAntenna()
            self.ln_in = nn.LayerNorm(d_model)

            if mode == "canonical_subq_final_mlp":
                self.ln_mlp = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_mlp),
                    nn.GELU(),
                    nn.Linear(d_mlp, d_model),
                )
            elif mode == "canonical_subq_per_hop_mlp":
                self.ln_attn = nn.LayerNorm(d_model)
                self.ln_mlp = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_mlp),
                    nn.GELU(),
                    nn.Linear(d_mlp, d_model),
                )
            elif mode == "attention_driven_gru_cell":
                self.gru_cell = ParallelGRUCell(d_model)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            s = self.tok_emb(idx) + self.pos_emb(pos)

            w = self.antenna.router.init_wave_latent
            inv_sqrt_T = 1.0 / math.sqrt(n_hops)
            gate_stats = []

            for hop in range(1, n_hops + 1):
                z_norm = self.ln_in(s)
                context, w = self.antenna(z_norm, wave_latent=w)

                if self.mode == "canonical_subq_final_mlp":
                    s = s + inv_sqrt_T * context
                elif self.mode == "canonical_subq_per_hop_mlp":
                    s = s + inv_sqrt_T * context
                    s = self.ln_attn(s)
                    s = s + inv_sqrt_T * self.mlp(self.ln_mlp(s))
                elif self.mode == "attention_driven_gru_cell":
                    # Token state s is hidden state; Attention context is new input x
                    s, mean_z, mean_r = self.gru_cell(x=context, h=s)
                    gate_stats.append({"hop": hop, "mean_update_z": mean_z, "mean_reset_r": mean_r})

            if self.mode == "canonical_subq_final_mlp":
                s = s + self.mlp(self.ln_mlp(s))

            logits = self.head(self.ln_f(s))
            return logits, gate_stats

    model = Model(model_type).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_type}] Total Trainable Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

    t0 = time.time()
    eval_history = []

    for step in range(1, n_steps + 1):
        model.train()
        bx, by = get_batch("train", step_seed=10000 + step)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(bx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        if step % 500 == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_losses = []
                last_gate_stats = []
                for v_step in range(20):
                    vx, vy = get_batch("val", step_seed=90000 + v_step)
                    v_logits, g_stats = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(v_loss.item())
                    if v_step == 0 and g_stats:
                        last_gate_stats = g_stats
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(avg_val_loss)
                elapsed = time.time() - t0
                print(
                    f"[{model_type}] Step {step:>4}/{n_steps} | Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:>6.2f} | Time: {elapsed:.1f}s",
                    flush=True,
                )
                eval_history.append({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": avg_val_loss,
                    "val_ppl": val_ppl,
                    "elapsed_s": elapsed,
                    "gate_stats": last_gate_stats,
                })

    return {
        "model_type": model_type,
        "parameters": n_params,
        "n_hops": n_hops,
        "n_steps": n_steps,
        "final_val_loss": eval_history[-1]["val_loss"],
        "final_val_ppl": eval_history[-1]["val_ppl"],
        "total_time_s": time.time() - t0,
        "history": eval_history,
    }


@app.local_entrypoint()
def main():
    models_to_test = [
        "canonical_subq_final_mlp",
        "canonical_subq_per_hop_mlp",
        "attention_driven_gru_cell",
    ]

    print("=" * 115)
    print("  LAUNCHING STUDY S4-001: ATTENTION-DRIVEN PARALLEL GRU CELL (T=8)")
    print("  Comparing Canonical SubQ (Final MLP), SubQ (Per-Hop MLP), and Parallel GRU Cell")
    print("=" * 115)

    t0 = time.time()
    # Spawn all 3 jobs in parallel on Modal A10G
    calls = [train_model.spawn(m, n_hops=8, seed=42, n_steps=2000) for m in models_to_test]

    results = {}
    for m, call in zip(models_to_test, calls):
        print(f"Waiting for {m}...", flush=True)
        res = call.get()
        results[m] = res
        print(f"--> Finished {m}: Val Loss = {res['final_val_loss']:.4f} | Val PPL = {res['final_val_ppl']:.2f} ({res['total_time_s']:.1f}s)", flush=True)

    total_parallel_time = time.time() - t0

    print("\n" + "=" * 115)
    print("  STUDY S4-001 FINAL SCORECARD (T=8 HOPS, 2,000 STEPS)")
    print("=" * 115)
    print(f"{'Architecture':<32} | {'Parameters':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Time (s)':<10}")
    print("-" * 115)
    for m in models_to_test:
        r = results[m]
        print(f"{r['model_type']:<32} | {r['parameters']:<12,d} | {r['final_val_loss']:<10.4f} | {r['final_val_ppl']:<10.2f} | {r['total_time_s']:<10.1f}")
    print("=" * 115)
    print(f"Total Wall-Clock Parallel Runtime: {total_parallel_time:.1f}s")

    os.makedirs("season4/results", exist_ok=True)
    out_json = "season4/results/s4_001_attention_driven_gru_cell.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved S4-001 results to {out_json}]")
