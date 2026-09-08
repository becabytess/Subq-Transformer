"""
Study 88D: MQAR Ablation - Eliminating Both W_q and W_v (No-QV and Pure Router)
Benchmark: L=512, 16 Interleaved Pairs, 8 Queries, 1500 Steps
Evaluates:
1. MultiScale SubQ (No W_q & No W_v, Identity Q & V)
2. Harmonic Wave SubQ (No W_q & No W_v, Identity Q & V, max_d=512)
3. Pure Wave Router (No W_q, No W_k, No W_v - Zero Dot Products, Pure Wave Attention)
"""

import math
import time
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy")
)

app = modal.App("study88d-mqar-no-qv", image=image)


@app.function(gpu="A10G", timeout=3600)
def run_no_qv_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY 88D: MQAR ABLATION - ELIMINATING BOTH W_q AND W_v (NO-QV & PURE ROUTER)")
    print("  Benchmark: L=512, 16 Interleaved KV Pairs, 8 Queries, 1500 Steps")
    print(f"  Container GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 95)

    vocab_size = 256
    seq_len = 512
    batch_size = 32
    num_kv_pairs = 16
    num_queries = 8
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 1500

    def generate_mqar_batch(batch_size, seq_len=512, num_kv=16, num_q=8):
        x = torch.randint(100, 255, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        query_marker = 1

        for b in range(batch_size):
            keys = torch.randperm(40)[:num_kv] + 10
            vals = keys + 40
            kv_positions = torch.randperm(350)[:num_kv * 2].sort().values
            for k_idx in range(num_kv):
                p_k = kv_positions[k_idx * 2].item()
                p_v = p_k + 1
                x[b, p_k] = keys[k_idx]
                x[b, p_v] = vals[k_idx]

            query_indices = torch.randperm(num_kv)[:num_q]
            start_q_pos = seq_len - (num_q * 2) - 2
            for q_idx, k_orig_idx in enumerate(query_indices):
                q_pos = start_q_pos + (q_idx * 2)
                x[b, q_pos] = query_marker
                x[b, q_pos + 1] = keys[k_orig_idx]
                y[b, q_pos + 1] = vals[k_orig_idx]

        return x, y

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 450, 511]

    # Model 1: MultiScale SubQ (No W_q & No W_v)
    class MultiScale_NoQV(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=4, d_model=128, n_heads=4, d_mlp=512, max_len=512):
            super().__init__()
            self.jump_offsets = jump_offsets
            self.K = len(jump_offsets)
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)
            self.ln1 = nn.LayerNorm(d_model)

            # Both W_q and W_v are ELIMINATED! Only W_k remains
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru = nn.GRUCell(d_model, d_model)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            target_indices = torch.zeros((max_len, self.K), dtype=torch.long)
            valid_jump_mask = torch.zeros((max_len, self.K), dtype=torch.bool)
            for i in range(max_len):
                for k, offset in enumerate(self.jump_offsets):
                    target_pos = i - offset
                    if target_pos >= 0:
                        target_indices[i, k] = target_pos
                        valid_jump_mask[i, k] = True

            self.register_buffer("target_indices", target_indices)
            self.register_buffer("valid_jump_mask", valid_jump_mask)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)
            target_idx = self.target_indices[:L]
            valid_mask = self.valid_jump_mask[:L]

            s = x_emb
            for step in range(self.T):
                s_norm = self.ln1(s)
                # Identity Q and Identity V
                q = s_norm.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                k = self.k_proj(s_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                v = s_norm.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                k_cand = k[:, :, target_idx, :]
                v_cand = v[:, :, target_idx, :]

                scores = (q.unsqueeze(3) * k_cand).sum(dim=-1) / math.sqrt(self.d_k)
                scores = scores.masked_fill(~valid_mask.view(1, 1, L, self.K), -1e9)
                pi = F.softmax(scores, dim=-1)

                surfed_v = (pi.unsqueeze(-1) * v_cand).sum(dim=3)
                surfed_v = surfed_v.transpose(1, 2).contiguous().view(B, L, self.d_model)
                surfed_out = self.out_proj(surfed_v)

                s_flat = s.view(B * L, self.d_model)
                out_flat = surfed_out.view(B * L, self.d_model)
                s_next = self.gru(out_flat, s_flat)
                s = s_next.view(B, L, self.d_model)

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    # Model 2: Harmonic Wave SubQ (No W_q & No W_v)
    class HarmonicWave_NoQV(nn.Module):
        def __init__(self, vocab_size, num_waves=12, K_peaks=16, T=4, d_model=128, n_heads=4, d_mlp=512, max_len=512):
            super().__init__()
            self.num_waves = num_waves
            self.K_peaks = K_peaks
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.max_d = max_len

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)
            self.ln1 = nn.LayerNorm(d_model)

            # Both W_q and W_v ELIMINATED!
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru = nn.GRUCell(d_model, d_model)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(nn.Linear(num_waves * 4, 32), nn.GELU(), nn.Linear(32, num_waves * 4))
            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, self.max_d - 1, 1))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            s = x_emb
            for step in range(self.T):
                s_norm = self.ln1(s)
                # Identity Q and Identity V
                q = s_norm.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                k = self.k_proj(s_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                v = s_norm.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                amp = torch.tanh(curr_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
                omega = (F.softplus(curr_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
                phi = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
                decay = (F.softplus(curr_params[..., 3]) * 0.01).view(1, self.n_heads, 1, self.num_waves)

                grid_slice = self.d_grid[:, :, :L - 1, :]
                wave_comps = amp * torch.cos(omega * grid_slice + phi) * torch.exp(-decay * grid_slice)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=min(self.K_peaks - 1, L - 1), dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=idx.device)
                zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=idx.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                K_curr = peak_offsets.shape[-1]
                target_indices = q_pos - peak_offsets.unsqueeze(2)
                valid_mask = target_indices >= 0
                target_indices_clamped = torch.clamp(target_indices, min=0)
                idx_exp = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, K_curr, self.d_k)

                k_cand = torch.gather(k.unsqueeze(3).expand(B, self.n_heads, L, K_curr, self.d_k), dim=2, index=idx_exp)
                v_cand = torch.gather(v.unsqueeze(3).expand(B, self.n_heads, L, K_curr, self.d_k), dim=2, index=idx_exp)

                scores = (q.unsqueeze(3) * k_cand).sum(dim=-1) / math.sqrt(self.d_k) + peak_vals.unsqueeze(2)
                scores = scores.masked_fill(~valid_mask, -1e9)
                pi = F.softmax(scores, dim=-1)

                surfed_v = (pi.unsqueeze(-1) * v_cand).sum(dim=3)
                surfed_v = surfed_v.transpose(1, 2).contiguous().view(B, L, self.d_model)
                surfed_out = self.out_proj(surfed_v)

                s_flat = s.view(B * L, self.d_model)
                out_flat = surfed_out.view(B * L, self.d_model)
                s_next = self.gru(out_flat, s_flat)
                s = s_next.view(B, L, self.d_model)

                if step < self.T - 1:
                    flat_p = curr_params.view(self.n_heads, self.num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(self.n_heads, self.num_waves, 4)

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    # Model 3: Pure Wave Router (No W_q, No W_k, No W_v - Zero Dot Products!)
    class HarmonicWave_PureRouter(nn.Module):
        def __init__(self, vocab_size, num_waves=12, K_peaks=16, T=4, d_model=128, n_heads=4, d_mlp=512, max_len=512):
            super().__init__()
            self.num_waves = num_waves
            self.K_peaks = K_peaks
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.max_d = max_len

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)
            self.ln1 = nn.LayerNorm(d_model)

            # ALL PROJECTIONS W_q, W_k, W_v ARE ELIMINATED!
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru = nn.GRUCell(d_model, d_model)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(nn.Linear(num_waves * 4, 32), nn.GELU(), nn.Linear(32, num_waves * 4))
            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, self.max_d - 1, 1))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            s = x_emb
            for step in range(self.T):
                s_norm = self.ln1(s)
                # Identity V
                v = s_norm.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                amp = torch.tanh(curr_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
                omega = (F.softplus(curr_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
                phi = (curr_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
                decay = (F.softplus(curr_params[..., 3]) * 0.01).view(1, self.n_heads, 1, self.num_waves)

                grid_slice = self.d_grid[:, :, :L - 1, :]
                wave_comps = amp * torch.cos(omega * grid_slice + phi) * torch.exp(-decay * grid_slice)
                wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

                topk_vals, past_peak_offsets = torch.topk(wave_1d, k=min(self.K_peaks - 1, L - 1), dim=-1)
                past_peak_offsets = past_peak_offsets + 1
                zero_offset = torch.zeros((B, self.n_heads, 1), dtype=torch.long, device=idx.device)
                zero_val = torch.zeros((B, self.n_heads, 1), dtype=torch.float, device=idx.device)
                peak_offsets = torch.cat([zero_offset, past_peak_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                K_curr = peak_offsets.shape[-1]
                target_indices = q_pos - peak_offsets.unsqueeze(2)
                valid_mask = target_indices >= 0
                target_indices_clamped = torch.clamp(target_indices, min=0)
                idx_exp = target_indices_clamped.unsqueeze(-1).expand(B, self.n_heads, L, K_curr, self.d_k)

                v_cand = torch.gather(v.unsqueeze(3).expand(B, self.n_heads, L, K_curr, self.d_k), dim=2, index=idx_exp)

                # Pure wave bias: ZERO dot products!
                scores = peak_vals.unsqueeze(2).expand(-1, -1, L, -1)
                scores = scores.masked_fill(~valid_mask, -1e9)
                pi = F.softmax(scores, dim=-1)

                surfed_v = (pi.unsqueeze(-1) * v_cand).sum(dim=3)
                surfed_v = surfed_v.transpose(1, 2).contiguous().view(B, L, self.d_model)
                surfed_out = self.out_proj(surfed_v)

                s_flat = s.view(B * L, self.d_model)
                out_flat = surfed_out.view(B * L, self.d_model)
                s_next = self.gru(out_flat, s_flat)
                s = s_next.view(B, L, self.d_model)

                if step < self.T - 1:
                    flat_p = curr_params.view(self.n_heads, self.num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(self.n_heads, self.num_waves, 4)

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    models = {
        "1. MultiScale SubQ (No-QV, Only W_k)": MultiScale_NoQV(vocab_size, jump_offsets, T=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
        "2. Harmonic Wave SubQ (No-QV, Only W_k, max_d=512)": HarmonicWave_NoQV(vocab_size, num_waves=12, K_peaks=16, T=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
        "3. Harmonic Wave Pure Router (Zero Dot-Products, No Q/K/V)": HarmonicWave_PureRouter(vocab_size, num_waves=12, K_peaks=16, T=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
    }

    results = {}
    for name, model in models.items():
        print(f"\n" + "-" * 95)
        print(f"---> Training MQAR on: {name}")
        params = sum(p.numel() for p in model.parameters())
        print(f"     Parameter Count: {params:,}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

        t0 = time.time()
        for step in range(1, num_steps + 1):
            model.train()
            x, y = generate_mqar_batch(batch_size, seq_len, num_kv_pairs, num_queries)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 250 == 0 or step == 1 or step == num_steps:
                mask = (y != -100)
                preds = logits.argmax(dim=-1)
                acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
                print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Recall Acc: {acc:.1f}% | Elapsed: {time.time()-t0:.1f}s")

        model.eval()
        total_queries, correct_queries = 0, 0
        with torch.no_grad():
            for _ in range(50):
                x_val, y_val = generate_mqar_batch(32, seq_len, num_kv_pairs, num_queries)
                logits = model(x_val)
                mask = (y_val != -100)
                preds = logits.argmax(dim=-1)
                correct_queries += (preds[mask] == y_val[mask]).sum().item()
                total_queries += mask.sum().item()

        val_acc = (correct_queries / total_queries) * 100.0
        elapsed = time.time() - t0
        results[name] = {"params": params, "val_accuracy": val_acc, "time_s": elapsed}
        print(f"     => FINAL EXACT-MATCH RECALL ACCURACY: {val_acc:.2f}% (Time: {elapsed:.1f}s)")

    print("\n" + "=" * 95)
    print("  STUDY 88D SUMMARY (NO-QV & PURE ROUTER)")
    print("=" * 95)
    for name, r in results.items():
        print(f"{name:<58} | {r['params']:<10,d} | {r['val_accuracy']:<13.2f}% | {r['time_s']:<7.1f}s")
    print("=" * 95)

    return results


@app.local_entrypoint()
def main():
    print("Launching Study 88D (No-QV & Pure Router) on Modal...")
    res = run_no_qv_ablation.remote()
    print("Study 88D Complete!")
