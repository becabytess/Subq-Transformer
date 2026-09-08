import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("study86c-mqar-stress-test", image=image)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_mqar_stress_test():
    import math, time, json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  STUDY 86C: MULTI-QUERY ASSOCIATIVE RECALL (MQAR) STRESS TEST")
    print("  Task designed to be IMPOSSIBLE for standard GRUs (catastrophic finite-state bottleneck)")
    print("  Evaluating: Pure GRU vs Dense Transformer vs Pure SubQ vs GRU+SubQ vs GRU+SubQ (No-QV)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads # 32
    num_waves = 12
    K_peaks = 8
    max_d = 128
    T = 8
    d_mlp = 512
    total_steps = 1500
    eval_interval = 250
    num_pairs = 8
    num_keys = 32
    num_vals = 32

    # Vocab: 0..31: Keys, 32..63: Vals, 64..79: Noise, 80: Prompt ':'
    vocab_size = num_keys + num_vals + 16 + 1 # 81

    def get_mqar_batch(split, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randint(64, 80, (batch_size, seq_len), device=device)
        targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

        max_prefix_idx = (seq_len - 2 * num_pairs - 4) // 2
        for b in range(batch_size):
            keys = torch.randperm(num_keys, device=device)[:num_pairs]
            vals = torch.randint(32, 64, (num_pairs,), device=device)
            pair_slots = torch.randperm(max_prefix_idx, device=device)[:num_pairs].sort()[0] * 2
            for p, k, v in zip(pair_slots, keys, vals):
                x[b, p] = k
                x[b, p + 1] = v

            query_start = seq_len - num_pairs * 2
            perm = torch.randperm(num_pairs, device=device)
            for i, q_idx in enumerate(perm):
                k = keys[q_idx]
                v = vals[q_idx]
                pos = query_start + i * 2
                x[b, pos] = k
                x[b, pos + 1] = 80 # prompt token
                targets[b, pos + 1] = v

        return x, targets

    # -------------------------------------------------------------------------
    # Model 1: Pure GRU (Fixed-Capacity State Bottleneck)
    # -------------------------------------------------------------------------
    class PureGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True, bidirectional=False)
            self.ln_gru = nn.LayerNorm(d_model)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            gru_out, _ = self.gru(x)
            s = self.ln_gru(x + gru_out)
            s = s + self.mlp(self.ln_2(s))
            return self.head(self.ln_f(s))

    # -------------------------------------------------------------------------
    # Model 2: 1L Causal Dense Transformer
    # -------------------------------------------------------------------------
    class CausalDense1L(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
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
    # Model 3: Pure SubQ Alone (Zero GRU, T=8)
    # -------------------------------------------------------------------------
    class PureSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.ln_1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(nn.Linear(num_waves * 4, 32), nn.GELU(), nn.Linear(32, num_waves * 4))
            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            s = self.tok_emb(idx) + self.pos_emb(pos)
            inv_sqrt_T = 1.0 / math.sqrt(T)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            for t in range(T):
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

                if t < T - 1:
                    flat_p = curr_params.view(n_heads, num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(n_heads, num_waves, 4)

            return self.head(self.ln_f(s))

    # -------------------------------------------------------------------------
    # Model 4: GRU + SubQ Full QKV (T=8)
    # -------------------------------------------------------------------------
    class GRUSubQ_Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True, bidirectional=False)
            self.ln_gru = nn.LayerNorm(d_model)
            self.ln_1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(nn.Linear(num_waves * 4, 32), nn.GELU(), nn.Linear(32, num_waves * 4))
            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            gru_out, _ = self.gru(x)
            s = self.ln_gru(x + gru_out)

            inv_sqrt_T = 1.0 / math.sqrt(T)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            for t in range(T):
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

                if t < T - 1:
                    flat_p = curr_params.view(n_heads, num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(n_heads, num_waves, 4)

            return self.head(self.ln_f(s))

    # -------------------------------------------------------------------------
    # Model 5: GRU + SubQ No-Q & No-V (Only W_k, W_o, T=8)
    # -------------------------------------------------------------------------
    class GRUSubQ_NoQV(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True, bidirectional=False)
            self.ln_gru = nn.LayerNorm(d_model)
            self.ln_1 = nn.LayerNorm(d_model)
            # W_q and W_v removed!
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)
            self.wave_transition = nn.Sequential(nn.Linear(num_waves * 4, 32), nn.GELU(), nn.Linear(32, num_waves * 4))
            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            gru_out, _ = self.gru(x)
            s = self.ln_gru(x + gru_out)

            inv_sqrt_T = 1.0 / math.sqrt(T)
            curr_params = self.raw_wave_params
            q_pos = torch.arange(L, device=idx.device).view(1, 1, L, 1)

            for t in range(T):
                s_norm = self.ln_1(s)
                q = s_norm.view(B, L, n_heads, d_k).transpose(1, 2)
                k = self.k_proj(s_norm).view(B, L, n_heads, d_k).transpose(1, 2)
                v = s_norm.view(B, L, n_heads, d_k).transpose(1, 2)

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

                if t < T - 1:
                    flat_p = curr_params.view(n_heads, num_waves * 4)
                    curr_params = (flat_p + 0.1 * self.wave_transition(flat_p)).view(n_heads, num_waves, 4)

            return self.head(self.ln_f(s))

    # -------------------------------------------------------------------------
    # Train and Evaluate Loop on MQAR Retrieval Accuracy
    # -------------------------------------------------------------------------
    def train_and_eval(model, model_name):
        model.to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n" + "-" * 115)
        print(f"  TRAINING MQAR: {model_name} ({params:,} parameters)")
        print("-" * 115)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        scaler = torch.amp.GradScaler('cuda')

        t0 = time.time()
        for step in range(1, total_steps + 1):
            model.train()
            x, targets = get_mqar_batch('train', seed=step * 1000 + 42)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % eval_interval == 0 or step == total_steps:
                elapsed = time.time() - t0
                print(f"    Step {step:>4d}/{total_steps} ({elapsed:.1f}s) | MQAR Loss: {loss.item():.4f}")

        train_time = time.time() - t0

        # Evaluate Exact Retrieval Accuracy over 50 test batches
        model.eval()
        total_correct = 0
        total_queries = 0
        with torch.no_grad():
            for v_step in range(50):
                x_val, targets_val = get_mqar_batch('val', seed=v_step * 5000 + 1337)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x_val)
                preds = logits.argmax(dim=-1)
                mask = targets_val != -100
                total_correct += ((preds == targets_val) & mask).sum().item()
                total_queries += mask.sum().item()

        retrieval_acc = (total_correct / total_queries) * 100.0
        print(f"  --> {model_name}: MQAR Retrieval Accuracy: {retrieval_acc:.2f}% ({total_correct}/{total_queries}) | Time: {train_time:.1f}s")
        return {
            "name": model_name,
            "params": params,
            "accuracy": round(retrieval_acc, 2),
            "time": round(train_time, 1)
        }

    models_to_test = [
        (PureGRU(), "1. Pure GRU Alone (Finite-State Bottleneck)"),
        (CausalDense1L(), "2. 1L Causal Dense Transformer (Full Attention)"),
        (PureSubQ(), "3. Pure SubQ Alone (Zero GRU, T=8)"),
        (GRUSubQ_Full(), "4. GRU + SubQ Full QKV (T=8)"),
        (GRUSubQ_NoQV(), "5. GRU + SubQ No-Q & No-V (Only Wk, Wo, T=8)")
    ]

    results = []
    for model_inst, name in models_to_test:
        res = train_and_eval(model_inst, name)
        results.append(res)

    print("\n" + "=" * 125)
    print("  STUDY 86C FINAL RESULTS: MQAR ASSOCIATIVE RETRIEVAL STRESS TEST")
    print("=" * 125)
    print(f"{'Model Architecture':<55} | {'Params':<10} | {'Retrieval Acc (%)':<20} | {'Train Time':<10}")
    print("-" * 125)
    for r in results:
        print(f"{r['name']:<55} | {r['params']:<10,d} | {r['accuracy']:<20.2f}% | {r['time']:<10.1f}s")
    print("=" * 125)

    with open("/models/study86c_mqar_stress_results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results

@app.local_entrypoint()
def main():
    run_mqar_stress_test.remote()
