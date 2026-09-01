import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy")
)

app = modal.App("exp-dyck4-rematch", image=image)

@app.function(gpu="A10G", timeout=2400)
def run_dyck4_rematch():
    import math
    import random
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 100)
    print("  STUDY 60: THE DYCK-4 DEEP BRACKET REMATCH")
    print("  Evaluating New Harmonic SubQ (Full Evolving Q,K,V) vs Standard 4-Layer Dense Transformer")
    print("  Nesting Depths 1 to 30+ | L = 256 | 1,500 Steps | Rigorous Depth Tier Breakdown")
    print("=" * 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    vocab_size = 16
    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    d_mlp = 512
    num_steps = 1500
    num_waves = 12
    K_peaks = 8
    max_d = 128

    open_to_close = {1: 2, 3: 4, 5: 6, 7: 8}
    open_brackets = [1, 3, 5, 7]

    def generate_dyck_sequence(target_len=256, max_depth=30):
        tokens = []
        stack = []
        depth_at_pos = []

        while len(tokens) < target_len - 1:
            curr_depth = len(stack)
            p_close = 0.0 if curr_depth == 0 else (0.45 if curr_depth < max_depth else 0.90)

            if len(tokens) + curr_depth >= target_len - 1:
                p_close = 1.0

            if random.random() < p_close and curr_depth > 0:
                last_open = stack.pop()
                expected_close = open_to_close[last_open]
                tokens.append(expected_close)
                depth_at_pos.append(curr_depth)
            else:
                b = random.choice(open_brackets)
                stack.append(b)
                tokens.append(b)
                depth_at_pos.append(len(stack))

        while stack and len(tokens) < target_len:
            last_open = stack.pop()
            tokens.append(open_to_close[last_open])
            depth_at_pos.append(len(stack) + 1)

        while len(tokens) < target_len:
            tokens.append(0)
            depth_at_pos.append(0)

        tokens = tokens[:target_len]
        depth_at_pos = depth_at_pos[:target_len]
        return tokens, depth_at_pos

    def generate_dyck_batch(batch_size, seq_len=256, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        depths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        for b in range(batch_size):
            tokens, depth_list = generate_dyck_sequence(seq_len, max_depth=30)
            x[b] = torch.tensor(tokens, device=device)
            depths[b] = torch.tensor(depth_list, device=device)

            for t in range(seq_len - 1):
                next_tok = tokens[t + 1]
                if next_tok in [2, 4, 6, 8]:
                    y[b, t] = next_tok

        return x, y, depths

    # -------------------------------------------------------------------------
    # 1. Standard 4-Layer Dense Transformer
    # -------------------------------------------------------------------------
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_mlp):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.d_k = d_model // n_heads
            self.n_heads = n_heads

        def forward(self, x, mask):
            B, L, D = x.shape
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + mask[:, :, :L, :L]
            attn = F.softmax(scores, dim=-1)
            ctx = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
            x = x + self.out(ctx)
            x = x + self.mlp(self.ln2(x))
            return x

    class DyckTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers=4, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, mask):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x, mask)
            return self.head(self.ln_f(x))

    # -------------------------------------------------------------------------
    # 2. Old Gravimem Baseline (Static K,V, Fixed Jumps, T=4)
    # -------------------------------------------------------------------------
    class OldDyckGravimem(nn.Module):
        def __init__(self, vocab_size, d_model=128, d_mlp=512, T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 128]
            self.K = len(self.jump_offsets)
            self.T = T
            self.d_model = d_model

            self.ln1 = nn.LayerNorm(d_model)
            self.jump_policy = nn.Linear(d_model, self.K)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru = nn.GRUCell(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            targets = []
            valid = []
            for i in range(seq_len):
                row_t = []
                row_v = []
                for off in self.jump_offsets:
                    target = i - off
                    if target >= 0:
                        row_t.append(target)
                        row_v.append(True)
                    else:
                        row_t.append(0)
                        row_v.append(False)
                targets.append(row_t)
                valid.append(row_v)
            self.register_buffer("target_indices", torch.tensor(targets, dtype=torch.long))
            self.register_buffer("valid_jump_mask", torch.tensor(valid, dtype=torch.bool))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            # Old static V from input
            V = self.v_proj(self.ln1(x_emb))
            target_idx = self.target_indices[:L]
            valid_mask = self.valid_jump_mask[:L]

            flat_targets = target_idx.unsqueeze(0).expand(B, -1, -1)
            V_jumps = torch.gather(
                V.unsqueeze(2).expand(-1, -1, self.K, -1),
                1,
                flat_targets.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            )
            V_jumps = V_jumps.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(-1), 0.0)

            s = x_emb
            for step in range(self.T):
                s_norm = self.ln1(s)
                policy_logits = self.jump_policy(s_norm)
                policy_logits = policy_logits.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                pi = F.softmax(policy_logits, dim=-1)

                surfed_v = torch.sum(pi.unsqueeze(-1) * V_jumps, dim=2)
                surfed_out = self.out_proj(surfed_v)

                s_flat = s.view(B * L, self.d_model)
                out_flat = surfed_out.view(B * L, self.d_model)
                s_next = self.gru(out_flat, s_flat)
                s = s_next.view(B, L, self.d_model)

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    # -------------------------------------------------------------------------
    # 3. New Harmonic SubQ (Full Evolving Q, K, V + Wave-Peak Routing)
    # -------------------------------------------------------------------------
    class HarmonicSubQAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, num_waves=12, K_peaks=8, max_d=128):
            super().__init__()
            self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
            self.num_waves, self.K_peaks, self.max_d = num_waves, K_peaks, max_d

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.raw_wave_params = nn.Parameter(torch.randn(n_heads, num_waves, 4) * 0.1)

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(1, 1, max_d - 1, 1))

        def forward(self, x, s):
            B, L, D = x.shape
            # Q, K, V are ALL projected from evolving recurrent state s!
            Q = self.q_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(s).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            amp = torch.tanh(self.raw_wave_params[..., 0]).view(1, self.n_heads, 1, self.num_waves)
            omega = (F.softplus(self.raw_wave_params[..., 1]).view(1, self.n_heads, 1, self.num_waves) * self.base_freqs)
            phi = (self.raw_wave_params[..., 2] * math.pi).view(1, self.n_heads, 1, self.num_waves)
            decay = (F.softplus(self.raw_wave_params[..., 3]) * 0.05).view(1, self.n_heads, 1, self.num_waves)

            wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
            wave_1d = wave_comps.sum(dim=-1).expand(B, -1, -1)

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

    class HarmonicSubQLM(nn.Module):
        def __init__(self, vocab_size, T=4):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.attn = HarmonicSubQAttention(d_model=d_model, n_heads=n_heads, num_waves=num_waves, K_peaks=K_peaks, max_d=max_d)
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
    # Rematch Models to Test
    # -------------------------------------------------------------------------
    causal_mask = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    models_to_test = [
        ("1. Old 1L Gravimem (Static KV, T=4)", OldDyckGravimem(vocab_size, d_model=d_model, d_mlp=d_mlp, T=4), False),
        ("2. Standard 4L Dense Transformer (830k params)", DyckTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp), True),
        ("3. New Harmonic SubQ (1L, T=4, Evolving QKV)", HarmonicSubQLM(vocab_size, T=4), False),
        ("4. New Harmonic SubQ (1L, T=8, Evolving QKV)", HarmonicSubQLM(vocab_size, T=8), False),
    ]

    results = {}

    for name, model, is_dense in models_to_test:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n---> Training Dyck-4 Bracket Matching on [{name}] (Parameters: {param_count:,})...")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

        t0 = time.time()
        for step in range(1, num_steps + 1):
            model.train()
            x, y, _ = generate_dyck_batch(batch_size, seq_len, seed=1000 + step)
            optimizer.zero_grad()

            if is_dense:
                logits = model(x, causal_mask)
            else:
                logits = model(x)

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 500 == 0 or step == num_steps:
                mask = (y != -100)
                preds = logits.argmax(dim=-1)
                acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
                print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Closing Acc: {acc:.1f}% | Elapsed: {time.time()-t0:.1f}s")

        elapsed = time.time() - t0

        # Rigorous evaluation broken down by Nesting Depth Tiers:
        # Tier 1: Shallow (Depth 1-5)
        # Tier 2: Medium  (Depth 6-15)
        # Tier 3: Deep    (Depth 16-30)
        model.eval()
        tier_correct = {"Overall": 0, "Depth 1-5": 0, "Depth 6-15": 0, "Depth 16-30": 0}
        tier_total = {"Overall": 0, "Depth 1-5": 0, "Depth 6-15": 0, "Depth 16-30": 0}

        with torch.no_grad():
            for v_step in range(50):
                x_val, y_val, depth_val = generate_dyck_batch(32, seq_len, seed=90000 + v_step)
                if is_dense:
                    logits = model(x_val, causal_mask)
                else:
                    logits = model(x_val)

                mask = (y_val != -100)
                preds = logits.argmax(dim=-1)
                is_correct = (preds == y_val) & mask

                tier_correct["Overall"] += is_correct.sum().item()
                tier_total["Overall"] += mask.sum().item()

                # Depth 1-5
                m1 = mask & (depth_val <= 5)
                tier_correct["Depth 1-5"] += ((preds == y_val) & m1).sum().item()
                tier_total["Depth 1-5"] += m1.sum().item()

                # Depth 6-15
                m2 = mask & (depth_val > 5) & (depth_val <= 15)
                tier_correct["Depth 6-15"] += ((preds == y_val) & m2).sum().item()
                tier_total["Depth 6-15"] += m2.sum().item()

                # Depth 16-30
                m3 = mask & (depth_val > 15)
                tier_correct["Depth 16-30"] += ((preds == y_val) & m3).sum().item()
                tier_total["Depth 16-30"] += m3.sum().item()

        accs = {}
        for tier in tier_total:
            accs[tier] = (tier_correct[tier] / max(1, tier_total[tier])) * 100.0
            print(f"     => {tier:<12}: Accuracy = {accs[tier]:.2f}%")

        results[name] = {"params": param_count, "accs": accs, "time_s": elapsed}

    print("\n" + "=" * 110)
    print("  STUDY 60 FINAL SUMMARY: DYCK-4 DEEP BRACKET MATCHING REMATCH")
    print("=" * 110)
    print(f"{'Model Architecture':<44} | {'Params':<10} | {'Overall Acc':<12} | {'Depth 1-5':<11} | {'Depth 6-15':<12} | {'Depth 16-30':<12}")
    print("-" * 110)
    for name, r in results.items():
        a = r["accs"]
        print(f"{name:<44} | {r['params']:<10,d} | {a['Overall']:<11.2f}% | {a['Depth 1-5']:<10.2f}% | {a['Depth 6-15']:<11.2f}% | {a['Depth 16-30']:<11.2f}%")
    print("=" * 110)

    return results

@app.local_entrypoint()
def main():
    run_dyck4_rematch.remote()
