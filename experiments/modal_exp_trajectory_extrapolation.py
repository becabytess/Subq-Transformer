import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests",
        "tiktoken"
    )
)

app = modal.App("subq-trajectory-extrapolation", image=image)

@app.function(gpu="T4", timeout=900)
def run_extrapolation_study():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import tiktoken

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  STUDY: TRAJECTORY EXTRAPOLATION & CURVATURE-GATED ACCELERATION (BPE TOKENS)")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Setup
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_text = response.read().decode('utf-8')
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    vocab_size = enc.n_vocab

    print(f"Corpus: {len(tokens):,} BPE tokens | Vocabulary: {vocab_size:,}")

    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    seq_len = 256
    batch_size = 32
    d_model = 256
    d_mlp = 1024
    n_heads = 8
    head_dim = d_model // n_heads
    fib_jumps = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127]
    K = len(fib_jumps)

    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # 2. SubQ Model Architecture
    class SubQSurferBPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x_norm, T=8):
            B, L, D = x_norm.shape
            pos = torch.arange(L, device=device)
            jump_indices = []
            valid_masks = []
            for j in fib_jumps:
                idx = pos - j
                mask = idx >= 0
                idx = torch.clamp(idx, min=0)
                jump_indices.append(idx)
                valid_masks.append(mask)
            jump_indices = torch.stack(jump_indices, dim=-1)
            valid_masks = torch.stack(valid_masks, dim=-1)

            q = self.q_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, jump_indices, :]
            v_cand = v[:, :, jump_indices, :]

            scores = (q.unsqueeze(3) * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~valid_masks.unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
            gates_ctx = self.w_ih(ctx)
            r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

            states = [x_norm]
            s = x_norm
            for _ in range(T):
                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)
                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s
                states.append(s)

            return self.out_proj(s), [self.out_proj(st) for st in states]

    class SubQLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = SubQSurferBPE()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, T=4, return_states=False):
            B, L = idx.shape
            x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
            x_norm = self.ln1(x)
            surfer_out, traj = self.surfer(x_norm, T=T)
            
            if not return_states:
                x_settled = x + surfer_out
                x_out = x_settled + self.mlp(self.ln2(x_settled))
                return self.head(self.norm(x_out))

            return x, traj

        def decode_state(self, x, st):
            x_t = x + st
            x_t_out = x_t + self.mlp(self.ln2(x_t))
            return self.head(self.norm(x_t_out))

    total_steps = 800
    model = SubQLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    print("Training SubQ model (800 steps with mixed T in [1, 6])...")
    start = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        x, y = get_batch("train")
        T_train = np.random.randint(1, 7)
        logits = model(x, T=T_train)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Loss: {loss.item():.4f} | PPL: {math.exp(loss.item()):.2f}")

    print(f"Training completed in {time.time() - start:.1f}s\n")
    model.eval()

    # -------------------------------------------------------------------------
    # PART 1: Geometric Extrapolation vs Ground Truth Settled State (s^8)
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("  PART 1: STATE VECTOR EXTRAPOLATION ACCURACY (Measuring distance to s^8)")
    print("=" * 90)

    n_eval = 40
    T_max = 8

    # Accumulators
    cos_sim_raw = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    cos_sim_extrap_2h = 0.0
    cos_sim_extrap_3h = 0.0

    loss_raw = {1: 0.0, 2: 0.0, 3: 0.0, 8: 0.0}
    loss_extrap_2h = 0.0
    loss_extrap_3h = 0.0

    straight_tokens_sample = []
    curved_tokens_sample = []

    with torch.no_grad():
        for b_idx in range(n_eval):
            x, y = get_batch("val")
            emb_x, traj = model(x, T=T_max, return_states=True)

            s0 = traj[0] # [B, L, D]
            s1 = traj[1]
            s2 = traj[2]
            s3 = traj[3]
            s8 = traj[8] # Ground truth target destination

            # 1. Measure standard unrolled states
            for t_step in [1, 2, 3, 8]:
                logits_t = model.decode_state(emb_x, traj[t_step])
                loss_t = F.cross_entropy(logits_t.view(-1, vocab_size), y.view(-1)).item()
                loss_raw[t_step] += loss_t / n_eval
                del logits_t

                if t_step in cos_sim_raw:
                    sim = F.cosine_similarity(traj[t_step], s8, dim=-1).mean().item()
                    cos_sim_raw[t_step] += sim / n_eval

            # 2. Geometric Extrapolation at T=2 (Using v1 -> v2)
            # Velocity: v2 = s2 - s1
            v1 = s1 - s0
            v2 = s2 - s1
            v3 = s3 - s2

            # Contraction ratio estimate: gamma = ||v2|| / ||v1||
            norm_v1 = v1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_v2 = v2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_v3 = v3.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            
            gamma_2 = (norm_v2 / norm_v1).clamp(0.0, 0.95)
            # Extrapolated fixed point: s* = s2 + (gamma / (1 - gamma)) * v2
            s_extrap_2h = s2 + (gamma_2 / (1.0 - gamma_2)) * v2

            # Anderson Extrapolation at T=3: s* = s3 + (gamma3 / (1 - gamma3)) * v3
            gamma_3 = (norm_v3 / norm_v2).clamp(0.0, 0.95)
            s_extrap_3h = s3 + (gamma_3 / (1.0 - gamma_3)) * v3

            # Cosine similarities to true s8
            sim_2h = F.cosine_similarity(s_extrap_2h, s8, dim=-1).mean().item()
            sim_3h = F.cosine_similarity(s_extrap_3h, s8, dim=-1).mean().item()
            cos_sim_extrap_2h += sim_2h / n_eval
            cos_sim_extrap_3h += sim_3h / n_eval

            # Decode losses for extrapolated states
            logits_extrap_2 = model.decode_state(emb_x, s_extrap_2h)
            loss_extrap_2h += F.cross_entropy(logits_extrap_2.view(-1, vocab_size), y.view(-1)).item() / n_eval
            del logits_extrap_2

            logits_extrap_3 = model.decode_state(emb_x, s_extrap_3h)
            loss_extrap_3h += F.cross_entropy(logits_extrap_3.view(-1, vocab_size), y.view(-1)).item() / n_eval
            del logits_extrap_3

            # 3. Curvature Profiling: cos_theta between v2 and v3
            if b_idx == 0:
                cos_theta = (v2 * v3).sum(dim=-1) / (norm_v2.squeeze(-1) * norm_v3.squeeze(-1)) # [B, L]
                straight_idx = (cos_theta > 0.95).nonzero()
                curved_idx = (cos_theta < 0.70).nonzero()

                for p in straight_idx[:20]:
                    tok_id = x[p[0], p[1]].item()
                    straight_tokens_sample.append(enc.decode([tok_id]))
                for p in curved_idx[:20]:
                    tok_id = x[p[0], p[1]].item()
                    curved_tokens_sample.append(enc.decode([tok_id]))

    print(f"\n{'State / Inference Method':<38} | {'Hops Used':<10} | {'Cosine Sim to s^8':<20} | {'Val Loss':<10} | {'Perplexity':<12}")
    print("-" * 96)
    print(f"{'1. Raw Hop 1 State (s^1)':<38} | {'1 hop':<10} | {cos_sim_raw[1]:<20.4f} | {loss_raw[1]:<10.4f} | {math.exp(loss_raw[1]):<12.2f}")
    print(f"{'2. Raw Hop 2 State (s^2)':<38} | {'2 hops':<10} | {cos_sim_raw[2]:<20.4f} | {loss_raw[2]:<10.4f} | {math.exp(loss_raw[2]):<12.2f}")
    print(f"{'3. Raw Hop 3 State (s^3)':<38} | {'3 hops':<10} | {cos_sim_raw[3]:<20.4f} | {loss_raw[3]:<10.4f} | {math.exp(loss_raw[3]):<12.2f}")
    print(f"{'4. True Settled State (s^8 Ground Truth)':<38} | {'8 hops':<10} | {'1.0000 (Target)':<20} | {loss_raw[8]:<10.4f} | {math.exp(loss_raw[8]):<12.2f}")
    print("-" * 96)
    print(f"{'★ 2-Hop Extrapolated State (s* @ T=2)':<38} | {'2 hops':<10} | {cos_sim_extrap_2h:<20.4f} | {loss_extrap_2h:<10.4f} | {math.exp(loss_extrap_2h):<12.2f}")
    print(f"{'★ 3-Hop Anderson Extrapolated (s* @ T=3)':<38} | {'3 hops':<10} | {cos_sim_extrap_3h:<20.4f} | {loss_extrap_3h:<10.4f} | {math.exp(loss_extrap_3h):<12.2f}")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # PART 2: Curvature & Token Linguistic Breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  PART 2: CURVATURE (cos theta) AS GEOMETRIC UNCERTAINTY SIGNAL")
    print("=" * 90)
    print("\n[A] High Straightness Tokens (cos theta > 0.95 -> Direct Geodesic to Endpoint):")
    print("   " + ", ".join([repr(t) for t in straight_tokens_sample[:15]]))

    print("\n[B] High Curvature Tokens (cos theta < 0.70 -> Bending / Resolving Context):")
    print("   " + ", ".join([repr(t) for t in curved_tokens_sample[:15]]))
    print("=" * 90)

@app.local_entrypoint()
def main():
    run_extrapolation_study.remote()
