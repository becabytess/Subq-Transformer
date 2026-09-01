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

app = modal.App("subq-bpe-parallel-benchmark", image=image)

# Common helper for dataset
def get_corpus():
    import urllib.request
    import tiktoken
    import torch

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_text = response.read().decode('utf-8')
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    return data, enc

# -----------------------------------------------------------------------------
# CONTAINER 1: Fixed Uniform T=3 Baseline (Runs on dedicated T4 GPU)
# -----------------------------------------------------------------------------
@app.function(gpu="T4", timeout=900)
def run_fixed_t3_worker():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker 1: Fixed T=3] Starting on {torch.cuda.get_device_name(0)}")

    data, enc = get_corpus()
    vocab_size = enc.n_vocab
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

        def forward(self, x_norm, T=3):
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

            s = x_norm
            for _ in range(T):
                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)
                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s

            return self.out_proj(s)

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

        def forward(self, idx, T=3):
            B, L = idx.shape
            x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
            x_norm = self.ln1(x)
            surfer_out = self.surfer(x_norm, T=T)
            x_settled = x + surfer_out
            x_out = x_settled + self.mlp(self.ln2(x_settled))
            return self.head(self.norm(x_out))

    total_steps = 800
    model = SubQLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    for step in range(1, total_steps + 1):
        model.train()
        x, y = get_batch("train")
        logits = model(x, T=3)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    val_loss = 0.0
    n_eval = 30
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = get_batch("val")
            logits = model(x, T=3)
            val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item() / n_eval
            del logits

    ppl = math.exp(val_loss)
    print(f"[Worker 1 DONE] Fixed T=3 -> Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
    return {"mode": "Fixed T=3", "val_loss": val_loss, "perplexity": ppl}


# -----------------------------------------------------------------------------
# CONTAINER 2: Dynamic Precision Allocation (Runs concurrently on dedicated T4)
# -----------------------------------------------------------------------------
@app.function(gpu="T4", timeout=900)
def run_dynamic_precision_worker():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker 2: Dynamic Precision] Starting on {torch.cuda.get_device_name(0)}")

    data, enc = get_corpus()
    vocab_size = enc.n_vocab
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

        def forward(self, x_norm, T=6):
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

        def forward(self, idx, T=4, return_reps=False):
            B, L = idx.shape
            x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
            x_norm = self.ln1(x)
            surfer_out, traj = self.surfer(x_norm, T=T)
            
            if not return_reps:
                x_settled = x + surfer_out
                x_out = x_settled + self.mlp(self.ln2(x_settled))
                return self.head(self.norm(x_out))

            step_reps = []
            for st in traj:
                x_t = x + st
                x_t_out = x_t + self.mlp(self.ln2(x_t))
                step_reps.append(self.norm(x_t_out))
            return traj, step_reps

    total_steps = 800
    model = SubQLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

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

    model.eval()
    val_loss = 0.0
    n_eval = 30
    hop_dist = {1: 0, 2: 0, 4: 0, 5: 0}
    sample_t1 = []
    sample_t5 = []

    with torch.no_grad():
        for batch_idx in range(n_eval):
            x, y = get_batch("val")
            traj, step_reps = model(x, T=6, return_reps=True)
            
            # Error / velocity at Hop 1
            e_1 = (traj[1] - traj[0]).norm(dim=-1) # [B, L]
            q1, q2, q3 = torch.quantile(e_1, torch.tensor([0.30, 0.50, 0.70], device=device))

            T_alloc = torch.ones_like(e_1, dtype=torch.long) * 3
            T_alloc[e_1 < q1] = 1
            T_alloc[(e_1 >= q1) & (e_1 < q2)] = 2
            T_alloc[(e_1 >= q2) & (e_1 < q3)] = 4
            T_alloc[e_1 >= q3] = 5

            if batch_idx == 0:
                t1_idx = (T_alloc == 1).nonzero()
                t5_idx = (T_alloc == 5).nonzero()
                for p in t1_idx[:25]:
                    sample_t1.append(enc.decode([x[p[0], p[1]].item()]))
                for p in t5_idx[:25]:
                    sample_t5.append(enc.decode([x[p[0], p[1]].item()]))

            rep_alloc = torch.zeros_like(step_reps[0])
            for t_val in [1, 2, 4, 5]:
                mask = (T_alloc == t_val).unsqueeze(-1)
                rep_alloc = torch.where(mask, step_reps[t_val], rep_alloc)
                hop_dist[t_val] += (T_alloc == t_val).sum().item()

            logits = model.head(rep_alloc)
            val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item() / n_eval
            del logits, rep_alloc

    tot = sum(hop_dist.values())
    avg_t = sum(k * v for k, v in hop_dist.items()) / tot
    ppl = math.exp(val_loss)
    print(f"[Worker 2 DONE] Dynamic Precision (Mean T={avg_t:.2f}) -> Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

    return {
        "mode": "Dynamic Precision",
        "mean_t": avg_t,
        "val_loss": val_loss,
        "perplexity": ppl,
        "sample_t1": sample_t1,
        "sample_t5": sample_t5
    }

# -----------------------------------------------------------------------------
# LOCAL ENTRYPOINT: Spawns both workers simultaneously
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  LAUNCHING PARALLEL MODAL GPU WORKERS (2x Tesla T4)")
    print("=" * 80)
    
    # Run both functions concurrently in parallel
    handle_fixed = run_fixed_t3_worker.spawn()
    handle_dynamic = run_dynamic_precision_worker.spawn()

    res_fixed = handle_fixed.get()
    res_dynamic = handle_dynamic.get()

    print("\n" + "=" * 80)
    print("  FINAL PARALLEL BENCHMARK RESULTS (BPE TOKENS - 50,257 VOCAB)")
    print("=" * 80)
    print(f"| {'Model Architecture / Strategy':<38} | {'Mean T':<8} | {'Val Loss':<10} | {'Perplexity':<12} |")
    print("-" * 80)
    print(f"| {res_fixed['mode']:<38} | {3.00:<8.2f} | {res_fixed['val_loss']:<10.4f} | {res_fixed['perplexity']:<12.2f} |")
    print(f"| {res_dynamic['mode']:<38} | {res_dynamic['mean_t']:<8.2f} | {res_dynamic['val_loss']:<10.4f} | {res_dynamic['perplexity']:<12.2f} |")
    print("-" * 80)
    delta_ppl = res_fixed['perplexity'] - res_dynamic['perplexity']
    print(f"Net Perplexity Difference: {delta_ppl:+.2f} PPL")

    print("\n[Token Breakdown Samples]")
    print(f"T=1 Tokens (Low Error / Fast Exit): {', '.join([repr(t) for t in res_dynamic['sample_t1'][:15]])}")
    print(f"T=5 Tokens (High Error / Deep Thought): {', '.join([repr(t) for t in res_dynamic['sample_t5'][:15]])}")
    print("=" * 80)
