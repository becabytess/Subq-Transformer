"""
Modal Experiment: Shared SubQ Attention with Unique MLPs Across Layers
Investigates:
Can we reuse the exact same SubQ Attention / Surfer weights across multiple physical layers,
pairing it with distinct, unique MLPs (MLP_1, MLP_2, ..., MLP_L)?
Compares:
1. Standard 1-Layer SubQ (1 Surfer + 1 MLP)
2. 2-Layer Independent SubQ (2 Unique Surfers + 2 Unique MLPs)
3. 2-Layer Shared-Surfer SubQ (1 Shared Surfer + 2 Unique MLPs)
4. 4-Layer Shared-Surfer SubQ (1 Shared Surfer + 4 Unique MLPs)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests", "tiktoken")
)

app = modal.App("subq-shared-attention-benchmark", image=image)


@app.function(gpu="T4", timeout=1200)
def run_shared_attention_benchmark():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tiktoken

    print("=" * 80)
    print("  SUBQ: SHARED ATTENTION WITH UNIQUE MLPS BENCHMARK")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Container: {torch.cuda.get_device_name(0)}")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = urllib.request.urlopen(url).read().decode('utf-8')

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    vocab_size = enc.n_vocab

    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    block_size = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    head_dim = d_model // n_heads
    d_mlp = 512
    K = 16
    T = 3
    num_steps = 600

    # Fibonacci jump offsets
    offsets = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 255]
    K_jumps = len(offsets)

    cand_indices = torch.zeros((block_size, K_jumps), dtype=torch.long, device=device)
    cand_mask = torch.zeros((block_size, K_jumps), dtype=torch.bool, device=device)

    for i in range(block_size):
        for k_idx, off in enumerate(offsets):
            pos = i - off
            if pos >= 0:
                cand_indices[i, k_idx] = pos
                cand_mask[i, k_idx] = True
            else:
                cand_indices[i, k_idx] = 0
                cand_mask[i, k_idx] = False

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([d[i+1:i+block_size+1] for i in ix]).to(device)
        return x, y

    class SubQSurfer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_hh = nn.Linear(d_model, 3 * d_model, bias=False)

        def forward(self, x_norm):
            B, L, D = x_norm.shape
            q = self.q_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, cand_indices[:L], :]
            v_cand = v[:, :, cand_indices[:L], :]

            q_exp = q.unsqueeze(3)
            scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~cand_mask[:L].unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
            gates_ctx = self.w_ih(ctx)

            s = x_norm
            for _ in range(T):
                gates_h = self.w_hh(s)
                gates = gates_ctx + gates_h
                r, z, n_c = gates.chunk(3, dim=-1)
                r = torch.sigmoid(r)
                z = torch.sigmoid(z)
                c_in = gates_ctx.chunk(3, dim=-1)[2] + (self.w_hh(r * s)).chunk(3, dim=-1)[2]
                n = torch.tanh(c_in)
                s = (1.0 - z) * n + z * s

            return self.out_proj(s)

    def create_mlp():
        return nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=False),
            nn.GELU(),
            nn.Linear(d_mlp, d_model, bias=False)
        )

    # -------------------------------------------------------------------------
    # Models to Compare
    # -------------------------------------------------------------------------
    
    # 1. Standard 1-Layer SubQ
    class Model_1Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.surfer = SubQSurfer()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = create_mlp()
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            x = self.tok_emb(idx)
            x = x + self.surfer(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    # 2. 2-Layer Independent SubQ (2 Surfers, 2 MLPs)
    class Model_2Layer_Independent(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.surfer1 = SubQSurfer()
            self.surfer2 = SubQSurfer()
            self.ln1_1 = nn.LayerNorm(d_model)
            self.ln2_1 = nn.LayerNorm(d_model)
            self.ln1_2 = nn.LayerNorm(d_model)
            self.ln2_2 = nn.LayerNorm(d_model)
            self.mlp1 = create_mlp()
            self.mlp2 = create_mlp()
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            x = self.tok_emb(idx)
            # Layer 1
            x = x + self.surfer1(self.ln1_1(x))
            x = x + self.mlp1(self.ln2_1(x))
            # Layer 2
            x = x + self.surfer2(self.ln1_2(x))
            x = x + self.mlp2(self.ln2_2(x))
            return self.head(self.ln_f(x))

    # 3. 2-Layer Shared-Surfer SubQ (1 Shared Surfer, 2 Unique MLPs)
    class Model_2Layer_SharedSurfer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.shared_surfer = SubQSurfer()  # Shared attention core!
            self.ln1_1 = nn.LayerNorm(d_model)
            self.ln2_1 = nn.LayerNorm(d_model)
            self.ln1_2 = nn.LayerNorm(d_model)
            self.ln2_2 = nn.LayerNorm(d_model)
            self.mlp1 = create_mlp()  # Unique MLP 1
            self.mlp2 = create_mlp()  # Unique MLP 2
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            x = self.tok_emb(idx)
            # Block 1 (Shared Surfer + MLP 1)
            x = x + self.shared_surfer(self.ln1_1(x))
            x = x + self.mlp1(self.ln2_1(x))
            # Block 2 (Shared Surfer + MLP 2)
            x = x + self.shared_surfer(self.ln1_2(x))
            x = x + self.mlp2(self.ln2_2(x))
            return self.head(self.ln_f(x))

    # 4. 4-Layer Shared-Surfer SubQ (1 Shared Surfer, 4 Unique MLPs)
    class Model_4Layer_SharedSurfer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.shared_surfer = SubQSurfer()  # Single shared attention core across all 4 layers!
            self.lns1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
            self.lns2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
            self.mlps = nn.ModuleList([create_mlp() for _ in range(4)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            x = self.tok_emb(idx)
            for l in range(4):
                x = x + self.shared_surfer(self.lns1[l](x))
                x = x + self.mlps[l](self.lns2[l](x))
            return self.head(self.ln_f(x))

    configs = [
        ("1-Layer SubQ Baseline", Model_1Layer),
        ("2-Layer Independent SubQ (2 Surfers + 2 MLPs)", Model_2Layer_Independent),
        ("2-Layer Shared-Surfer SubQ (1 Shared Surfer + 2 MLPs)", Model_2Layer_SharedSurfer),
        ("4-Layer Shared-Surfer SubQ (1 Shared Surfer + 4 MLPs)", Model_4Layer_SharedSurfer),
    ]

    results = []

    for name, model_cls in configs:
        model = model_cls().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

        print(f"\n---> Training {name} ({n_params:,} params) for {num_steps} steps...")
        t0 = time.time()
        for step in range(1, num_steps + 1):
            model.train()
            xb, yb = get_batch('train')
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(30):
                xb_v, yb_v = get_batch('val')
                v_loss = F.cross_entropy(model(xb_v).view(-1, vocab_size), yb_v.view(-1))
                val_losses.append(v_loss.item())

        mean_val_loss = sum(val_losses) / len(val_losses)
        val_ppl = math.exp(mean_val_loss)
        elapsed = time.time() - t0

        print(f"     => {name} | Params: {n_params:,} | Val Loss: {mean_val_loss:.4f} | Perplexity: {val_ppl:.2f} | Time: {elapsed:.1f}s")
        results.append({
            "name": name,
            "params": n_params,
            "val_loss": mean_val_loss,
            "val_ppl": val_ppl,
            "time_s": elapsed
        })

    print("\n" + "=" * 85)
    print("  SHARED ATTENTION WITH UNIQUE MLPS: HEAD-TO-HEAD COMPARISON")
    print("=" * 85)
    print(f"{'Model Architecture':<52} | {'Params':<10} | {'Val Loss':<10} | {'Perplexity':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<52} | {r['params']:<10,d} | {r['val_loss']:<10.4f} | {r['val_ppl']:<10.2f}")
    print("=" * 85)

    return results


@app.local_entrypoint()
def main():
    print("Launching Shared Attention Benchmark on Modal GPU...")
    res = run_shared_attention_benchmark.remote()
    print("\nBenchmark completed!")
