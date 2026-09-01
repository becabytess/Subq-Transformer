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

app = modal.App("subq-bpe-fixed-t3", image=image)

@app.function(gpu="T4", timeout=900)
def run_fixed_t3():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import tiktoken

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  EXPERIMENT 1: FIXED UNIFORM T=3 BASELINE (BPE TOKENS)")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_text = response.read().decode('utf-8')
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    vocab_size = enc.n_vocab

    print(f"Corpus: {len(tokens):,} BPE tokens | Vocab: {vocab_size:,}")

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

    start = time.time()
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

        if step % 200 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Loss: {loss.item():.4f} | PPL: {math.exp(loss.item()):.2f}")

    print(f"\nTraining completed in {time.time() - start:.1f}s")
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
    print("=" * 80)
    print(f"  FINAL RESULT: Fixed T=3 -> Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
    print("=" * 80)

@app.local_entrypoint()
def main():
    run_fixed_t3.remote()
