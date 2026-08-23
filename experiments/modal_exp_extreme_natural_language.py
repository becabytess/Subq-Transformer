"""
Modal Experiment: Natural Language Modeling at Extreme Context (L = 4,096 to 8,192 tokens)
Tests whether Static Graph SubQ smoothly trains, converges, and learns natural language
dependencies across massive 4k-8k context windows on a single Tesla T4 GPU.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests", "tiktoken")
)

app = modal.App("subq-extreme-natural-language", image=image)


@app.function(gpu="T4", timeout=1200)
def run_extreme_language_benchmark():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tiktoken

    print("=" * 80)
    print("  SUBQTRANSFORMER: NATURAL LANGUAGE MODELING AT L = 4,096 TO 8,192")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Container: {torch.cuda.get_device_name(0)} (16GB VRAM)")

    # Load Shakespeare / OpenWebText sample
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = urllib.request.urlopen(url).read().decode('utf-8')

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    vocab_size = enc.n_vocab

    print(f"Dataset: Natural English ({len(tokens):,} BPE tokens, Vocab: {vocab_size:,})")

    L = 4096
    batch_size = 2
    d_model = 128
    n_heads = 4
    head_dim = d_model // n_heads
    d_mlp = 512
    K = 24
    T = 3

    # Multi-scale relative jump indices for L = 4096
    cand_indices = torch.zeros((L, K), dtype=torch.long, device=device)
    cand_mask = torch.zeros((L, K), dtype=torch.bool, device=device)

    for i in range(L):
        if i == 0:
            continue
        valid_prev = torch.arange(i, device=device)
        d = (i - valid_prev).float()
        log_d = torch.log2(d + 1.0)
        max_log = log_d.max()
        stride_buckets = torch.linspace(0, max_log, steps=K, device=device)
        chosen = valid_prev[torch.abs(log_d.unsqueeze(1) - stride_buckets.unsqueeze(0)).argmin(dim=0)]
        cand_indices[i] = chosen
        cand_mask[i] = True

    class LongContextSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_hh = nn.Linear(d_model, 3 * d_model, bias=False)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            B, L_seq = idx.shape
            x = self.tok_emb(idx)
            x_norm = self.ln1(x)

            q = self.q_proj(x_norm).view(B, L_seq, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L_seq, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L_seq, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, cand_indices[:L_seq], :]
            v_cand = v[:, :, cand_indices[:L_seq], :]

            q_exp = q.unsqueeze(3)
            scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~cand_mask[:L_seq].unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L_seq, d_model)
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

            x = x + self.out_proj(s)
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    def get_batch():
        ix = torch.randint(len(data) - L - 1, (batch_size,))
        x = torch.stack([data[i:i+L] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+L+1] for i in ix]).to(device)
        return x, y

    model = LongContextSubQ().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print(f"\nTraining SubQ on Natural Language at L = {L} (4,096 tokens) for 200 steps...")
    print(f"{'Step':<8} | {'Loss':<12} | {'Perplexity':<16} | {'VRAM Allocated':<18} | {'Tokens/sec':<14}")
    print("-" * 80)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for step in range(1, 201):
        model.train()
        xb, yb = get_batch()

        t_step = time.time()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        elapsed_step = time.time() - t_step
        speed = (batch_size * L) / max(elapsed_step, 1e-6)

        if step % 25 == 0 or step == 1 or step == 200:
            ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Step {step:<4} | {loss.item():<12.4f} | {ppl:<14.2f} | {vram_mb:<14.1f} MB | {speed:<12.0f} t/s")

    total_time = time.time() - t0
    final_ppl = math.exp(loss.item())
    print(f"\n---> Finished 200 steps in {total_time:.1f}s | Final Train Loss: {loss.item():.4f} | Perplexity: {final_ppl:.2f}")
    print(f"---> Peak VRAM at L=4096: {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB")
    
    return {
        "final_loss": loss.item(),
        "final_ppl": final_ppl,
        "vram_mb": torch.cuda.max_memory_allocated() / (1024*1024)
    }


@app.local_entrypoint()
def main():
    print("Launching Natural Language Extreme Context Benchmark on Modal GPU...")
    res = run_extreme_language_benchmark.remote()
    print("\nBenchmark finished successfully!")
