"""
Modal Experiment: Dynamic Recomputing Routing vs Static Frozen Graph in Long Contexts
Investigates:
Does dynamic routing (recomputing attention query q^(t) and candidate selection at each hop)
allow SubQ to solve long-range needle retrieval across wide gaps where static graphs fail?
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests")
)

app = modal.App("subq-dynamic-vs-static-long-context", image=image)


@app.function(gpu="T4", timeout=1200)
def run_dynamic_vs_static_needle_test():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    print("=" * 80)
    print("  DYNAMIC RECOMPUTING VS STATIC GRAPH IN LONG CONTEXT RETRIEVAL")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Container: {torch.cuda.get_device_name(0)}")

    L = 2048
    vocab_size = 512
    num_kv = 8
    batch_size = 4
    K = 24
    T = 3
    d_model = 128
    n_heads = 4
    head_dim = d_model // n_heads

    # Precompute multi-scale logarithmic relative jumps
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

    # 1. Static Frozen Routing Model
    class StaticSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru = nn.GRUCell(d_model, d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            x = self.tok_emb(idx)

            q = self.q_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, cand_indices, :]
            v_cand = v[:, :, cand_indices, :]

            # Compute graph routing ONCE at hop 1
            q_exp = q.unsqueeze(3)
            scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~cand_mask.unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
            s = x

            # Recurrent hops with frozen static context
            ctx_flat = ctx.view(B * L, d_model)
            s_flat = s.view(B * L, d_model)
            for _ in range(T):
                s_flat = self.gru(ctx_flat, s_flat)

            s = s_flat.view(B, L, d_model)
            return self.head(x + self.out_proj(s))

    # 2. Dynamic Recomputing Routing Model
    class DynamicSubQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru = nn.GRUCell(d_model, d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            x = self.tok_emb(idx)

            k = self.k_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, cand_indices, :]
            v_cand = v[:, :, cand_indices, :]

            s = x
            s_flat = s.view(B * L, d_model)

            for _ in range(T):
                # Recompute Q from the updated recurrent state s at EVERY hop!
                q_t = self.q_proj(s).view(B, L, n_heads, head_dim).transpose(1, 2)
                q_exp = q_t.unsqueeze(3)
                scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(head_dim)
                scores = scores.masked_fill(~cand_mask.unsqueeze(0).unsqueeze(0), -1e9)
                pi_t = F.softmax(scores, dim=-1)

                ctx_t = (pi_t.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
                ctx_flat = ctx_t.view(B * L, d_model)
                s_flat = self.gru(ctx_flat, s_flat)
                s = s_flat.view(B, L, d_model)

            return self.head(x + self.out_proj(s))

    # Data Generator
    def get_batch():
        x = torch.randint(150, vocab_size, (batch_size, L), device=device)
        y = torch.full((batch_size, L), -100, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            keys = np.random.choice(range(1, 50), size=num_kv, replace=False)
            vals = np.random.choice(range(50, 100), size=num_kv, replace=False)
            
            # Scatter keys across the sequence
            insert_pos = sorted(np.random.choice(range(20, L - 100), size=num_kv, replace=False))
            for k_v, v_v, pos in zip(keys, vals, insert_pos):
                x[b, pos] = int(k_v)
                x[b, pos + 1] = int(v_v)
            
            # Query at end
            query_pos = sorted(np.random.choice(range(L - 50, L - 5), size=num_kv, replace=False))
            for k_v, v_v, q_pos in zip(keys, vals, query_pos):
                x[b, q_pos] = int(k_v)
                y[b, q_pos] = int(v_v)
                
        return x, y

    def train_and_eval(model, name, steps=250):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        print(f"\n---> Training {name} on L={L} for {steps} steps...")
        t0 = time.time()

        for step in range(1, steps + 1):
            model.train()
            xb, yb = get_batch()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    mask = (yb != -100)
                    preds = logits[mask].argmax(dim=-1)
                    targets = yb[mask]
                    acc = (preds == targets).float().mean().item() * 100.0
                    print(f"[{name}] Step {step:<4} | Loss: {loss.item():.4f} | Accuracy: {acc:6.2f}%")

        elapsed = time.time() - t0
        print(f"[{name}] Finished in {elapsed:.1f}s | Final Accuracy: {acc:.2f}%")
        return acc, elapsed

    print("\n--- Model 1: Static Frozen Graph (No per-hop recomputation) ---")
    static_model = StaticSubQ().to(device)
    acc_static, time_static = train_and_eval(static_model, "Static Frozen SubQ")

    print("\n--- Model 2: Dynamic Recomputed Routing (Query updated every hop) ---")
    dynamic_model = DynamicSubQ().to(device)
    acc_dynamic, time_dynamic = train_and_eval(dynamic_model, "Dynamic Recomputing SubQ")

    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD SUMMARY: STATIC VS DYNAMIC IN LONG CONTEXT (L = 2048)")
    print("=" * 80)
    print(f"Static Frozen Graph Accuracy:       {acc_static:6.2f}% | Time: {time_static:.1f}s")
    print(f"Dynamic Recomputing Graph Accuracy: {acc_dynamic:6.2f}% | Time: {time_dynamic:.1f}s")
    print("=" * 80)

    return {
        "acc_static": acc_static,
        "acc_dynamic": acc_dynamic
    }


@app.local_entrypoint()
def main():
    print("Launching Long-Context Dynamic vs Static Routing Benchmark on Modal GPU...")
    res = run_dynamic_vs_static_needle_test.remote()
    print("\nTest completed!")
