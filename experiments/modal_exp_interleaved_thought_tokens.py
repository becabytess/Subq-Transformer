import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests"
    )
)

app = modal.App("subq-interleaved-thought-tokens", image=image)

@app.function(gpu="T4", timeout=600)
def run_interleaved_benchmark():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  STUDY: INTERLEAVED THOUGHT TOKENS (Inserting Feedback Before Each Token)")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Setup (Character TinyShakespeare)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_text = response.read().decode('utf-8')
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_ix[c] for c in raw_text], dtype=torch.long)
    print(f"Corpus: {len(data):,} characters | Vocabulary: {vocab_size} unique chars")

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

    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # -------------------------------------------------------------------------
    # 1. Baseline Model: Standard SubQ (Lateral Hops)
    # -------------------------------------------------------------------------
    class BaselineSubQLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            self.ln1 = nn.LayerNorm(d_model)
            
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, T=4):
            B, L = idx.shape
            x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
            x_norm = self.ln1(x)

            pos = torch.arange(L, device=device)
            jump_indices = []
            valid_masks = []
            for j in fib_jumps:
                ij = pos - j
                m = ij >= 0
                ij = torch.clamp(ij, min=0)
                jump_indices.append(ij)
                valid_masks.append(m)
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

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)
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

            x_settled = x + self.out_proj(s)
            x_out = x_settled + self.mlp(self.ln2(x_settled))
            return self.head(self.norm(x_out))

    # -------------------------------------------------------------------------
    # 2. Interleaved Thought-Token Model
    # -------------------------------------------------------------------------
    class InterleavedThoughtSubQLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            
            # Initial learned thought token embedding
            self.init_thought = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.init_thought, std=0.02)

            self.ln1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)

            # Thought projector: transforms decision hypothesis into preceding thought token
            self.w_thought_proj = nn.Linear(d_model, d_model, bias=False)

            # Separate projection for attending to preceding thought token
            self.thought_attn_k = nn.Linear(d_model, d_model, bias=False)
            self.thought_attn_v = nn.Linear(d_model, d_model, bias=False)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, T=4):
            B, L = idx.shape
            x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
            x_norm = self.ln1(x)

            pos = torch.arange(L, device=device)
            jump_indices = []
            valid_masks = []
            for j in fib_jumps:
                ij = pos - j
                m = ij >= 0
                ij = torch.clamp(ij, min=0)
                jump_indices.append(ij)
                valid_masks.append(m)
            jump_indices = torch.stack(jump_indices, dim=-1)
            valid_masks = torch.stack(valid_masks, dim=-1)

            # Base jump context across the sequence
            q = self.q_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(B, L, n_heads, head_dim).transpose(1, 2)

            k_cand = k[:, :, jump_indices, :]
            v_cand = v[:, :, jump_indices, :]

            scores = (q.unsqueeze(3) * k_cand).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~valid_masks.unsqueeze(0).unsqueeze(0), -1e9)
            pi = F.softmax(scores, dim=-1)
            ctx_jumps = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, d_model)

            # Initial thought token placed right before each position
            thought_toks = self.init_thought.expand(B, L, d_model)

            s = x_norm
            for t in range(T):
                # 1. Thought token at position i provides direct contextual input right before x_i
                # Compute combined context: Jump context + Preceding Thought Token context
                k_th = self.thought_attn_k(thought_toks) # [B, L, D]
                v_th = self.thought_attn_v(thought_toks) # [B, L, D]

                # Attention weight from current state to its preceding thought token
                q_s = self.q_proj(s) # [B, L, D]
                th_score = (q_s * k_th).sum(dim=-1, keepdim=True) / math.sqrt(d_model)
                th_gate = torch.sigmoid(th_score) # how much to attend to preceding thought token
                
                # Context augmented by preceding thought token
                ctx_augmented = ctx_jumps + th_gate * v_th
                gates_ctx = self.w_ih(ctx_augmented)
                r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

                # 2. Recurrent Hop
                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)
                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s

                # 3. Update the preceding thought token for the next iteration based on what was predicted
                x_t = x + self.out_proj(s)
                x_t_out = x_t + self.mlp(self.ln2(x_t))
                decision_feature = self.norm(x_t_out)
                thought_toks = self.w_thought_proj(decision_feature) # Updates thought token before each x_i

            x_settled = x + self.out_proj(s)
            x_out = x_settled + self.mlp(self.ln2(x_settled))
            return self.head(self.norm(x_out))

    total_steps = 800

    # 1. Train Baseline
    print("\n[1/2] Training Baseline SubQ (Lateral Hops)...")
    m_base = BaselineSubQLM().to(device)
    opt_b = torch.optim.AdamW(m_base.parameters(), lr=2e-3, weight_decay=1e-2)
    sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=total_steps, eta_min=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        m_base.train()
        x, y = get_batch("train")
        logits = m_base(x, T=4)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt_b.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m_base.parameters(), 1.0)
        opt_b.step()
        sch_b.step()
        if step % 400 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Loss: {loss.item():.4f} | PPL: {math.exp(loss.item()):.2f}")
    print(f"Baseline training completed in {time.time() - t0:.1f}s")

    # 2. Train Interleaved Thought Model
    print("\n[2/2] Training Interleaved Thought-Token SubQ...")
    m_th = InterleavedThoughtSubQLM().to(device)
    opt_th = torch.optim.AdamW(m_th.parameters(), lr=2e-3, weight_decay=1e-2)
    sch_th = torch.optim.lr_scheduler.CosineAnnealingLR(opt_th, T_max=total_steps, eta_min=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        m_th.train()
        x, y = get_batch("train")
        logits = m_th(x, T=4)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt_th.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m_th.parameters(), 1.0)
        opt_th.step()
        sch_th.step()
        if step % 400 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Loss: {loss.item():.4f} | PPL: {math.exp(loss.item()):.2f}")
    print(f"Interleaved Thought-Token training completed in {time.time() - t0:.1f}s")

    # 3. Validation Evaluation across Hops
    m_base.eval()
    m_th.eval()
    eval_hops = [1, 2, 3, 4, 6]
    n_eval = 50

    base_res = {t: 0.0 for t in eval_hops}
    th_res = {t: 0.0 for t in eval_hops}

    with torch.no_grad():
        for _ in range(n_eval):
            x, y = get_batch("val")
            for t in eval_hops:
                base_res[t] += F.cross_entropy(m_base(x, T=t).view(-1, vocab_size), y.view(-1)).item() / n_eval
                th_res[t] += F.cross_entropy(m_th(x, T=t).view(-1, vocab_size), y.view(-1)).item() / n_eval

    print("\n" + "=" * 90)
    print("  FINAL EVALUATION: BASELINE SUBQ VS INTERLEAVED THOUGHT-TOKEN SUBQ")
    print("=" * 90)
    print(f"{'Thought Hops (T)':<18} | {'Baseline Val Loss':<18} | {'Baseline PPL':<14} | {'Interleaved Val Loss':<22} | {'Interleaved PPL':<16} | {'PPL Delta'}")
    print("-" * 115)

    for t in eval_hops:
        loss_b = base_res[t]
        ppl_b = math.exp(loss_b)
        loss_th = th_res[t]
        ppl_th = math.exp(loss_th)
        delta = ppl_th - ppl_b
        winner = "🏆 (Interleaved wins)" if delta < 0 else "— (Baseline wins)"
        print(f"T = {t:<14} | {loss_b:<18.4f} | {ppl_b:<14.2f} | {loss_th:<22.4f} | {ppl_th:<16.2f} | {delta:>+6.2f} PPL {winner}")
    print("=" * 115)

@app.local_entrypoint()
def main():
    run_interleaved_benchmark.remote()
