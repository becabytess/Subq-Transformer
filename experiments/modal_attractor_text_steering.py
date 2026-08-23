"""
Modal Experiment: Qualitative Text Generation Steering via Attractor Biasing
Demonstrates that injecting a directional bias vector into SubQ's recurrent state
steers the model's generated text towards a target semantic topic/style at runtime
without any gradient updates or weight fine-tuning.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests", "tiktoken")
)

app = modal.App("subq-attractor-text-steering", image=image)


@app.function(gpu="T4", timeout=1200)
def run_text_steering_experiment():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tiktoken

    print("=" * 80)
    print("  SUBQTRANSFORMER: RUNTIME TEXT GENERATION STEERING EXPERIMENT")
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

    block_size = 128
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512
    K = 16

    def get_batch():
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix])
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    class SubQSurfer(nn.Module):
        def __init__(self, d_model, n_heads, K):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.K = K

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru_cell = nn.GRUCell(d_model, d_model)

        def forward(self, x, T=4, custom_bias=None):
            B, L, D = x.shape
            device = x.device

            q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

            pos = torch.arange(L, device=device)
            cand_indices = []
            cand_mask = []
            for i in range(L):
                if i == 0:
                    cand_indices.append(torch.zeros(self.K, dtype=torch.long, device=device))
                    cand_mask.append(torch.zeros(self.K, dtype=torch.bool, device=device))
                    continue
                valid_prev = torch.arange(i, device=device)
                d = i - valid_prev
                log_d = torch.log2(d.float() + 1.0)
                max_log = log_d.max()
                stride_buckets = torch.linspace(0, max_log, steps=self.K, device=device)
                chosen = valid_prev[torch.abs(log_d.unsqueeze(1) - stride_buckets.unsqueeze(0)).argmin(dim=0)]
                cand_indices.append(chosen)
                cand_mask.append(torch.ones(self.K, dtype=torch.bool, device=device))

            cand_indices = torch.stack(cand_indices, dim=0)
            cand_mask = torch.stack(cand_mask, dim=0)

            k_cand = k[:, :, cand_indices, :]
            v_cand = v[:, :, cand_indices, :]

            q_exp = q.unsqueeze(3)
            scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~cand_mask.unsqueeze(0).unsqueeze(0), -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            ctx_h = (attn_weights.unsqueeze(-1) * v_cand).sum(dim=3)

            ctx = ctx_h.transpose(1, 2).contiguous().view(B, L, D)
            
            if custom_bias is not None:
                ctx = ctx + custom_bias

            ctx_flat = ctx.view(B * L, D)
            s_t = torch.zeros(B * L, D, device=device)

            for _ in range(T):
                s_t = self.gru_cell(ctx_flat, s_t)

            out = self.out_proj(s_t.view(B, L, D))
            return out

    class SubQLanguageModel(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, d_mlp, block_size, K):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = SubQSurfer(d_model, n_heads, K=K)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, T=4, custom_bias=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s_out = self.surfer(self.ln1(x), T=T, custom_bias=custom_bias)
            x = x + s_out
            x = x + self.mlp(self.ln2(x))
            logits = self.head(self.ln_f(x))
            return logits

    print("\n---> Training SubQ Model on Shakespeare (750 steps)...")
    model = SubQLanguageModel(vocab_size, d_model, n_heads, d_mlp, block_size, K=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, 751):
        model.train()
        xb, yb = get_batch()
        logits = model(xb, T=4)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Training finished in {time.time()-t0:.1f}s | Final Loss: {loss.item():.4f}")
    model.eval()

    # Construct two distinct semantic concept steering directions:
    # 1. Royalty / Nobility Direction: ["king", "queen", "crown", "lord"]
    # 2. War / Conflict Direction: ["sword", "battle", "blood", "kill"]
    
    def get_concept_vector(words):
        vecs = []
        for w in words:
            tok = enc.encode(" " + w)[0]
            vecs.append(model.tok_emb.weight[tok].detach())
        return torch.stack(vecs).mean(dim=0)

    vec_royalty = get_concept_vector(["king", "queen", "crown", "lord", "noble"])
    vec_war = get_concept_vector(["sword", "battle", "blood", "kill", "death"])

    steering_royalty = vec_royalty / vec_royalty.norm()
    steering_war = vec_war / vec_war.norm()

    prompt_text = "The messenger arrived at the gates and declared"
    prompt_ids = torch.tensor([enc.encode(prompt_text)], dtype=torch.long, device=device)

    def generate_steered(prompt, bias_vec=None, alpha=0.0, max_tokens=30, temp=0.8):
        current_ids = prompt.clone()
        bias_tensor = (bias_vec * alpha).unsqueeze(0).unsqueeze(0) if bias_vec is not None else None
        
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model(current_ids[:, -block_size:], T=4, custom_bias=bias_tensor)
                logits = logits[:, -1, :] / temp
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat((current_ids, next_tok), dim=1)
        
        return enc.decode(current_ids[0].tolist())

    print("\n" + "=" * 80)
    print("  QUALITATIVE GENERATION UNDER ATTRACTOR STEERING BIAS")
    print("=" * 80)
    print(f"Prompt: \"{prompt_text}\"\n")

    print("[1. Unsteered Baseline (alpha = 0.0)]:")
    for i in range(2):
        print(f"  Sample {i+1}: {generate_steered(prompt_ids, alpha=0.0)}")

    print("\n[2. Steered with Royalty / Nobility Vector (alpha = +2.0)]:")
    for i in range(2):
        print(f"  Sample {i+1}: {generate_steered(prompt_ids, bias_vec=steering_royalty, alpha=2.0)}")

    print("\n[3. Steered with War / Conflict Vector (alpha = +2.0)]:")
    for i in range(2):
        print(f"  Sample {i+1}: {generate_steered(prompt_ids, bias_vec=steering_war, alpha=2.0)}")

    print("\n" + "=" * 80)
    print("Generation steering complete!")
