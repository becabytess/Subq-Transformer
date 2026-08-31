import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "numpy",
        "requests",
        "accelerate"
    )
)

app = modal.App("gpt2-subq-zero-backprop-trainer", image=image)

@app.function(gpu="T4", timeout=900)
def run_gpt2_subq_adaptation_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  EXPERIMENT: ZERO-BACKPROP SUBQ TRAINER FOR PRE-TRAINED GPT-2 (124M)")
    print("  Setup:")
    print("    - Main Model: Real Standard Pre-Trained GPT-2 (124M params, d_model=768, Vocab=50,257)")
    print("    - GPT-2 Backbone: 100% FROZEN (Standard Dense Attention, No Backprop through GPT-2)")
    print("    - Trainer Model: SubQ Dynamical Error Twin (d_model=768, T=4 Hops)")
    print("    - Task: Continual Online Domain Adaptation on Specialized Text with ZERO BACKPROP")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Pre-Trained GPT-2 and Tokenizer
    print("Loading Pre-Trained GPT-2 (124M) from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_model.eval()

    # Freeze ALL GPT-2 weights permanently
    for param in gpt2_model.parameters():
        param.requires_grad = False

    vocab_size = gpt2_model.config.vocab_size # 50,257
    d_model = gpt2_model.config.n_embd # 768
    print(f"GPT-2 Loaded: {vocab_size:,} vocab | {d_model} hidden dimension | Backbone 100% Frozen\n")

    # 2. Download Real Specialized Domain Dataset (Shakespearean Drama Corpus)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(url).text
    tokens = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    n_total = len(data_tensor)
    n_train = int(n_total * 0.60) # 60% Phase 1 (Trainer Meta-Training)
    n_test = n_total - n_train    # 40% Phase 2 (Zero-Backprop Streaming Adaptation)

    train_data = data_tensor[:n_train]
    stream_data = data_tensor[n_train:]

    print(f"Domain Dataset Tokenized: {n_total:,} tokens")
    print(f"  - Phase 1 (Trainer Training 60%): {len(train_data):,} tokens")
    print(f"  - Phase 2 (Online Zero-BP Stream 40%): {len(stream_data):,} tokens\n")

    # 3. SubQ Error Trainer Architecture
    class SubQCore(nn.Module):
        def __init__(self, d_model=768, n_heads=12):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.ln = nn.LayerNorm(d_model)

        def forward(self, h_init, T=4):
            B, L, D = h_init.shape
            s = h_init
            for _ in range(T):
                s_norm = self.ln(s)
                Q = self.q_proj(s_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                K = self.k_proj(s_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                V = self.v_proj(s_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
                mask = torch.tril(torch.ones(L, L, device=s.device)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(mask == 0, float("-inf"))
                attn = F.softmax(scores, dim=-1)

                context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
                context = self.out_proj(context)

                gates_ih = self.w_ih(context)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))

                s = (1.0 - z) * n + z * s
            return s

    class SubQGPT2Trainer(nn.Module):
        def __init__(self, d_model=768, n_heads=12):
            super().__init__()
            self.d_model = d_model
            self.subq_engine = SubQCore(d_model=d_model, n_heads=n_heads)
            self.ln = nn.LayerNorm(d_model)
            self.lr_head = nn.Parameter(torch.tensor(0.05))

        def compute_gpt2_fast_delta(self, gpt2_features, logits, targets, T=4):
            # gpt2_features: [B, L, 768]
            # logits: [B, L, 50257]
            # targets: [B, L]
            B, L, D = gpt2_features.shape
            V = logits.shape[-1]

            probs = F.softmax(logits, dim=-1)
            targets_onehot = F.one_hot(targets, V).float()
            error = targets_onehot - probs # [B, L, V]

            # Pass GPT-2 features through SubQ dynamical settling
            h_settled = self.subq_engine(gpt2_features, T=T)
            h_norm = self.ln(h_settled)

            # Bilinear Co-Settling outer product: Delta W = E^T * H -> [V, D]
            h_flat = h_norm.view(-1, D)
            e_flat = error.view(-1, V)
            dw_head = torch.matmul(e_flat.t(), h_flat) / (B * L * math.sqrt(D))
            return dw_head

    subq_trainer = SubQGPT2Trainer(d_model=d_model, n_heads=12).to(device)
    opt_trainer = torch.optim.AdamW(subq_trainer.parameters(), lr=1e-3, weight_decay=1e-4)

    seq_len = 64
    batch_size = 16

    def get_batch(data, batch_size=16, seq_len=64):
        max_idx = len(data) - seq_len - 1
        ix = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        return x, y

    def evaluate_gpt2_ppl(custom_head_w=None, n_eval_batches=20):
        gpt2_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(n_eval_batches):
                x, y = get_batch(stream_data, batch_size=batch_size, seq_len=seq_len)
                transformer_outputs = gpt2_model.transformer(x)
                h = transformer_outputs.last_hidden_state # [B, L, 768]

                if custom_head_w is None:
                    logits = gpt2_model.lm_head(h)
                else:
                    logits = torch.matmul(h, custom_head_w.t())

                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()
        avg_nll = total_loss / n_eval_batches
        return avg_nll, math.exp(avg_nll)

    # 4. Measure Pre-Trained Zero-Shot GPT-2 Perplexity on the Novel Domain
    print("Measuring Pre-Trained Zero-Shot GPT-2 Baseline Performance...")
    zero_shot_nll, zero_shot_ppl = evaluate_gpt2_ppl()
    print(f"  --> Pre-Trained Zero-Shot GPT-2 Perplexity: {zero_shot_ppl:.2f} (NLL: {zero_shot_nll:.4f})\n")

    # -------------------------------------------------------------------------
    # Phase 1: Meta-Training the SubQ Trainer on the First 60% of Domain Data
    # -------------------------------------------------------------------------
    print("--- Phase 1: Training SubQ Error Trainer on 60% Domain Data (1000 Steps) ---")
    t0 = time.time()
    steps_train = 600

    for step in range(1, steps_train + 1):
        subq_trainer.train()
        x_supp, y_supp = get_batch(train_data, batch_size=batch_size, seq_len=seq_len)
        x_query, y_query = get_batch(train_data, batch_size=batch_size, seq_len=seq_len)

        # 1. Forward through frozen GPT-2 on support batch
        with torch.no_grad():
            h_supp = gpt2_model.transformer(x_supp).last_hidden_state
            logits_supp = gpt2_model.lm_head(h_supp)

        # 2. SubQ Trainer generates delta for GPT-2 head
        base_head_w = gpt2_model.lm_head.weight # [V, D]
        dw = subq_trainer.compute_gpt2_fast_delta(h_supp, logits_supp, y_supp, T=4)
        adapted_head_w = base_head_w + subq_trainer.lr_head * dw

        # 3. Forward through frozen GPT-2 on query batch using adapted head
        with torch.no_grad():
            h_query = gpt2_model.transformer(x_query).last_hidden_state
        logits_query = torch.matmul(h_query, adapted_head_w.t())
        loss_meta = F.cross_entropy(logits_query.view(-1, vocab_size), y_query.view(-1))

        # 4. Update ONLY the SubQ Trainer
        opt_trainer.zero_grad()
        loss_meta.backward()
        nn.utils.clip_grad_norm_(subq_trainer.parameters(), 1.0)
        opt_trainer.step()

        if step % 200 == 0 or step == steps_train:
            print(f"  [Trainer Training] Step {step:>4}/{steps_train} | Meta-Query NLL: {loss_meta.item():.4f}")

    print(f"SubQ Trainer training completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # Phase 2: Continual Online Streaming Adaptation on Unseen 40% Stream (ZERO BACKPROP)
    # -------------------------------------------------------------------------
    print("=" * 135)
    print("  PHASE 2: ONLINE STREAMING ADAPTATION ON UNSEEN 40% STREAM (ZERO BACKPROP THROUGH GPT-2)")
    print("  SubQ Trainer Is FROZEN. Testing Continuous Zero-Backprop Fine-Tuning of GPT-2.")
    print("=" * 135)

    subq_trainer.eval()
    gpt2_adapted_head_w = gpt2_model.lm_head.weight.clone() # Start from pre-trained GPT-2 head
    stream_lr = 0.08
    decay = 1e-4

    n_stream_steps = 300
    nll_static_hist = []
    nll_adapted_hist = []

    print(f"{'Streaming Step':<16} | {'Static GPT-2 PPL':<18} | {'SubQ Zero-BP GPT-2 PPL':<28} | {'Perplexity Reduction':<20}")
    print("-" * 135)

    for step in range(1, n_stream_steps + 1):
        # Sample next incoming stream batch
        bx, by = get_batch(stream_data, batch_size=batch_size, seq_len=seq_len)

        with torch.no_grad():
            # 1. Evaluate Static GPT-2
            h_stat = gpt2_model.transformer(bx).last_hidden_state
            logits_stat = gpt2_model.lm_head(h_stat)
            loss_stat = F.cross_entropy(logits_stat.view(-1, vocab_size), by.view(-1)).item()
            nll_static_hist.append(loss_stat)

            # 2. Evaluate & Adapt GPT-2 using SubQ Trainer (ZERO BACKPROP)
            logits_curr = torch.matmul(h_stat, gpt2_adapted_head_w.t())
            loss_curr = F.cross_entropy(logits_curr.view(-1, vocab_size), by.view(-1)).item()
            nll_adapted_hist.append(loss_curr)

            # SubQ Trainer generates fast-weight delta
            dw = subq_trainer.compute_gpt2_fast_delta(h_stat, logits_curr, by, T=4)

            # Update GPT-2 head online with zero backprop
            gpt2_adapted_head_w = (1.0 - decay) * gpt2_adapted_head_w + stream_lr * dw

        if step % 50 == 0 or step == n_stream_steps:
            recent = 50
            stat_ppl = math.exp(np.mean(nll_static_hist[-recent:]))
            adapt_ppl = math.exp(np.mean(nll_adapted_hist[-recent:]))
            ppl_drop = ((stat_ppl - adapt_ppl) / stat_ppl) * 100.0

            print(f"Step {step:>4}/{n_stream_steps}    | {stat_ppl:>16.2f} | {adapt_ppl:>20.2f} (Zero-BP) | {ppl_drop:>+16.1f}%")

    print("-" * 135)
    final_stat_ppl = math.exp(np.mean(nll_static_hist[-100:]))
    final_adapt_ppl = math.exp(np.mean(nll_adapted_hist[-100:]))
    print(f"\nFINAL CONTINUAL STREAMING RESULTS ON FROZEN GPT-2:")
    print(f"  - Pre-Trained Static GPT-2 Stream Perplexity : {final_stat_ppl:.2f}")
    print(f"  - SubQ Zero-Backprop Adapted GPT-2 Perplexity : {final_adapt_ppl:.2f}")
    print(f"  - Total Perplexity Reduction                : {((final_stat_ppl - final_adapt_ppl) / final_stat_ppl) * 100.0:.2f}% GAIN (ZERO BACKPROP)")
    print("=" * 135)

@app.local_entrypoint()
def main():
    run_gpt2_subq_adaptation_study.remote()
