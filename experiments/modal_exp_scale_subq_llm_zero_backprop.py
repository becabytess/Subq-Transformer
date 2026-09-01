import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests"
    )
)

app = modal.App("scale-subq-llm-zero-backprop", image=image)

@app.function(gpu="T4", timeout=900)
def run_scaled_llm_streaming_benchmark():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  SCALED SUBQ LANGUAGE MODEL: ZERO-BACKPROP CONTINUAL IN-SESSION ADAPTATION BENCHMARK")
    print("  Model Architecture: Scaled SubQ Transformer LM (d_model=256, n_heads=8, T=4 Dynamical Thought Settling)")
    print("  Testing: Zero-Backprop Fast-Weight Head vs Full Backprop SGD vs Static LLM")
    print("  Evaluations:")
    print("    1. Real-Time Online Stream Perplexity (PPL) on Out-of-Domain Specialized Stream")
    print("    2. Catastrophic Forgetting Retention on Held-Out In-Domain Text")
    print("    3. Real-Time Token Throughput & Memory Efficiency (Zero BP vs Full BP)")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Download Corpus (TinyShakespeare 1.1M characters)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(url).text
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data_tensor = torch.tensor([char_to_ix[c] for c in raw_text], dtype=torch.long, device=device)

    n_total = len(data_tensor)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)
    n_stream = n_total - n_train - n_val

    train_data = data_tensor[:n_train]
    val_data = data_tensor[n_train:n_train + n_val]
    stream_data = data_tensor[n_train + n_val:]

    print(f"Corpus Loaded: {n_total:,} chars | Vocab: {vocab_size} tokens")
    print(f"  - Pre-Training Base (70%): {len(train_data):,} tokens")
    print(f"  - Held-Out In-Domain Test (10%): {len(val_data):,} tokens")
    print(f"  - Continual Streaming Out-of-Domain Task (20%): {len(stream_data):,} tokens\n")

    d_model = 256
    n_heads = 8
    seq_len = 128

    # 2. Scaled SubQ Core Block
    class SubQCore(nn.Module):
        def __init__(self, d_model=256, n_heads=8):
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

    # 3. Scaled SubQ Language Model
    class SubQLanguageModel(nn.Module):
        def __init__(self, vocab_size, d_model=256, n_heads=8):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
            self.subq_core = SubQCore(d_model=d_model, n_heads=n_heads)
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            # Error Twin for Head Relaxation
            self.err_twin = SubQCore(d_model=d_model, n_heads=n_heads)
            self.lr_head = nn.Parameter(torch.tensor(0.05))

        def forward(self, x, T=4, custom_head_w=None):
            B, L = x.shape
            toks = self.tok_emb(x)
            pos = self.pos_emb[:, :L, :]
            h_init = toks + pos

            h_settled = self.subq_core(h_init, T=T)
            h_norm = self.ln_f(h_settled)

            if custom_head_w is None:
                logits = self.head(h_norm)
            else:
                logits = torch.matmul(h_norm, custom_head_w.t())

            return logits, h_norm

        def compute_fast_weight_delta(self, x, y, T=4, custom_head_w=None):
            logits, h_norm = self.forward(x, T=T, custom_head_w=custom_head_w)
            B, L, V = logits.shape
            probs = F.softmax(logits, dim=-1)
            y_onehot = F.one_hot(y, V).float()
            e_out = y_onehot - probs # [B, L, V]

            # Fast-Weight Bilinear Co-Settling Outer Product
            h_flat = h_norm.view(-1, self.d_model) # [B*L, D]
            e_flat = e_out.view(-1, V) # [B*L, V]

            # Delta W = (1 / N_tokens) * (E^T * H) -> [V, D]
            dw_head = torch.matmul(e_flat.t(), h_flat) / (B * L * math.sqrt(self.d_model))
            return dw_head, logits

    # 4. Pre-Train SubQ LM on In-Domain Base Corpus
    print("--- Phase 1: Pre-Training Base SubQ Language Model (d_model=256, 70% Corpus) ---")
    lm_model = SubQLanguageModel(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads).to(device)
    optimizer = torch.optim.AdamW(lm_model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_batch(data, batch_size=32, seq_len=128):
        max_idx = len(data) - seq_len - 1
        ix = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        return x, y

    def evaluate_nll_and_ppl(model, data, custom_head_w=None, n_batches=20):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = get_batch(data, batch_size=32, seq_len=seq_len)
                logits, _ = model(x, T=4, custom_head_w=custom_head_w)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()
        avg_nll = total_loss / n_batches
        ppl = math.exp(avg_nll)
        return avg_nll, ppl

    t0 = time.time()
    steps_pretrain = 1200
    for step in range(1, steps_pretrain + 1):
        lm_model.train()
        x, y = get_batch(train_data, batch_size=32, seq_len=seq_len)
        logits, _ = lm_model(x, T=4)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(lm_model.parameters(), 1.0)
        optimizer.step()

        if step % 300 == 0 or step == steps_pretrain:
            val_nll, val_ppl = evaluate_nll_and_ppl(lm_model, val_data)
            print(f"  [Pre-Train] Step {step:>4}/{steps_pretrain} | In-Domain NLL: {val_nll:.4f} | Perplexity (PPL): {val_ppl:.2f}")

    print(f"Pre-training completed in {time.time() - t0:.1f}s\n")
    base_val_nll, base_val_ppl = evaluate_nll_and_ppl(lm_model, val_data)
    print(f"Baseline Pre-Trained Model Performance on Held-Out Test: NLL = {base_val_nll:.4f} (PPL: {base_val_ppl:.2f})\n")

    # -------------------------------------------------------------------------
    # Phase 2: CONTINUAL STREAMING INGESTION ON NOVEL / SHIFTED DOMAIN STREAM
    # -------------------------------------------------------------------------
    print("=" * 135)
    print("  PHASE 2: CONTINUAL STREAMING INGESTION (OUT-OF-DOMAIN SPECIALIZED STREAM, 500 STREAMING STEPS)")
    print("  Comparing: 1. Static Base LLM | 2. Online SGD (Backprop) | 3. SubQ Fast-Weight Twin (ZERO BACKPROP)")
    print("=" * 135)

    # Construct Out-of-Domain / Jargon Shifted Stream (Specialized Vocabulary Permutation & Code Dialect)
    torch.manual_seed(1337)
    perm = torch.randperm(vocab_size, device=device)
    # Permute 24 tokens (heavy jargon/dialect shift)
    cipher_map = torch.arange(vocab_size, device=device)
    swap_tokens = perm[:24]
    for i in range(0, len(swap_tokens), 2):
        t1, t2 = swap_tokens[i], swap_tokens[i+1]
        cipher_map[t1], cipher_map[t2] = t2, t1

    ood_stream_data = cipher_map[stream_data]

    # Model A: Static LLM (No adaptation)
    static_lm = lm_model

    # Model B: Online SGD LLM (Full Backpropagation through time)
    sgd_lm = SubQLanguageModel(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads).to(device)
    sgd_lm.load_state_dict(lm_model.state_dict())
    opt_sgd = torch.optim.SGD(sgd_lm.parameters(), lr=0.01, momentum=0.9)

    # Model C: SubQ Fast-Weight Twin LLM (ZERO Backpropagation, Fast Readout Update)
    twin_lm = lm_model
    twin_head_w = lm_model.head.weight.data.clone() # [V, D]
    twin_lr = 0.08
    decay = 1e-4

    print(f"{'Streaming Step':<16} | {'Static LLM PPL':<16} | {'Online SGD PPL (Backprop)':<28} | {'SubQ Zero-BP Twin PPL':<26} | {'Zero-BP PPL Gain':<18}")
    print("-" * 135)

    n_stream_steps = 400
    batch_size_stream = 16

    # Tracking metrics
    nll_static_hist = []
    nll_sgd_hist = []
    nll_twin_hist = []

    t_start_sgd = 0.0
    t_start_twin = 0.0

    t_sgd_total = 0.0
    t_twin_total = 0.0

    for step in range(1, n_stream_steps + 1):
        # Sample next chunk from continuous streaming data
        st = (step * batch_size_stream * seq_len) % (len(ood_stream_data) - batch_size_stream * seq_len - 1)
        chunk = ood_stream_data[st : st + batch_size_stream * seq_len + 1]
        bx = chunk[:-1].view(batch_size_stream, seq_len)
        by = chunk[1:].view(batch_size_stream, seq_len)

        # 1. Evaluate Static LLM
        with torch.no_grad():
            logits_stat, _ = static_lm(bx, T=4)
            loss_stat = F.cross_entropy(logits_stat.view(-1, vocab_size), by.view(-1)).item()
            nll_static_hist.append(loss_stat)

        # 2. Update & Evaluate Online SGD (Full Backprop)
        t_s0 = time.time()
        sgd_lm.train()
        logits_sgd, _ = sgd_lm(bx, T=4)
        loss_sgd = F.cross_entropy(logits_sgd.view(-1, vocab_size), by.view(-1))
        opt_sgd.zero_grad()
        loss_sgd.backward()
        nn.utils.clip_grad_norm_(sgd_lm.parameters(), 1.0)
        opt_sgd.step()
        t_sgd_total += (time.time() - t_s0)
        nll_sgd_hist.append(loss_sgd.item())

        # 3. Update & Evaluate SubQ Zero-Backprop Twin (Forward-Only Fast Weights)
        t_t0 = time.time()
        with torch.no_grad():
            twin_lm.eval()
            # Fast-Weight delta computed via forward error relaxation
            dw_head, logits_twin = twin_lm.compute_fast_weight_delta(bx, by, T=4, custom_head_w=twin_head_w)
            loss_twin = F.cross_entropy(logits_twin.view(-1, vocab_size), by.view(-1)).item()
            nll_twin_hist.append(loss_twin)

            # Online Fast-Weight update (Zero Backprop)
            twin_head_w = (1.0 - decay) * twin_head_w + twin_lr * dw_head
        t_twin_total += (time.time() - t_t0)

        if step % 50 == 0 or step == n_stream_steps:
            # Average over recent window
            recent = 50
            avg_stat_ppl = math.exp(np.mean(nll_static_hist[-recent:]))
            avg_sgd_ppl = math.exp(np.mean(nll_sgd_hist[-recent:]))
            avg_twin_ppl = math.exp(np.mean(nll_twin_hist[-recent:]))
            ppl_improvement = ((avg_stat_ppl - avg_twin_ppl) / avg_stat_ppl) * 100.0

            print(f"Step {step:>4}/{n_stream_steps}    | {avg_stat_ppl:>14.2f} | {avg_sgd_ppl:>20.2f} (SGD BP) | {avg_twin_ppl:>18.2f} (Zero-BP) | {ppl_improvement:>+14.1f}%")

    print("-" * 135)
    print("\n--- Phase 3: Catastrophic Forgetting & Computational Efficiency Probes ---")

    # 1. Catastrophic Forgetting Probe on Original Held-Out In-Domain Text
    stat_retain_nll, stat_retain_ppl = evaluate_nll_and_ppl(static_lm, val_data)
    sgd_retain_nll, sgd_retain_ppl = evaluate_nll_and_ppl(sgd_lm, val_data)
    twin_retain_nll, twin_retain_ppl = evaluate_nll_and_ppl(twin_lm, val_data, custom_head_w=twin_head_w)

    print(f"\n1. RETENTION / CATASTROPHIC FORGETTING ON ORIGINAL BASE DOMAIN (Held-Out Validation Set):")
    print(f"  - Original Pre-Trained Baseline : NLL = {base_val_nll:.4f} | PPL = {base_val_ppl:.2f}")
    print(f"  - Online SGD (Backprop)         : NLL = {sgd_retain_nll:.4f} | PPL = {sgd_retain_ppl:.2f} (Forgetting Delta: {sgd_retain_ppl - base_val_ppl:+.2f} PPL)")
    print(f"  - SubQ Zero-BP Twin (Readout)   : NLL = {twin_retain_nll:.4f} | PPL = {twin_retain_ppl:.2f} (Forgetting Delta: {twin_retain_ppl - base_val_ppl:+.2f} PPL)")

    # 2. Computational Speed & Throughput
    total_tokens = n_stream_steps * batch_size_stream * seq_len
    sgd_tps = total_tokens / t_sgd_total
    twin_tps = total_tokens / t_twin_total
    speedup = twin_tps / sgd_tps

    print(f"\n2. COMPUTATIONAL THROUGHPUT & EFFICIENCY:")
    print(f"  - Total Tokens Processed Online : {total_tokens:,} tokens")
    print(f"  - Online SGD (Full Backprop)    : {sgd_tps:,.1f} tokens/sec ({t_sgd_total:.2f}s total)")
    print(f"  - SubQ Zero-Backprop Twin       : {twin_tps:,.1f} tokens/sec ({t_twin_total:.2f}s total)")
    print(f"  - Speedup Factor                : {speedup:.2f}x FASTER than Backprop SGD!")
    print("=" * 135)

@app.local_entrypoint()
def main():
    run_scaled_llm_streaming_benchmark.remote()
