import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests"
    )
)

app = modal.App("subq-twin-continual-self-training", image=image)

@app.function(gpu="T4", timeout=900)
def run_continual_training_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY: CONTINUAL ZERO-BACKPROP SELF-TRAINING VIA FORWARD-BACKWARD TWIN")
    print("  Phase 1: Train on 20% of data (Learn to Teach)")
    print("  Phase 2: Continual Self-Training on remaining 80% of data (Zero Backpropagation)")
    print("=" * 95)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Download TinyShakespeare
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data_tensor = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long, device=device)

    # 20% for Phase 1, 80% for Continual Online Training Phase 2
    n_phase1 = int(len(data_tensor) * 0.20)
    phase1_data = data_tensor[:n_phase1]
    phase2_stream = data_tensor[n_phase1:]
    
    print(f"TinyShakespeare: {len(text):,} chars | Vocab = {vocab_size}")
    print(f"Phase 1 Meta-Training Data: {len(phase1_data):,} chars (20%)")
    print(f"Phase 2 Continual Streaming Data: {len(phase2_stream):,} chars (80%)\n")

    # 2. SubQ Core Block
    class SubQBlock(nn.Module):
        def __init__(self, d_model=128, n_heads=4):
            super().__init__()
            self.d_model = d_model
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)

        def forward(self, h_init, T=4):
            s = h_init
            for _ in range(T):
                Q = self.q_proj(s)
                K = self.k_proj(s)
                V = self.v_proj(s)

                scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
                attn = F.softmax(scores, dim=-1)
                context = self.out_proj(torch.bmm(attn, V))

                gates_ih = self.w_ih(context)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))

                s = (1.0 - z) * n + z * s
            return s

    # 3. Language Twin System
    class LanguageTwinSystem(nn.Module):
        def __init__(self, vocab_size=65, d_model=128, n_heads=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model

            # Main Forward Model
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
            self.fwd_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Backward Error Twin
            self.err_proj = nn.Linear(vocab_size, d_model)
            self.err_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.meta_lr = nn.Parameter(torch.tensor(0.05))

        def forward_pass(self, x, T=4, custom_lm_head=None):
            B, L = x.shape
            h = self.token_emb(x) + self.pos_emb[:, :L]
            h_fwd = self.fwd_subq(h, T=T)

            if custom_lm_head is None:
                logits = self.lm_head(h_fwd)
            else:
                # custom_lm_head: [d_model, vocab_size]
                logits = F.linear(h_fwd, custom_lm_head.t())
            return logits, h_fwd

        def compute_twin_delta(self, x, y, T_fwd=4, T_err=4, current_head=None):
            B, L = x.shape
            logits, h_fwd = self.forward_pass(x, T=T_fwd, custom_lm_head=current_head)

            prob = F.softmax(logits, dim=-1)
            y_onehot = F.one_hot(y, num_classes=self.vocab_size).float()
            error_field = y_onehot - prob # [B, L, vocab_size]

            h_err_in = self.err_proj(error_field)
            h_err = self.err_subq(h_err_in, T=T_err)

            # Average bilinear delta across batch and sequence length
            # [d_model, vocab_size]
            delta_fwd = torch.einsum('bld,blv->dv', h_fwd, error_field) / (B * L)
            delta_err = torch.einsum('bld,blv->dv', h_err, error_field) / (B * L)
            total_delta = delta_fwd + delta_err
            return total_delta, logits

    # 4. Standard Model (Pure Supervised Baseline)
    class StandardLM(nn.Module):
        def __init__(self, vocab_size=65, d_model=128, n_heads=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
            self.fwd_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x, T=4):
            B, L = x.shape
            h = self.token_emb(x) + self.pos_emb[:, :L]
            h_fwd = self.fwd_subq(h, T=T)
            return self.lm_head(h_fwd)

    # -------------------------------------------------------------------------
    # Phase 1: Train on First 20% of Data (500 Steps)
    # -------------------------------------------------------------------------
    seq_len = 64
    batch_size = 32
    phase1_steps = 500

    def sample_batch(data_source, bsz, slen):
        max_idx = len(data_source) - (slen + 1)
        starts = torch.randint(0, max_idx, (bsz,))
        x_list, y_list = [], []
        for s in starts:
            x_list.append(data_source[s : s + slen])
            y_list.append(data_source[s + 1 : s + slen + 1])
        return torch.stack(x_list), torch.stack(y_list)

    print("Phase 1: Training Models on 20% Data...")

    # Train Standard Baseline on 20%
    standard_model = StandardLM(vocab_size, d_model=128, n_heads=4).to(device)
    opt_std = torch.optim.AdamW(standard_model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, phase1_steps + 1):
        standard_model.train()
        bx, by = sample_batch(phase1_data, batch_size, seq_len)
        logits = standard_model(bx, T=4)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))

        opt_std.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(standard_model.parameters(), 1.0)
        opt_std.step()

        if step % 250 == 0 or step == phase1_steps:
            acc = (logits.argmax(dim=-1) == by).float().mean().item() * 100.0
            print(f"  [Standard Model] Step {step:>4}/{phase1_steps} | Loss: {loss.item():.4f} | Top-1 Acc: {acc:.2f}%")

    # Train Twin System on 20%
    twin_system = LanguageTwinSystem(vocab_size, d_model=128, n_heads=4).to(device)
    opt_twin = torch.optim.AdamW(twin_system.parameters(), lr=1e-3, weight_decay=1e-4)

    for step in range(1, phase1_steps + 1):
        twin_system.train()
        bx, by = sample_batch(phase1_data, batch_size, seq_len)
        delta, logits = twin_system.compute_twin_delta(bx, by, T_fwd=4, T_err=4)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))

        opt_twin.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(twin_system.parameters(), 1.0)
        opt_twin.step()

        if step % 250 == 0 or step == phase1_steps:
            acc = (logits.argmax(dim=-1) == by).float().mean().item() * 100.0
            print(f"  [Twin System]    Step {step:>4}/{phase1_steps} | Loss: {loss.item():.4f} | Top-1 Acc: {acc:.2f}%")

    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # Phase 2: Continual Online Self-Training on Remaining 80% Stream (Zero Backprop!)
    # -------------------------------------------------------------------------
    print("=" * 95)
    print("  PHASE 2: CONTINUAL STREAMING TRAINING ON 80% UNSEEN TEXT (ZERO BACKPROP)")
    print("=" * 95)
    print(f"{'Stream Progress':<20} | {'Static (20% Only) Acc':<22} | {'Online SGD (Backprop) Acc':<26} | {'SubQ Twin (Zero Backprop) Acc':<30}")
    print("-" * 105)

    # 1. Static model (fixed, no further updates)
    standard_model.eval()

    # 2. Online SGD model (initialized from standard model, updates via real backprop)
    sgd_online_model = StandardLM(vocab_size, d_model=128, n_heads=4).to(device)
    sgd_online_model.load_state_dict(standard_model.state_dict())
    opt_online_sgd = torch.optim.SGD(sgd_online_model.parameters(), lr=0.01)

    # 3. SubQ Twin (freeze Error Twin, only accumulate weights via forward-only Twin updates)
    twin_system.eval()
    accumulated_twin_head = twin_system.lm_head.weight.t().clone() # [d_model, vocab_size]
    twin_lr = twin_system.meta_lr.item()

    # Stream chunks of text across 10 evaluation milestones
    n_stream_steps = 200
    eval_interval = 40

    chunk_size = len(phase2_stream) // n_stream_steps

    for step in range(1, n_stream_steps + 1):
        idx_start = (step - 1) * chunk_size
        idx_end = idx_start + chunk_size
        chunk_data = phase2_stream[idx_start:idx_end]

        bx, by = sample_batch(chunk_data, batch_size, seq_len)

        # 1. Update Online SGD Model via Backpropagation
        sgd_online_model.train()
        logits_sgd = sgd_online_model(bx, T=4)
        loss_sgd = F.cross_entropy(logits_sgd.view(-1, vocab_size), by.view(-1))
        opt_online_sgd.zero_grad()
        loss_sgd.backward()
        opt_online_sgd.step()

        # 2. Update SubQ Twin via Zero-Backprop Forward Error Relaxation
        with torch.no_grad():
            twin_delta, _ = twin_system.compute_twin_delta(bx, by, T_fwd=4, T_err=4, current_head=accumulated_twin_head)
            accumulated_twin_head = accumulated_twin_head + twin_lr * twin_delta

        # Periodic Evaluation on held-out chunk
        if step % eval_interval == 0 or step == n_stream_steps:
            pct_stream = (step / n_stream_steps) * 100
            
            # Evaluate on fresh test batch
            test_x, test_y = sample_batch(chunk_data, 64, seq_len)
            with torch.no_grad():
                # Static
                acc_static = (standard_model(test_x, T=4).argmax(dim=-1) == test_y).float().mean().item() * 100.0
                # Online SGD
                acc_sgd = (sgd_online_model(test_x, T=4).argmax(dim=-1) == test_y).float().mean().item() * 100.0
                # SubQ Twin
                logits_twin, _ = twin_system.forward_pass(test_x, T=4, custom_lm_head=accumulated_twin_head)
                acc_twin = (logits_twin.argmax(dim=-1) == test_y).float().mean().item() * 100.0

            print(f"{pct_stream:>5.1f}% Streamed ({step:>3}/{n_stream_steps}) | {acc_static:>18.2f}% | {acc_sgd:>22.2f}% | {acc_twin:>26.2f}%")

    print("=" * 105)

@app.local_entrypoint()
def main():
    run_continual_training_study.remote()
