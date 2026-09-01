import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("subq-twin-associative-memory", image=image)

@app.function(gpu="T4", timeout=600)
def run_associative_twin_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY 3/3: SUBQ FORWARD-BACKWARD TWIN ON ASSOCIATIVE KEY-VALUE RETRIEVAL")
    print("=" * 95)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    vocab_size = 128
    d_model = 128
    n_pairs = 16 # 16 Key-Value pairs per context episode

    # 1. Episode Sampler: Key-Value Dictionary Lookup with Distractors
    def sample_kv_episodes(batch_size, n_pairs=16, n_queries=8):
        # Sample unique keys and values from vocab
        keys = torch.randint(1, vocab_size // 2, (batch_size, n_pairs), device=device)
        values = torch.randint(vocab_size // 2, vocab_size, (batch_size, n_pairs), device=device)

        # Construct support sequence: [K1, V1, K2, V2, ..., K_N, V_N]
        support_seq = torch.stack([keys, values], dim=-1).view(batch_size, 2 * n_pairs)

        # Pick random query keys from the stored pairs
        query_indices = torch.randint(0, n_pairs, (batch_size, n_queries), device=device)
        query_keys = torch.gather(keys, 1, query_indices)
        query_targets = torch.gather(values, 1, query_indices)

        return support_seq, query_keys, query_targets

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

    # 3. Associative Twin Architecture
    class AssociativeTwinSystem(nn.Module):
        def __init__(self, vocab_size=128, d_model=128, n_heads=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model

            self.emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

            # Main Forward SubQ Memory Settler
            self.fwd_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.readout_head = nn.Linear(d_model, vocab_size, bias=False)

            # Backward Error SubQ Twin
            self.err_in_proj = nn.Linear(vocab_size, d_model)
            self.err_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.meta_lr = nn.Parameter(torch.tensor(0.2))

        def retrieve(self, q_keys, memory_bank, custom_head=None):
            # q_keys: [B, N_q], memory_bank: [B, 2*N_pairs, D]
            q_emb = self.emb(q_keys) # [B, N_q, D]
            
            # Query cross-attends to settled memory bank
            scores = torch.bmm(q_emb, memory_bank.transpose(1, 2)) / math.sqrt(self.d_model)
            attn = F.softmax(scores, dim=-1)
            retrieved = torch.bmm(attn, memory_bank) # [B, N_q, D]

            if custom_head is None:
                logits = self.readout_head(retrieved)
            else:
                logits = torch.bmm(retrieved, custom_head)
            return logits

        def self_teach_and_evaluate(self, support_seq, q_keys_supp, q_targets_supp, q_keys_test, T_fwd=4, T_err=4):
            B, L = support_seq.shape
            
            # 1. Forward SubQ settles key-value associative bindings
            h_in = self.emb(support_seq) + self.pos_emb[:, :L]
            memory_bank = self.fwd_subq(h_in, T=T_fwd) # [B, L, D]

            # 2. Test initial retrieval on support queries
            logits_supp = self.retrieve(q_keys_supp, memory_bank)
            prob_supp = F.softmax(logits_supp, dim=-1)
            y_onehot = F.one_hot(q_targets_supp, num_classes=self.vocab_size).float()
            error_field = y_onehot - prob_supp # [B, N_q_supp, vocab_size]

            # 3. Error Twin Multi-Hop Settling on retrieval errors
            h_err_in = self.err_in_proj(error_field)
            h_err = self.err_subq(h_err_in, T=T_err) # [B, N_q_supp, D]

            # 4. Bilinear Co-Settling Outer Product Delta
            q_emb_supp = self.emb(q_keys_supp) # [B, N_q_supp, D]
            delta_head = torch.bmm(q_emb_supp.transpose(1, 2), error_field) / q_keys_supp.shape[1]
            delta_twin = torch.bmm(h_err.transpose(1, 2), error_field) / q_keys_supp.shape[1]

            base_head = self.readout_head.weight.t().unsqueeze(0).expand(B, self.d_model, self.vocab_size)
            adapted_head = base_head + self.meta_lr * (delta_head + delta_twin)

            # 5. Evaluate on Held-Out Test Queries
            logits_test = self.retrieve(q_keys_test, memory_bank, custom_head=adapted_head)
            return logits_test

    total_steps = 600
    batch_size = 32
    print(f"Meta-Training Associative Twin ({total_steps} Steps)...")

    twin_model = AssociativeTwinSystem(vocab_size=vocab_size, d_model=128, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(twin_model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        twin_model.train()
        supp_seq, q_k, q_t = sample_kv_episodes(batch_size, n_pairs=16, n_queries=8)
        q_k_supp, q_k_test = q_k[:, :4], q_k[:, 4:]
        q_t_supp, q_t_test = q_t[:, :4], q_t[:, 4:]

        logits_test = twin_model.self_teach_and_evaluate(supp_seq, q_k_supp, q_t_supp, q_k_test, T_fwd=4, T_err=4)
        loss = F.cross_entropy(logits_test.reshape(-1, vocab_size), q_t_test.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(twin_model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == total_steps:
            acc = (logits_test.argmax(dim=-1) == q_t_test).float().mean().item() * 100.0
            print(f"  Step {step:>4}/{total_steps} | Retrieval Loss: {loss.item():.4f} | Retrieval Top-1 Acc: {acc:.2f}%")
    print(f"Associative Twin training completed in {time.time() - t0:.1f}s\n")

    # Evaluation on 200 Episodes
    twin_model.eval()
    eval_episodes = 200

    print("=" * 95)
    print("  FINAL EVALUATION: ASSOCIATIVE MEMORY RETRIEVAL ACCURACY (200 EPISODES)")
    print("=" * 95)
    print(f"{'Method / Configuration':<36} | {'Retrieval Top-1 Accuracy':<28} | {'Status'}")
    print("-" * 95)

    with torch.no_grad():
        supp_seq, q_k, q_t = sample_kv_episodes(eval_episodes, n_pairs=16, n_queries=8)
        q_k_supp, q_k_test = q_k[:, :4], q_k[:, 4:]
        q_t_supp, q_t_test = q_t[:, :4], q_t[:, 4:]

        # 1. Static Unadapted Retrieval
        h_in = twin_model.emb(supp_seq) + twin_model.pos_emb[:, :supp_seq.shape[1]]
        mem = twin_model.fwd_subq(h_in, T=4)
        logits_static = twin_model.retrieve(q_k_test, mem)
        acc_static = (logits_static.argmax(dim=-1) == q_t_test).float().mean().item() * 100.0
        print(f"{'Static Model (No Self-Teaching)':<36} | {acc_static:>24.2f}% | Baseline")

        # 2. SubQ Twin across Error Hops
        for t_err in [1, 2, 4, 6]:
            logits_twin = twin_model.self_teach_and_evaluate(supp_seq, q_k_supp, q_t_supp, q_k_test, T_fwd=4, T_err=t_err)
            acc_twin = (logits_twin.argmax(dim=-1) == q_t_test).float().mean().item() * 100.0
            print(f"{f'SubQ Twin (T_err = {t_err} Hops)':<36} | {acc_twin:>24.2f}% | Evaluated")

    print("=" * 95)

@app.local_entrypoint()
def main():
    run_associative_twin_study.remote()
