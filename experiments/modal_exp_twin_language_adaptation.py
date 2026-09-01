import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests"
    )
)

app = modal.App("subq-twin-language-adaptation", image=image)

@app.function(gpu="T4", timeout=600)
def run_language_twin_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY 1/3: SUBQ FORWARD-BACKWARD TWIN ON LANGUAGE ADAPTATION (CIPHER RECOVERY)")
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

    n_train = int(len(data_tensor) * 0.9)
    train_data = data_tensor[:n_train]
    test_data = data_tensor[n_train:]
    print(f"TinyShakespeare Loaded: {len(text):,} chars, Vocab Size = {vocab_size}")

    # 2. Episode Sampler: Text with dynamic random character cipher permutations
    def sample_cipher_episodes(data, batch_size, seq_len_support=64, seq_len_query=64, n_swaps=10):
        # Sample contiguous chunks
        max_idx = len(data) - (seq_len_support + seq_len_query + 1)
        starts = torch.randint(0, max_idx, (batch_size,))

        x_supp_list, y_supp_list = [], []
        x_query_list, y_query_list = [], []

        for b in range(batch_size):
            st = starts[b].item()
            chunk = data[st : st + seq_len_support + seq_len_query + 1]

            # Generate random character permutation cipher for this specific episode
            cipher = torch.arange(vocab_size, device=device)
            swap_pairs = torch.randperm(vocab_size)[: 2 * n_swaps]
            for i in range(0, len(swap_pairs), 2):
                c1, c2 = swap_pairs[i], swap_pairs[i+1]
                cipher[c1], cipher[c2] = c2, c1

            # Encrypt the text chunk
            encrypted_chunk = cipher[chunk]

            x_s = encrypted_chunk[:seq_len_support]
            y_s = encrypted_chunk[1 : seq_len_support + 1]
            x_q = encrypted_chunk[seq_len_support : seq_len_support + seq_len_query]
            y_q = encrypted_chunk[seq_len_support + 1 : seq_len_support + seq_len_query + 1]

            x_supp_list.append(x_s)
            y_supp_list.append(y_s)
            x_query_list.append(x_q)
            y_query_list.append(y_q)

        return (
            torch.stack(x_supp_list),
            torch.stack(y_supp_list),
            torch.stack(x_query_list),
            torch.stack(y_query_list)
        )

    # 3. SubQ Core Block
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

    # 4. Language Forward-Backward Twin
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
            self.meta_lr = nn.Parameter(torch.tensor(0.2))

        def forward_pass(self, x, T=4, custom_lm_head=None):
            B, L = x.shape
            h = self.token_emb(x) + self.pos_emb[:, :L]
            h_fwd = self.fwd_subq(h, T=T)

            if custom_lm_head is None:
                logits = self.lm_head(h_fwd)
            else:
                logits = torch.bmm(h_fwd, custom_lm_head) # [B, L, vocab_size]
            return logits, h_fwd

        def self_teach_and_evaluate(self, x_s, y_s, x_q, T_fwd=4, T_err=4):
            B, L_s = x_s.shape
            logits_s, h_fwd = self.forward_pass(x_s, T=T_fwd)

            # Categorical Cross-Entropy Error Field: OneHot(Y) - Softmax(Logits)
            prob_s = F.softmax(logits_s, dim=-1)
            y_onehot = F.one_hot(y_s, num_classes=self.vocab_size).float()
            error_field = y_onehot - prob_s # [B, L_s, vocab_size]

            # Error Twin processes the error field across T_err hops
            h_err_in = self.err_proj(error_field)
            h_err = self.err_subq(h_err_in, T=T_err) # [B, L_s, d_model]

            # Bilinear Outer Product: Update the LM head dynamically
            # [B, d_model, L_s] x [B, L_s, vocab_size] -> [B, d_model, vocab_size]
            delta_head = torch.bmm(h_fwd.transpose(1, 2), error_field) / L_s
            delta_guided = torch.bmm(h_err.transpose(1, 2), error_field) / L_s

            base_head = self.lm_head.weight.t().unsqueeze(0).expand(B, self.d_model, self.vocab_size)
            adapted_head = base_head + self.meta_lr * (delta_head + delta_guided)

            # Evaluate on Query Text
            logits_q, _ = self.forward_pass(x_q, T=T_fwd, custom_lm_head=adapted_head)
            return logits_q

    # 5. Training Loop
    batch_size = 32
    total_steps = 600
    print(f"Meta-Training Language Twin on Cipher Shifts ({total_steps} Steps)...")

    twin_model = LanguageTwinSystem(vocab_size=vocab_size, d_model=128, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(twin_model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        twin_model.train()
        x_s, y_s, x_q, y_q = sample_cipher_episodes(train_data, batch_size, seq_len_support=48, seq_len_query=48, n_swaps=8)

        logits_q = twin_model.self_teach_and_evaluate(x_s, y_s, x_q, T_fwd=4, T_err=4)
        loss = F.cross_entropy(logits_q.view(-1, vocab_size), y_q.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(twin_model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == total_steps:
            acc = (logits_q.argmax(dim=-1) == y_q).float().mean().item() * 100.0
            print(f"  Step {step:>4}/{total_steps} | Query Loss: {loss.item():.4f} | Char Top-1 Acc: {acc:.2f}%")
    print(f"Training completed in {time.time() - t0:.1f}s\n")

    # 6. Evaluation on Held-Out Test Text (200 Episodes)
    twin_model.eval()
    eval_episodes = 200

    print("=" * 95)
    print("  FINAL EVALUATION: LANGUAGE CIPHER ADAPTATION ON HELD-OUT TEST TEXT")
    print("=" * 95)
    print(f"{'Method / Configuration':<36} | {'Query Cross-Entropy':<22} | {'Char Top-1 Accuracy'}")
    print("-" * 95)

    with torch.no_grad():
        x_s, y_s, x_q, y_q = sample_cipher_episodes(test_data, eval_episodes, seq_len_support=48, seq_len_query=48, n_swaps=8)

        # 1. Static Model (No Adaptation)
        logits_static, _ = twin_model.forward_pass(x_q, T=4)
        loss_static = F.cross_entropy(logits_static.view(-1, vocab_size), y_q.view(-1)).item()
        acc_static = (logits_static.argmax(dim=-1) == y_q).float().mean().item() * 100.0
        print(f"{'Static Model (No Adaptation)':<36} | {loss_static:<22.4f} | {acc_static:>18.2f}%")

        # 2. SubQ Twin across Error Hops
        for t_err in [1, 2, 4, 6]:
            logits_twin = twin_model.self_teach_and_evaluate(x_s, y_s, x_q, T_fwd=4, T_err=t_err)
            loss_twin = F.cross_entropy(logits_twin.view(-1, vocab_size), y_q.view(-1)).item()
            acc_twin = (logits_twin.argmax(dim=-1) == y_q).float().mean().item() * 100.0
            tag = f"SubQ Twin (T_err = {t_err} Hops)"
            print(f"{tag:<36} | {loss_twin:<22.4f} | {acc_twin:>18.2f}%")

    print("=" * 95)

@app.local_entrypoint()
def main():
    run_language_twin_study.remote()
