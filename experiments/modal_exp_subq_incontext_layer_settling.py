import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("subq-incontext-layer-settling", image=image)

@app.function(gpu="T4", timeout=600)
def run_incontext_settling_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY: IN-CONTEXT LAYER SETTLING VIA MULTI-HOP SUBQ ATTENTION")
    print("  Sequence Layout: [INPUTS] [SEP] [LAYER TOKENS] [SEP] [OUTPUTS]")
    print("  No Handcrafted Error Injection — Pure Multi-Hop Contraction Settling (T Hops)")
    print("=" * 95)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # 1. Task Generator: Non-linear Function Tasks
    # -------------------------------------------------------------------------
    def sample_tasks(batch_size, n_support, n_query):
        amplitudes = torch.rand(batch_size, 1, 1, device=device) * 4.0 + 1.0  # [1.0, 5.0]
        phases = torch.rand(batch_size, 1, 1, device=device) * math.pi        # [0, pi]
        freqs = torch.rand(batch_size, 1, 1, device=device) * 1.5 + 0.5       # [0.5, 2.0]

        x_all = (torch.rand(batch_size, n_support + n_query, 1, device=device) - 0.5) * 6.0 # [-3, 3]
        y_all = amplitudes * torch.sin(freqs * x_all + phases)

        x_s, x_q = x_all[:, :n_support], x_all[:, n_support:]
        y_s, y_q = y_all[:, :n_support], y_all[:, n_support:]
        return x_s, y_s, x_q, y_q

    # -------------------------------------------------------------------------
    # 2. SubQ In-Context Token-Layer Architecture
    # -------------------------------------------------------------------------
    class SubQInContextSettler(nn.Module):
        def __init__(self, d_model=128, n_heads=4, n_layer_tokens=8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layer_tokens = n_layer_tokens

            # Input and Output Projectors into d_model embedding space
            self.input_proj = nn.Linear(1, d_model)
            self.output_proj = nn.Linear(1, d_model)

            # Special Tokens: [SEP_1], [SEP_2]
            self.sep1 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.sep2 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

            # Initial Layer Tokens in the middle (trainable base state)
            self.init_layer_tokens = nn.Parameter(torch.randn(1, n_layer_tokens, d_model) * 0.02)

            # SubQ Multi-Head Attention components
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            # SubQ Contraction GRU Gates
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)

            # Downstream Prediction Head: Uses Query Input x_q + Settled Layer Tokens -> y_q
            self.query_encoder = nn.Linear(1, d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.readout = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            )

        def settle_layers(self, x_s, y_s, T=4):
            # x_s: [B, K, 1], y_s: [B, K, 1]
            B, K, _ = x_s.shape

            # 1. Project inputs and outputs to d_model
            h_inputs = self.input_proj(x_s)   # [B, K, d_model]
            h_outputs = self.output_proj(y_s) # [B, K, d_model]

            sep1_tok = self.sep1.expand(B, 1, self.d_model)
            sep2_tok = self.sep2.expand(B, 1, self.d_model)
            h_layers = self.init_layer_tokens.expand(B, self.n_layer_tokens, self.d_model)

            # 2. Assemble Full Sequence: [INPUTS] [SEP1] [LAYERS] [SEP2] [OUTPUTS]
            # Total sequence length = K + 1 + n_layer_tokens + 1 + K
            seq = torch.cat([h_inputs, sep1_tok, h_layers, sep2_tok, h_outputs], dim=1) # [B, L_seq, d_model]

            # Indices of the layer tokens in the sequence
            idx_start = K + 1
            idx_end = idx_start + self.n_layer_tokens

            delta_norms = []

            # 3. SubQ Multi-Hop Settling Loop across T hops
            for hop in range(T):
                # Attention across the sequence
                Q = self.q_proj(seq)
                K_mat = self.k_proj(seq)
                V = self.v_proj(seq)

                # Scaled dot-product attention
                attn_scores = torch.bmm(Q, K_mat.transpose(1, 2)) / math.sqrt(self.d_model)
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_out = torch.bmm(attn_weights, V)
                n_raw = self.out_proj(attn_out)

                # SubQ Gated Contraction Hop on sequence state
                gates_ih = self.w_ih(n_raw)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(seq)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * seq))

                new_seq = (1.0 - z) * n + z * seq

                # Track settlement delta on layer tokens
                delta_layer = torch.norm(new_seq[:, idx_start:idx_end] - seq[:, idx_start:idx_end], dim=-1).mean().item()
                delta_norms.append(delta_layer)

                seq = new_seq

            # Extract settled layer tokens
            settled_layers = seq[:, idx_start:idx_end] # [B, n_layer_tokens, d_model]
            return settled_layers, delta_norms

        def predict_query(self, x_q, settled_layers):
            # x_q: [B, N_q, 1], settled_layers: [B, n_layer_tokens, d_model]
            q_emb = self.query_encoder(x_q) # [B, N_q, d_model]

            # Query attends to the settled layer tokens to extract adapted function knowledge
            attn_out, _ = self.cross_attn(query=q_emb, key=settled_layers, value=settled_layers)
            y_pred = self.readout(attn_out + q_emb) # [B, N_q, 1]
            return y_pred

        def forward(self, x_s, y_s, x_q, T=4):
            settled_layers, delta_norms = self.settle_layers(x_s, y_s, T=T)
            y_pred = self.predict_query(x_q, settled_layers)
            return y_pred, delta_norms

    # -------------------------------------------------------------------------
    # 3. Training Meta-Learner (800 Steps)
    # -------------------------------------------------------------------------
    batch_size = 64
    n_support = 10
    n_query = 15
    train_T = 4  # Trained with T = 4 settling hops
    total_steps = 1000

    print(f"Meta-Training SubQ In-Context Settler (T = {train_T} Hops, Support K = {n_support})...")
    model = SubQInContextSettler(d_model=128, n_heads=4, n_layer_tokens=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        x_s, y_s, x_q, y_q = sample_tasks(batch_size, n_support, n_query)

        y_pred, _ = model(x_s, y_s, x_q, T=train_T)
        loss = F.mse_loss(y_pred, y_q)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 250 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Query MSE: {loss.item():.4f}")

    print(f"Training completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # 4. Evaluation: Test Query MSE as a function of Settling Depth T (T = 0 to 10)
    # -------------------------------------------------------------------------
    model.eval()
    eval_tasks = 200
    n_support_eval = 10

    print("=" * 95)
    print("  EVALUATION: QUERY MSE & LAYER SETTLING DELTA AS A FUNCTION OF THOUGHT HOPS (T)")
    print("=" * 95)
    print(f"{'Settling Hops (T)':<20} | {'Query MSE':<18} | {'Layer Settlement Delta ||ΔL||':<32} | {'Behavior'}")
    print("-" * 95)

    with torch.no_grad():
        # Generate evaluation batch
        x_s, y_s, x_q, y_q = sample_tasks(eval_tasks, n_support_eval, n_query)

        # T = 0 (No settling, raw initial layer tokens)
        init_layers = model.init_layer_tokens.expand(eval_tasks, model.n_layer_tokens, model.d_model)
        y_pred_0 = model.predict_query(x_q, init_layers)
        mse_0 = F.mse_loss(y_pred_0, y_q).item()
        print(f"T = 0 (Unadapted)    | {mse_0:<18.4f} | {'N/A':<32} | Baseline (No In-Context adaptation)")

        for test_T in [1, 2, 3, 4, 6, 8, 10]:
            settled_layers, delta_norms = model.settle_layers(x_s, y_s, T=test_T)
            y_pred = model.predict_query(x_q, settled_layers)
            mse = F.mse_loss(y_pred, y_q).item()
            last_delta = delta_norms[-1] if len(delta_norms) > 0 else 0.0

            status = "🏆 Optimal Settling" if mse == min(mse, mse_0) else ("Converged" if last_delta < 0.05 else "Settling...")
            print(f"T = {test_T:<16} | {mse:<18.4f} | {last_delta:<32.4f} | {status}")

    print("=" * 95)

@app.local_entrypoint()
def main():
    run_incontext_settling_study.remote()
