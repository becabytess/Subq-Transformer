import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("subq-forward-backward-twin", image=image)

@app.function(gpu="T4", timeout=600)
def run_twin_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("  STUDY: SUBQ FORWARD-BACKWARD TWIN ARCHITECTURE (Learned Credit Assignment)")
    print("  Main Model (Forward SubQ) <---> Twin Model (Error SubQ)")
    print("  Weight Update: ΔW = H_forward ⊗ H_error (Multi-Hop Co-Settling)")
    print("=" * 95)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # 1. Non-linear Task Family: Waveform & Sequence Function Adaptation
    # -------------------------------------------------------------------------
    def sample_task_batch(batch_size, seq_len_support, seq_len_query):
        amplitudes = torch.rand(batch_size, 1, 1, device=device) * 4.0 + 1.0
        phases = torch.rand(batch_size, 1, 1, device=device) * math.pi
        freqs = torch.rand(batch_size, 1, 1, device=device) * 1.5 + 0.5

        # Sample support and query sequences
        x_all = (torch.rand(batch_size, seq_len_support + seq_len_query, 1, device=device) - 0.5) * 6.0
        y_all = amplitudes * torch.sin(freqs * x_all + phases)

        x_supp = x_all[:, :seq_len_support]
        y_supp = y_all[:, :seq_len_support]
        x_query = x_all[:, seq_len_support:]
        y_query = y_all[:, seq_len_support:]
        return x_supp, y_supp, x_query, y_query

    # -------------------------------------------------------------------------
    # 2. SubQ Core Block (Used in both Forward and Error Twin Models)
    # -------------------------------------------------------------------------
    class SubQBlock(nn.Module):
        def __init__(self, d_model=64, n_heads=4):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)

        def forward(self, h_init, T=4):
            # h_init: [B, L, D]
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

    # -------------------------------------------------------------------------
    # 3. The Forward-Backward Twin System
    # -------------------------------------------------------------------------
    class SubQTwinSystem(nn.Module):
        def __init__(self, d_model=64, n_heads=4):
            super().__init__()
            self.d_model = d_model

            # 1. Main Model (Forward Network)
            self.fwd_in_proj = nn.Linear(1, d_model)
            self.fwd_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.fwd_head = nn.Linear(d_model, 1, bias=False)

            # 2. Twin Model (Error Network - Identical Capacity)
            self.err_in_proj = nn.Linear(1, d_model)
            self.err_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.err_head = nn.Linear(d_model, d_model, bias=False)

            # Learned Meta-Learning Rate for Hebbian self-update
            self.meta_lr = nn.Parameter(torch.tensor(0.1))

        def forward_pass(self, x, T=4, custom_head_weight=None):
            # x: [B, L, 1]
            h_in = self.fwd_in_proj(x)
            h_fwd = self.fwd_subq(h_in, T=T) # [B, L, D]
            
            if custom_head_weight is None:
                y_pred = self.fwd_head(h_fwd) # [B, L, 1]
            else:
                # custom_head_weight: [B, D, 1]
                y_pred = torch.bmm(h_fwd, custom_head_weight) # [B, L, 1]

            return y_pred, h_fwd

        def error_pass(self, error, T_err=4):
            # error: [B, L, 1]
            h_err_in = self.err_in_proj(error)
            h_err = self.err_subq(h_err_in, T=T_err) # [B, L, D]
            h_err_transformed = self.err_head(h_err) # [B, L, D]
            return h_err_transformed

        def self_teach_and_evaluate(self, x_supp, y_supp, x_query, T_fwd=4, T_err=4):
            B, L_s, _ = x_supp.shape

            # Step 1: Forward pass on support examples
            y_supp_pred, h_fwd = self.forward_pass(x_supp, T=T_fwd)

            # Step 2: Compute error signal
            error = y_supp_pred - y_supp # [B, L_s, 1]

            # Step 3: Error Twin processes the error tokens across T_err hops
            h_err = self.error_pass(error, T_err=T_err) # [B, L_s, D]

            # Step 4: Bilinear Hebbian Outer Product: H_fwd^T * H_err -> ΔW
            # [B, D, L_s] x [B, L_s, 1] -> [B, D, 1]
            # (Interacting settled forward thoughts with settled error thoughts)
            delta_W = torch.bmm(h_fwd.transpose(1, 2), error) / L_s # [B, D, 1]
            delta_W_gated = torch.bmm(h_fwd.transpose(1, 2), h_err.mean(dim=-1, keepdim=True)) / L_s

            # Combine direct Hebbian delta with the Error Model's settled guidance
            effective_delta = delta_W + delta_W_gated

            # Step 5: Update Main Model weights dynamically
            base_W = self.fwd_head.weight.t().unsqueeze(0).expand(B, self.d_model, 1) # [B, D, 1]
            adapted_W = base_W - self.meta_lr * effective_delta

            # Step 6: Evaluate adapted Main Model on Query data
            y_query_pred, _ = self.forward_pass(x_query, T=T_fwd, custom_head_weight=adapted_W)
            return y_query_pred

    # -------------------------------------------------------------------------
    # 4. Standard Backprop (SGD) Baseline
    # -------------------------------------------------------------------------
    class MAMLBaseline(nn.Module):
        def __init__(self, d_model=64, n_heads=4):
            super().__init__()
            self.d_model = d_model
            self.fwd_in_proj = nn.Linear(1, d_model)
            self.fwd_subq = SubQBlock(d_model=d_model, n_heads=n_heads)
            self.fwd_head = nn.Linear(d_model, 1, bias=False)
            self.inner_lr = nn.Parameter(torch.tensor(0.1))

        def adapt_and_evaluate(self, x_supp, y_supp, x_query, T_fwd=4):
            B = x_supp.shape[0]
            # Forward on support
            h_in = self.fwd_in_proj(x_supp)
            h_fwd = self.fwd_subq(h_in, T=T_fwd)
            y_pred = self.fwd_head(h_fwd)
            loss_supp = F.mse_loss(y_pred, y_supp)

            # Compute analytic gradient w.r.t head weights
            grad_w = torch.autograd.grad(loss_supp, self.fwd_head.weight, create_graph=True)[0]
            adapted_w = self.fwd_head.weight - self.inner_lr * grad_w

            # Evaluate on query
            h_q_in = self.fwd_in_proj(x_query)
            h_q_fwd = self.fwd_subq(h_q_in, T=T_fwd)
            y_q_pred = F.linear(h_q_fwd, adapted_w)
            return y_q_pred

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    batch_size = 64
    seq_len_supp = 10
    seq_len_query = 15
    total_steps = 1000

    print("Meta-Training: Twin Architecture vs MAML Baseline (1000 Steps)...\n")

    # 1. Train Twin System
    print("[1/2] Training SubQ Forward-Backward Twin...")
    twin_system = SubQTwinSystem(d_model=64, n_heads=4).to(device)
    opt_twin = torch.optim.AdamW(twin_system.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        twin_system.train()
        x_s, y_s, x_q, y_q = sample_task_batch(batch_size, seq_len_supp, seq_len_query)

        y_q_pred = twin_system.self_teach_and_evaluate(x_s, y_s, x_q, T_fwd=4, T_err=4)
        loss = F.mse_loss(y_q_pred, y_q)

        opt_twin.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(twin_system.parameters(), 1.0)
        opt_twin.step()

        if step % 250 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Query MSE: {loss.item():.4f}")
    print(f"Twin training completed in {time.time() - t0:.1f}s\n")

    # 2. Train MAML Baseline
    print("[2/2] Training MAML Backpropagation Baseline...")
    maml_model = MAMLBaseline(d_model=64, n_heads=4).to(device)
    opt_maml = torch.optim.AdamW(maml_model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        maml_model.train()
        x_s, y_s, x_q, y_q = sample_task_batch(batch_size, seq_len_supp, seq_len_query)

        y_q_pred = maml_model.adapt_and_evaluate(x_s, y_s, x_q, T_fwd=4)
        loss = F.mse_loss(y_q_pred, y_q)

        opt_maml.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(maml_model.parameters(), 1.0)
        opt_maml.step()

        if step % 250 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Query MSE: {loss.item():.4f}")
    print(f"MAML training completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # 6. Evaluation: Impact of Error-Model Thought Settling Depth (T_err = 0, 1, 2, 4, 8)
    # -------------------------------------------------------------------------
    twin_system.eval()
    maml_model.eval()
    eval_tasks = 300

    print("=" * 95)
    print("  EVALUATION: QUERY MSE ACROSS ERROR-TWIN THOUGHT DEPTH (T_err)")
    print("=" * 95)
    print(f"{'Method / Configuration':<36} | {'Query Test MSE':<18} | {'Error Delta vs Baseline'}")
    print("-" * 95)

    with torch.no_grad():
        x_s, y_s, x_q, y_q = sample_task_batch(eval_tasks, seq_len_supp, seq_len_query)

        # 1. Unadapted Static Forward Model
        y_q_static, _ = twin_system.forward_pass(x_query=x_q, T=4) if False else twin_system.forward_pass(x_q, T=4)
        mse_static = F.mse_loss(y_q_static, y_q).item()
        print(f"{'Static Model (No Adaptation)':<36} | {mse_static:<18.4f} | Baseline Prior")

        # 2. MAML Gradient Descent Baseline
        with torch.enable_grad():
            y_q_maml = maml_model.adapt_and_evaluate(x_s, y_s, x_q, T_fwd=4)
        mse_maml = F.mse_loss(y_q_maml, y_q).item()
        delta_maml = ((mse_maml - mse_static) / mse_static) * 100
        print(f"{'Standard MAML (SGD Inner Loop)':<36} | {mse_maml:<18.4f} | {delta_maml:>+6.1f}%")

        # 3. SubQ Twin with different Error-Settling Hops (T_err)
        for t_err in [1, 2, 4, 6, 8]:
            y_q_twin = twin_system.self_teach_and_evaluate(x_s, y_s, x_q, T_fwd=4, T_err=t_err)
            mse_twin = F.mse_loss(y_q_twin, y_q).item()
            delta_twin = ((mse_twin - mse_static) / mse_static) * 100
            tag = "🏆 SubQ Twin" if mse_twin < mse_maml else "SubQ Twin"
            print(f"{tag + f' (T_err = {t_err} Hops)':<36} | {mse_twin:<18.4f} | {delta_twin:>+6.1f}%")

    print("=" * 95)

@app.local_entrypoint()
def main():
    run_twin_study.remote()
