import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy"
    )
)

app = modal.App("subq-meta-optimizer-benchmark", image=image)

@app.function(gpu="T4", timeout=600)
def run_meta_optimizer_study():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  STUDY: SUBQ AS A DYNAMICAL META-OPTIMIZER (In-Context Layer Adaptation)")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # 1. Meta-Learning Task Generator: Random Non-linear Sine Waves
    # -------------------------------------------------------------------------
    def sample_tasks(batch_size, n_support, n_query):
        # Sample task parameters
        amplitudes = torch.rand(batch_size, 1, device=device) * 4.9 + 0.1  # [0.1, 5.0]
        phases = torch.rand(batch_size, 1, device=device) * math.pi        # [0, pi]
        freqs = torch.rand(batch_size, 1, device=device) * 1.5 + 0.5       # [0.5, 2.0]

        # Sample support and query inputs
        x_all = (torch.rand(batch_size, n_support + n_query, 1, device=device) - 0.5) * 10.0 # [-5, 5]
        y_all = amplitudes.unsqueeze(1) * torch.sin(freqs.unsqueeze(1) * x_all + phases.unsqueeze(1))

        x_supp, x_query = x_all[:, :n_support], x_all[:, n_support:]
        y_supp, y_query = y_all[:, :n_support], y_all[:, n_support:]
        return x_supp, y_supp, x_query, y_query

    # -------------------------------------------------------------------------
    # 2. Downstream Architecture to be Adapted
    # -------------------------------------------------------------------------
    # 2-layer MLP with 32 hidden units: Input (1) -> Hidden (32) -> Output (1)
    # Total parameter vector dimension = (1*32 + 32) + (32*1 + 1) = 32 + 32 + 32 + 1 = 97
    d_param = 97

    def forward_mlp(x, params):
        # x: [B, N, 1], params: [B, 97]
        B, N, _ = x.shape
        w1 = params[:, :32].view(B, 1, 32)         # [B, 1, 32]
        b1 = params[:, 32:64].view(B, 1, 32)        # [B, 1, 32]
        w2 = params[:, 64:96].view(B, 32, 1)        # [B, 32, 1]
        b2 = params[:, 96:97].view(B, 1, 1)         # [B, 1, 1]

        # Layer 1
        h = torch.bmm(x, w1) + b1                   # [B, N, 32]
        h = F.gelu(h)
        # Layer 2
        out = torch.bmm(h, w2) + b2                 # [B, N, 1]
        return out

    # -------------------------------------------------------------------------
    # 3. SubQ Meta-Optimizer Architecture
    # -------------------------------------------------------------------------
    class SubQMetaOptimizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.init_params = nn.Parameter(torch.zeros(1, d_param))
            nn.init.normal_(self.init_params, std=0.1)

            # Error + Context encoder
            # Input to SubQ at step t: current (x_t, y_t, error_t) -> d_param context
            self.error_encoder = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, d_param)
            )

            # SubQ GRU gates for updating the layer state
            self.w_gate_h = nn.Linear(d_param, 2 * d_param, bias=False)
            self.w_cand_h = nn.Linear(d_param, d_param, bias=False)
            self.w_ih = nn.Linear(d_param, 3 * d_param, bias=False)

        def forward(self, x_supp, y_supp):
            # x_supp: [B, K, 1], y_supp: [B, K, 1]
            B, K, _ = x_supp.shape
            params = self.init_params.expand(B, d_param)

            # Sequential adaptation loop: T = K examples shown one by one
            for t in range(K):
                xt = x_supp[:, t : t + 1, :] # [B, 1, 1]
                yt = y_supp[:, t : t + 1, :] # [B, 1, 1]

                # 1. Predict with current layer state
                y_pred = forward_mlp(xt, params) # [B, 1, 1]
                error = y_pred - yt             # [B, 1, 1]

                # 2. Encode step experience
                step_input = torch.cat([xt, yt, error], dim=-1).squeeze(1) # [B, 3]
                ctx = self.error_encoder(step_input)                       # [B, d_param]

                # 3. SubQ Contraction Hop to update parameter state
                gates_ctx = self.w_ih(ctx)
                r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

                gates_h = self.w_gate_h(params)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * params))
                params = (1.0 - z) * n + z * params

            return params

    # -------------------------------------------------------------------------
    # 4. MAML Inner-Loop Gradient Descent Baseline
    # -------------------------------------------------------------------------
    class MAMLBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.init_params = nn.Parameter(torch.zeros(1, d_param))
            nn.init.normal_(self.init_params, std=0.1)
            self.lr = nn.Parameter(torch.tensor(0.01))

        def adapt(self, x_supp, y_supp):
            B, K, _ = x_supp.shape
            params = self.init_params.expand(B, d_param).clone()

            # Sequential inner-loop SGD steps
            for t in range(K):
                xt = x_supp[:, t : t + 1, :]
                yt = y_supp[:, t : t + 1, :]
                y_pred = forward_mlp(xt, params)
                loss = F.mse_loss(y_pred, yt)
                # Compute gradient w.r.t params
                grads = torch.autograd.grad(loss, params, create_graph=True)[0]
                params = params - self.lr * grads

            return params

    # -------------------------------------------------------------------------
    # 5. Training Both Meta-Learners
    # -------------------------------------------------------------------------
    batch_size = 64
    n_support = 10  # T = 10 examples shown sequentially
    n_query = 15
    total_steps = 1200

    print(f"Task: 10-Shot Sequential Meta-Learning (T = {n_support} Support Examples)")
    print("Training SubQ Meta-Optimizer vs MAML Gradient Descent Baseline...\n")

    # Train SubQ Meta-Optimizer
    subq_model = SubQMetaOptimizer().to(device)
    opt_subq = torch.optim.AdamW(subq_model.parameters(), lr=2e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        subq_model.train()
        x_s, y_s, x_q, y_q = sample_tasks(batch_size, n_support, n_query)
        
        # 1. SubQ settles layer parameters across T = K support points
        adapted_params = subq_model(x_s, y_s)

        # 2. Evaluate outer loss on unseen query points
        y_q_pred = forward_mlp(x_q, adapted_params)
        loss = F.mse_loss(y_q_pred, y_q)

        opt_subq.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(subq_model.parameters(), 1.0)
        opt_subq.step()

        if step % 300 == 0 or step == total_steps:
            print(f"[SubQ Meta-Optimizer] Step {step:>4}/{total_steps} | Query MSE: {loss.item():.4f}")
    print(f"SubQ training completed in {time.time() - t0:.1f}s\n")

    # Train MAML Baseline
    maml_model = MAMLBaseline().to(device)
    opt_maml = torch.optim.AdamW(maml_model.parameters(), lr=2e-3, weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        maml_model.train()
        x_s, y_s, x_q, y_q = sample_tasks(batch_size, n_support, n_query)
        
        # MAML adapts inner loop via SGD
        adapted_params = maml_model.adapt(x_s, y_s)

        # Evaluate outer loss on query points
        y_q_pred = forward_mlp(x_q, adapted_params)
        loss = F.mse_loss(y_q_pred, y_q)

        opt_maml.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(maml_model.parameters(), 1.0)
        opt_maml.step()

        if step % 300 == 0 or step == total_steps:
            print(f"[MAML Baseline]       Step {step:>4}/{total_steps} | Query MSE: {loss.item():.4f}")
    print(f"MAML training completed in {time.time() - t0:.1f}s\n")

    # -------------------------------------------------------------------------
    # 6. Evaluation: Query MSE as a Function of Support Examples Shown (K = 1, 2, 5, 10, 20)
    # -------------------------------------------------------------------------
    subq_model.eval()
    maml_model.eval()

    test_shots = [0, 1, 2, 5, 10, 15, 20]
    n_eval_tasks = 200

    print("=" * 90)
    print("  EVALUATION: QUERY MSE AS A FUNCTION OF EXAMPLES SHOWN (T = K)")
    print("=" * 90)
    print(f"{'Support Examples (T=K)':<24} | {'Static (No Adapt) MSE':<22} | {'MAML (SGD) MSE':<18} | {'SubQ Meta-Optimizer MSE':<24} | {'Winner'}")
    print("-" * 105)

    with torch.no_grad():
        for k in test_shots:
            x_s, y_s, x_q, y_q = sample_tasks(n_eval_tasks, max(k, 1), n_query)

            # 1. Static (0 shots)
            static_params = subq_model.init_params.expand(n_eval_tasks, d_param)
            mse_static = F.mse_loss(forward_mlp(x_q, static_params), y_q).item()

            if k == 0:
                print(f"K = {k:<20} | {mse_static:<22.4f} | {'N/A':<18} | {'N/A':<24} | Baseline")
                continue

            # 2. SubQ
            subq_params = subq_model(x_s[:, :k], y_s[:, :k])
            mse_subq = F.mse_loss(forward_mlp(x_q, subq_params), y_q).item()

            # 3. MAML
            # enable grad for inner loop of MAML
            with torch.enable_grad():
                maml_params = maml_model.adapt(x_s[:, :k], y_s[:, :k])
            mse_maml = F.mse_loss(forward_mlp(x_q, maml_params), y_q).item()

            winner = "🏆 SubQ Meta-Optimizer" if mse_subq < mse_maml else "— MAML"
            print(f"K = {k:<20} | {mse_static:<22.4f} | {mse_maml:<18.4f} | {mse_subq:<24.4f} | {winner}")
    print("=" * 105)

@app.local_entrypoint()
def main():
    run_meta_optimizer_study.remote()
