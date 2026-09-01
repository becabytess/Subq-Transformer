import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "requests",
        "tiktoken",
        "scipy",
        "scikit-learn"
    )
)

app = modal.App("subq-trajectory-error-signal", image=image)

@app.function(gpu="T4", timeout=900)
def run_trajectory_error_prediction():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("  STUDY: PREDICTING TRUE LOSS / ERROR DIRECTLY FROM TRAJECTORY GEOMETRY")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Setup (GPT-2 BPE)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_text = response.read().decode('utf-8')
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    vocab_size = enc.n_vocab

    print(f"Corpus: {len(tokens):,} BPE tokens | Vocabulary: {vocab_size:,}")

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
    K = len(fib_jumps)

    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # 2. SubQ Model Architecture
    class SubQSurferBPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=False)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=False)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x_norm, T=6):
            B, L, D = x_norm.shape
            pos = torch.arange(L, device=device)
            jump_indices = []
            valid_masks = []
            for j in fib_jumps:
                idx = pos - j
                mask = idx >= 0
                idx = torch.clamp(idx, min=0)
                jump_indices.append(idx)
                valid_masks.append(mask)
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

            ctx = (pi.unsqueeze(-1) * v_cand).sum(dim=3).transpose(1, 2).contiguous().view(B, L, D)
            gates_ctx = self.w_ih(ctx)
            r_ctx, z_ctx, n_ctx = gates_ctx.chunk(3, dim=-1)

            states = [x_norm]
            s = x_norm
            for _ in range(T):
                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)
                r = torch.sigmoid(r_ctx + r_h)
                z = torch.sigmoid(z_ctx + z_h)
                n = torch.tanh(n_ctx + self.w_cand_h(r * s))
                s = (1.0 - z) * n + z * s
                states.append(s)

            return self.out_proj(s), [self.out_proj(st) for st in states]

    class SubQLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = SubQSurferBPE()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp, bias=False),
                nn.GELU(),
                nn.Linear(d_mlp, d_model, bias=False)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, T=6, return_trajectory=False):
            B, L = idx.shape
            x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
            x_norm = self.ln1(x)
            surfer_out, traj = self.surfer(x_norm, T=T)
            x_settled = x + surfer_out
            x_out = x_settled + self.mlp(self.ln2(x_settled))
            logits = self.head(self.norm(x_out))
            
            if not return_trajectory:
                return logits
            return logits, traj

    total_steps = 800
    model = SubQLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    print("Training SubQ model (800 steps with mixed T in [1, 6])...")
    start = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        x, y = get_batch("train")
        T_train = np.random.randint(1, 7)
        logits = model(x, T=T_train)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == total_steps:
            print(f"  Step {step:>4}/{total_steps} | Loss: {loss.item():.4f} | PPL: {math.exp(loss.item()):.2f}")

    print(f"Training completed in {time.time() - start:.1f}s\n")
    model.eval()

    # -------------------------------------------------------------------------
    # 3. Extracting Trajectory Features vs True Prediction Error
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("  EXTRACTING TOKEN-BY-TOKEN TRAJECTORY METRICS & TRUE TARGET ERROR")
    print("=" * 90)

    n_eval_batches = 30
    T_eval = 6

    all_features = []
    all_true_losses = []
    token_str_list = []

    with torch.no_grad():
        for b_idx in range(n_eval_batches):
            x, y = get_batch("val")
            logits, traj = model(x, T=T_eval, return_trajectory=True)

            # Compute true token-level cross-entropy loss: -log P(y | s^T)
            # logits: [B, L, V], y: [B, L]
            log_probs = F.log_softmax(logits, dim=-1)
            true_loss = -log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1) # [B, L]

            B, L = x.shape
            
            # Trajectory Geometric Metrics for each token:
            # traj has T+1 states: s_0, s_1, ..., s_T, each [B, L, D]
            
            # 1. Hop-1 initial velocity: ||s_1 - s_0||
            v1 = (traj[1] - traj[0]).norm(dim=-1) # [B, L]
            
            # 2. Final settling velocity: ||s_T - s_{T-1}||
            v_end = (traj[T_eval] - traj[T_eval - 1]).norm(dim=-1) # [B, L]
            
            # 3. Total Path Length: sum_{t=1}^T ||s_t - s_{t-1}||
            path_len = torch.zeros((B, L), device=device)
            for t in range(1, T_eval + 1):
                path_len += (traj[t] - traj[t - 1]).norm(dim=-1)
                
            # 4. Net Euclidean Displacement: ||s_T - s_0||
            net_disp = (traj[T_eval] - traj[0]).norm(dim=-1).clamp(min=1e-6)
            
            # 5. Path Tortuosity / Curvature: Path Length / Net Displacement (1.0 = straight line, >1.0 = curved/turbulent)
            tortuosity = path_len / net_disp
            
            # 6. Directional alignment / Cosine angle between hop 1 and hop 2:
            v_step1 = traj[1] - traj[0]
            v_step2 = traj[2] - traj[1]
            cos_12 = (v_step1 * v_step2).sum(dim=-1) / (v_step1.norm(dim=-1) * v_step2.norm(dim=-1)).clamp(min=1e-6)

            # Pack features for each token: [v1, v_end, path_len, net_disp, tortuosity, cos_12]
            feats = torch.stack([v1, v_end, path_len, net_disp, tortuosity, cos_12], dim=-1) # [B, L, 6]

            all_features.append(feats.view(-1, 6).cpu().numpy())
            all_true_losses.append(true_loss.view(-1).cpu().numpy())

            if b_idx == 0:
                for tok_id in x[0, :30].cpu().numpy():
                    token_str_list.append(enc.decode([tok_id]))

    X = np.concatenate(all_features, axis=0) # [N, 6]
    y_true = np.concatenate(all_true_losses, axis=0) # [N]
    total_tokens = len(y_true)

    print(f"Collected Trajectory Geometric Signatures for {total_tokens:,} tokens.")

    # -------------------------------------------------------------------------
    # 4. Individual Feature Correlations with Ground Truth Loss
    # -------------------------------------------------------------------------
    feature_names = [
        "1. Hop-1 Velocity (||s1 - s0||)",
        "2. Final Velocity (||s6 - s5||)",
        "3. Total Path Length (Sum ||ds||)",
        "4. Net Displacement (||s6 - s0||)",
        "5. Path Tortuosity (Length / Net)",
        "6. Heading Alignment (cos theta 1->2)"
    ]

    print("\n" + "=" * 90)
    print("  CORRELATION BETWEEN TRAJECTORY GEOMETRY AND REAL TOKEN LOSS")
    print("=" * 90)
    print(f"{'Trajectory Feature':<40} | {'Pearson r':<12} | {'Spearman rank rho':<18} | {'P-value'}")
    print("-" * 90)

    for i, name in enumerate(feature_names):
        r_val, p_val = pearsonr(X[:, i], y_true)
        rho_val, _ = spearmanr(X[:, i], y_true)
        print(f"{name:<40} | {r_val:>+10.4f} | {rho_val:>+16.4f} | {p_val:.2e}")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # 5. Training Linear Regression to Predict Error from Trajectory
    # -------------------------------------------------------------------------
    split_idx = int(0.8 * total_tokens)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_true[:split_idx], y_true[split_idx:]

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r_overall, _ = pearsonr(y_pred, y_test)

    print("\n" + "=" * 90)
    print("  PREDICTED ERROR VS REAL ERROR (TEST SET EVALUATION)")
    print("=" * 90)
    print(f"Overall Pearson Correlation (r):  {r_overall:+.4f}")
    print(f"Variance Explained (R^2 Score):   {r2 * 100:.2f}%")
    print(f"Mean Absolute Error (MAE):        {mae:.4f} nats")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # 6. Trajectory Tortuosity / Turbulence vs Token Surprisal Extremes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  INSPECTING THE EXTREMES: WHAT TOKENS CREATE HIGH VS LOW TRAJECTORY TURBULENCE?")
    print("=" * 90)

    # Sort tokens by trajectory path length
    sorted_indices = np.argsort(X_test[:, 2]) # sort by path length
    low_turb_indices = sorted_indices[:15]
    high_turb_indices = sorted_indices[-15:]

    mean_loss_low = y_test[low_turb_indices].mean()
    mean_loss_high = y_test[high_turb_indices].mean()

    print(f"Tokens with Lowest Trajectory Length  -> Mean Real Loss: {mean_loss_low:.4f} (PPL: {math.exp(mean_loss_low):.2f})")
    print(f"Tokens with Highest Trajectory Length -> Mean Real Loss: {mean_loss_high:.4f} (PPL: {math.exp(mean_loss_high):.2f})")
    print("-" * 90)
    print(f"Ratio of Real Error (High Turbulence / Low Turbulence): {mean_loss_high / mean_loss_low:.2f}x higher error!")
    print("=" * 90)

@app.local_entrypoint()
def main():
    run_trajectory_error_prediction.remote()
