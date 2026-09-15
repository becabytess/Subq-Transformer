"""S3-034: Escrow Necessity Ablation on TinyShakespeare.

Scientific Question:
Is the Inverted Cumulative Query Escrow (E_q = cumsum(Q_raw)) providing critical
functional value, or is it overengineering where the front-end cuDNN GRU scanner
hidden state alone (s = H) does all the work?

Setup (Condition B - Without Escrow / GRU Alone):
- Remove E_q_all, remove cumsum, remove FEN gate G.
- Query is projected directly from the running token state at each hop:
  q = LayerNorm(W_q(state))
- Exact same cuDNN GRU scanner (H = GRU(X), state = H).
- Exact same discrete Keys and Values (K = W_k(state), V = W_v(state)).
- Exact same Global Harmonic Wave router (K=8 peaks, 12 waves).
- Exact same Inter-Hop MLP (mlp_ratio=2).
- Strict parity with S3-016: L=256, d=128, T=4 hops, 2,000 steps, batch size 32.

Reference Baselines (NOT re-trained):
- S3-016 (With Cumulative Query Escrow, T=4): Val Loss = 1.6535, Val PPL = 5.23
- S3-026 (State-Conditioned Per-Token Waves, T=4): Val Loss = 1.6185, Val PPL = 5.05
- S1 Study 63 (Pure Harmonic SubQ, NO GRU, T=8): Val Loss = 1.6790, Val PPL = 5.36
- S1 Study 58 (Pure Harmonic SubQ, NO GRU, T=4): Val Loss = 1.7092, Val PPL = 5.52
"""

import json
import math
import os
import time
import urllib.request
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season3-s3-034-no-escrow-gru-only")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_no_escrow_ablation(
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 250,
    val_batches: int = 30,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    n_hops: int = 4,
    k_peaks: int = 8,
    num_waves: int = 12,
    mlp_ratio: int = 2,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 105)
    print("  S3-034: ESCROW NECESSITY ABLATION (CONDITION B: NO ESCROW / GRU HIDDEN STATE ALONE)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Hops T = {n_hops}, K = {k_peaks} Peaks, Steps = {total_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

    # 1. Dataset Setup (Identical to S3-016)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode("utf-8")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix])
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Architecture Condition B: GRU Scanner + Discrete Multi-Hop SubQ (NO ESCROW!)
    class NoEscrowGRUOnlySubQLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_hops = n_hops
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = 128

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Phase 1: Causal cuDNN GRU Scanner (NO ESCROW VAULT, NO CUMSUM, NO GATE!)
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)

            # Phase 2: Q, K, V Projections from Hidden State
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            # Harmonic Wave Router (Identical to S3-016)
            self.init_wave_latent = nn.Parameter(torch.randn(num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4),
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, self.max_d - 1, 1))

            # Normalization
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            # Inter-Hop Non-Linear Token MLP (Identical to S3-016)
            mlp_dim = d_model * mlp_ratio
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model),
            )

            # Output Head
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # -------------------------------------------------------------
            # PHASE 1: Causal GRU Scan -> Hidden State (NO ESCROW VAULT!)
            # -------------------------------------------------------------
            H, _ = self.scanner(x)                     # [B, L, D]

            # -------------------------------------------------------------
            # PHASE 2: Multi-Hop Retrieval from Hidden State Alone
            # -------------------------------------------------------------
            state = H
            curr_wave = self.init_wave_latent
            pos_grid = torch.arange(L, device=idx.device).unsqueeze(1)

            for hop in range(self.n_hops):
                # 1. Harmonic Carrier Wave Evaluation
                params = curr_wave.view(self.num_waves, 4)
                amp = torch.tanh(params[:, 0]).view(1, 1, self.num_waves)
                omega = (F.softplus(params[:, 1]).view(1, 1, self.num_waves) * self.base_freqs)
                phi = (params[:, 2] * math.pi).view(1, 1, self.num_waves)
                decay = (F.softplus(params[:, 3]) * 0.05).view(1, 1, self.num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).squeeze(0)

                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros(1, dtype=torch.long, device=idx.device)
                zero_val = torch.zeros(1, dtype=torch.float, device=idx.device)

                active_offsets = torch.cat([zero_off, past_offsets])
                peak_vals = torch.cat([zero_val, topk_vals])

                # 2. Causal backward targets
                targets = pos_grid - active_offsets.unsqueeze(0)
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                # 3. Discrete Keys & Values from evolving state
                K_discrete = self.k_proj(state)        # [B, L, D]
                V_discrete = self.v_proj(state)        # [B, L, D]

                K_cand = K_discrete[:, targets_clamped, :]  # [B, L, K, D]
                V_cand = V_discrete[:, targets_clamped, :]  # [B, L, K, D]

                # 4. QUERY PROJECTED FROM HIDDEN STATE ALONE (NO E_q_all!)
                q = self.ln_q(self.q_proj(state))      # [B, L, D] - Direct from state!
                k = self.ln_k(K_cand)                  # [B, L, K, D]
                v = V_cand                             # [B, L, K, D]

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_peaks)
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.unsqueeze(0).float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                # 5. Residual State Update
                state = state + self.scale * context
                state = state + self.scale * self.mlp(self.ln_mlp(state))

                # 6. Wave Router Annealing
                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            return self.head(self.ln_f(state))

    @torch.no_grad()
    def evaluate(model):
        model.eval()
        total_loss = 0.0
        for b in range(val_batches):
            x_val, y_val = get_batch("val", step_seed=1000 + b)
            logits = model(x_val)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_val.view(-1))
            total_loss += loss.item()
        mean_val_loss = total_loss / val_batches
        val_ppl = math.exp(mean_val_loss)
        return mean_val_loss, val_ppl

    torch.manual_seed(seed)
    model = NoEscrowGRUOnlySubQLM().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters (Condition B): {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    records = []
    t0 = time.time()
    running_loss = 0.0

    print("\n--- Starting Training (2000 Steps) ---")
    for step in range(1, total_steps + 1):
        model.train()
        x_train, y_train = get_batch("train")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_train.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % eval_interval == 0 or step == total_steps:
            train_loss = running_loss / eval_interval if step % eval_interval == 0 else loss.item()
            running_loss = 0.0
            val_loss, val_ppl = evaluate(model)
            elapsed = time.time() - t0

            print(
                f"  Step {step:04d}/{total_steps:04d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True,
            )
            records.append({
                "step": step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 2),
                "elapsed_s": round(elapsed, 1),
            })

    total_time = time.time() - t0

    # Text sample
    model.eval()
    with torch.no_grad():
        prompt_str = "QUEEN:\n"
        prompt_ids = torch.tensor([char_to_ix[c] for c in prompt_str], dtype=torch.long, device=device).unsqueeze(0)
        out_ids = list(prompt_ids[0].cpu().numpy())
        for _ in range(200):
            curr_in = torch.tensor(out_ids[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(curr_in)
            next_token = torch.multinomial(F.softmax(logits[0, -1] / 0.8, dim=-1), 1).item()
            out_ids.append(next_token)
        gen_sample = "".join([ix_to_char[i] for i in out_ids])

    # Final Comparison Benchmark
    b_val_loss = records[-1]["val_loss"]
    b_val_ppl = records[-1]["val_ppl"]
    base_val_loss = 1.6535
    base_val_ppl = 5.23

    print("\n" + "=" * 105)
    print("  S3-034 ESCROW NECESSITY ABLATION: FINAL COMPARISON TABLE")
    print("=" * 105)
    print(f"{'Model Configuration':<50} | {'Params':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Delta PPL':<10}")
    print("-" * 105)
    print(f"{'S3-016: Baseline (Inverted Query Escrow E_q, T=4)':<50} | {'284,416':<10} | {base_val_loss:<10.4f} | {base_val_ppl:<10.2f} | {'Baseline':<10}")
    print(f"{'S3-034: Condition B (No Escrow / GRU Alone, T=4)':<50} | {num_params:<10,} | {b_val_loss:<10.4f} | {b_val_ppl:<10.2f} | {b_val_ppl - base_val_ppl:+10.2f}")
    print(f"{'S3-026: State-Conditioned Per-Token Waves (T=4)':<50} | {'287,664':<10} | {'1.6185':<10} | {'5.05':<10} | {'-0.18':<10}")
    print(f"{'S1 Study 63: Pure Harmonic SubQ (NO GRU, T=8)':<50} | {'253,000':<10} | {'1.6790':<10} | {'5.36':<10} | {'+0.13':<10}")
    print(f"{'S1 Study 58: Pure Harmonic SubQ (NO GRU, T=4)':<50} | {'247,000':<10} | {'1.7092':<10} | {'5.52':<10} | {'+0.29':<10}")
    print("=" * 105)

    delta_loss = b_val_loss - base_val_loss
    delta_ppl = b_val_ppl - base_val_ppl

    if abs(delta_ppl) < 0.10:
        conclusion = "ESCROW IS REDUNDANT / OVERENGINEERED. The GRU scanner hidden state alone achieves parity."
    elif delta_ppl > 0.10:
        conclusion = f"ESCROW IS ESSENTIAL (+{delta_ppl:.2f} PPL regression without it). Cumulative query vault E_q is a vital linear integrator."
    else:
        conclusion = f"CONDITION B OUTPERFORMS BASELINE ({delta_ppl:.2f} PPL). Removing escrow actually helped."

    print(f"\n>>> VERDICT: {conclusion}\n")

    res = {
        "model": "s3_034_no_escrow_gru_only",
        "parameters": num_params,
        "final_val_loss": b_val_loss,
        "final_val_ppl": b_val_ppl,
        "delta_loss_vs_s3_016": round(delta_loss, 4),
        "delta_ppl_vs_s3_016": round(delta_ppl, 2),
        "total_time_s": round(total_time, 2),
        "verdict": conclusion,
        "records": records,
        "sample_text": gen_sample,
    }

    os.makedirs("season3/results", exist_ok=True)
    with open("season3/results/s3_034_no_escrow_gru_only.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results to season3/results/s3_034_no_escrow_gru_only.json")

    return res


@app.local_entrypoint()
def main():
    print("\n>>> Launching S3-034 No-Escrow Ablation on Modal A10G...")
    res = run_no_escrow_ablation.remote(
        seed=42,
        total_steps=2000,
        eval_interval=250,
        val_batches=30,
        seq_len=256,
        batch_size=32,
        d_model=128,
        n_hops=4,
        k_peaks=8,
        num_waves=12,
        mlp_ratio=2,
    )
    print(f"\n>>> Completed S3-034: Val Loss = {res['final_val_loss']}, Val PPL = {res['final_val_ppl']}")
    print(f">>> Verdict: {res['verdict']}\n")
