"""S3-015: TinyShakespeare Language Modeling at T=8 Hops (FEN-SubQ With-MLP).

Tests whether increasing thought depth from T=4 to T=8 in FEN-SubQ With-MLP improves
validation loss and perplexity on TinyShakespeare, comparing directly against:
- S3-014 T=4 With-MLP (Val Loss: 1.9021, Val PPL: 6.70)
- Study 63 Pure SubQ T=8 (Val Loss: 1.6790, Val PPL: 5.36)

Configuration:
- Dataset: TinyShakespeare (1.11M chars, char vocab V=65, 90/10 split)
- Sequence Length L = 256, Batch Size = 32, Dim D = 128
- Hops T = 8, Offsets K = 8 backward causal peaks
- Steps = 2,000 (evaluated every 250 steps over 30 validation batches)
- Parameters: 251,648 (identical to S3-014 With-MLP due to weight-tying across hops)
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
app = modal.App("season3-s3-015-tinyshakespeare-t8-mlp")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_t8_mlp_experiment(
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 250,
    val_batches: int = 30,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    n_hops: int = 8,
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
    print(f"  S3-015: FEN-SubQ With-MLP at Deep Thought Depth T = {n_hops} Hops")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, K = {k_peaks} Peaks, Steps = {total_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

    # 1. Dataset Setup
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

    # 2. Causal FEN-SubQ LM Architecture (T=8)
    class CausalFENSubQLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_hops = n_hops
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = 128

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Phase 1: Causal cuDNN GRU Scanner + FEN Feature Extraction
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)

            # Harmonic Wave Router
            self.init_wave_latent = nn.Parameter(torch.randn(num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4),
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, self.max_d - 1, 1))

            # Phase 2: Q.K Attention Normalization
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            # Inter-Hop Non-Linear Token MLP (State Cleaning)
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

            # Phase 1: Causal Scan & Prefix Escrow
            H, _ = self.scanner(x)                     # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            V = self.v_proj(G * H)                     # [B, L, D]
            E_all = torch.cumsum(V, dim=1)             # [B, L, D]

            # Phase 2: T=8 Hops
            state = H
            curr_wave = self.init_wave_latent
            pos_grid = torch.arange(L, device=idx.device).unsqueeze(1)

            for hop in range(self.n_hops):
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

                targets = pos_grid - active_offsets.unsqueeze(0)
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                E_cand = E_all[:, targets_clamped, :]

                q = self.ln_q(state)
                k = self.ln_k(E_cand)
                v = E_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_peaks)
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.unsqueeze(0).float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                # Residual update
                state = state + self.scale * context
                # Inter-Hop MLP
                state = state + self.scale * self.mlp(self.ln_mlp(state))

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
    model = CausalFENSubQLM().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,} (Strict parity with T=4 With-MLP)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    records = []
    t0 = time.time()
    running_loss = 0.0

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

    print("\n" + "=" * 105)
    print("  S3-015 T=8 WITH-MLP FINAL COMPARISON")
    print("=" * 105)
    print(f"{'Model Architecture':<45} | {'Params':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Time (s)':<10}")
    print("-" * 105)
    print(f"{'FEN-SubQ With-MLP (T=4)':<45} | {'251,648':<10} | {'1.9021':<10} | {'6.70':<10} | {'39.8s':<10}")
    print(f"{'FEN-SubQ With-MLP (T=8)':<45} | {num_params:<10,} | {records[-1]['val_loss']:<10.4f} | {records[-1]['val_ppl']:<10.2f} | {total_time:<10.1f}s")
    print(f"{'Pure Harmonic SubQ (T=8, Study 63)':<45} | {'253,000':<10} | {'1.6790':<10} | {'5.36':<10} | {'—':<10}")
    print("=" * 105)

    res = {
        "model": "fen_subq_with_mlp_t8",
        "n_hops": n_hops,
        "parameters": num_params,
        "final_val_loss": records[-1]["val_loss"],
        "final_val_ppl": records[-1]["val_ppl"],
        "total_time_s": round(total_time, 2),
        "records": records,
        "sample_text": gen_sample,
    }

    os.makedirs("season3/results", exist_ok=True)
    with open("season3/results/s3_015_t8_mlp.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results to season3/results/s3_015_t8_mlp.json")

    return res


@app.local_entrypoint()
def main():
    print("\n>>> Launching S3-015 TinyShakespeare T=8 With-MLP on Modal A10G...")
    res = run_t8_mlp_experiment.remote(
        seed=42,
        total_steps=2000,
        eval_interval=250,
        val_batches=30,
        seq_len=256,
        batch_size=32,
        d_model=128,
        n_hops=8,
        k_peaks=8,
        num_waves=12,
        mlp_ratio=2,
    )
    print(f"\n>>> Completed S3-015: Final Val Loss = {res['final_val_loss']}, Val PPL = {res['final_val_ppl']}\n")
