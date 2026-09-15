"""S3-014: TinyShakespeare Autoregressive Language Modeling Shootout (MLP vs. No-MLP in FEN-SubQ).

Evaluates whether the inter-hop non-linear MLP provides genuine representation improvements
over pure linear attention accumulation on character-level autoregressive language modeling.

Benchmark Protocol:
- Dataset: TinyShakespeare (1.11M chars, char-level vocab V=65, 90% train / 10% val)
- Sequence Length: L = 256 tokens
- Hidden Dimension: D = 128
- Hops: T = 4, Offsets: K = 8 causal backward peaks
- Steps: 2,000 steps per model (evaluated every 250 steps over 30 validation batches)
- Strictly Causal:
  * Phase 1: Causal cuDNN GRU Scanner + FEN Gate + Static Causal Prefix Escrow (cumsum)
  * Phase 2: Causal Backward Wave Gathering (target = pos - delta >= 0)
- Conditions:
  * Model 1: FEN-SubQ No-MLP (state = state + scale * context)
  * Model 2: FEN-SubQ With-MLP (state = state + scale * context + scale * MLP(LN(state)))
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
app = modal.App("season3-s3-014-tinyshakespeare-mlp-shootout")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_tinyshakespeare_shootout(
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
    print("  S3-014: DEFINITIVE TINYSHAKESPEARE CAUSAL SHOOTOUT (MLP vs. No-MLP in FEN-SubQ)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Hops T = {n_hops}, K = {k_peaks} Peaks, Steps = {total_steps}")
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
    print(f"Dataset: TinyShakespeare ({len(data):,} chars, Vocab={vocab_size}, Train={len(train_data):,}, Val={len(val_data):,})")

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix])
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Causal FEN-SubQ LM Architecture
    class CausalFENSubQLM(nn.Module):
        def __init__(self, use_mlp: bool = False, mlp_ratio: int = 2):
            super().__init__()
            self.use_mlp = use_mlp
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

            # Optional Inter-Hop MLP
            if self.use_mlp:
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
            # PHASE 1: Causal cuDNN GRU Scan + Causal Prefix Escrow (ONCE!)
            # -------------------------------------------------------------
            H, _ = self.scanner(x)                     # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            V = self.v_proj(G * H)                     # [B, L, D]
            E_all = torch.cumsum(V, dim=1)             # [B, L, D] (strictly causal prefix vault)

            # -------------------------------------------------------------
            # PHASE 2: Causal Backward SubQ Multi-Hop Attention Hops
            # -------------------------------------------------------------
            state = H
            curr_wave = self.init_wave_latent
            pos_grid = torch.arange(L, device=idx.device).unsqueeze(1)  # [L, 1]

            for hop in range(self.n_hops):
                # 1. Harmonic Carrier Wave Evaluation
                params = curr_wave.view(self.num_waves, 4)
                amp = torch.tanh(params[:, 0]).view(1, 1, self.num_waves)
                omega = (F.softplus(params[:, 1]).view(1, 1, self.num_waves) * self.base_freqs)
                phi = (params[:, 2] * math.pi).view(1, 1, self.num_waves)
                decay = (F.softplus(params[:, 3]) * 0.05).view(1, 1, self.num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).squeeze(0)  # [max_d - 1]

                # 2. Extract Top-K Peak Offsets
                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros(1, dtype=torch.long, device=idx.device)
                zero_val = torch.zeros(1, dtype=torch.float, device=idx.device)

                active_offsets = torch.cat([zero_off, past_offsets])  # [K]
                peak_vals = torch.cat([zero_val, topk_vals])           # [K]

                # 3. Strictly Causal Backward Target Indices
                targets = pos_grid - active_offsets.unsqueeze(0)       # [L, K]
                valid_mask = targets >= 0                              # [L, K]
                targets_clamped = torch.clamp(targets, min=0)          # [L, K]

                # 4. Gather candidate escrows
                E_cand = E_all[:, targets_clamped, :]                  # [B, L, K, D]

                # 5. Content-Dependent Q.K Attention
                q = self.ln_q(state)                                   # [B, L, D]
                k = self.ln_k(E_cand)                                  # [B, L, K, D]
                v = E_cand                                             # [B, L, K, D]

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_peaks)
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.unsqueeze(0).float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
                context = (weights.unsqueeze(-1) * v).sum(dim=2)       # [B, L, D]

                # 6. Residual State Update
                state = state + self.scale * context

                # Optional Inter-Hop MLP State Cleaning
                if self.use_mlp:
                    state = state + self.scale * self.mlp(self.ln_mlp(state))

                # 7. Wave Router Annealing
                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            # -------------------------------------------------------------
            # PHASE 3: Classification Head
            # -------------------------------------------------------------
            logits = self.head(self.ln_f(state))                       # [B, L, vocab_size]
            return logits

    # Helper function to evaluate model loss & perplexity
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

    # Function to train a model condition
    def train_model(name: str, use_mlp: bool):
        print(f"\n{'=' * 90}")
        print(f"  TRAINING: {name} (use_mlp={use_mlp})")
        print(f"{'=' * 90}")

        torch.manual_seed(seed)
        model = CausalFENSubQLM(use_mlp=use_mlp).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {num_params:,}")

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

        # Sample 200 chars of text generation
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

        return {
            "name": name,
            "use_mlp": use_mlp,
            "parameters": num_params,
            "total_time_s": round(total_time, 2),
            "final_val_loss": records[-1]["val_loss"],
            "final_val_ppl": records[-1]["val_ppl"],
            "records": records,
            "sample_text": gen_sample,
        }

    # Run Shootout: Model 1 (No-MLP) vs Model 2 (With-MLP)
    res_no_mlp = train_model("FEN-SubQ (No-MLP, Pure Linear State)", use_mlp=False)
    res_with_mlp = train_model("FEN-SubQ (With-MLP, Non-Linear State Cleaning)", use_mlp=True)

    print("\n" + "=" * 105)
    print("  DEFINITIVE TINYSHAKESPEARE SHOOTOUT FINAL RESULTS")
    print("=" * 105)
    print(f"{'Model Architecture':<45} | {'Params':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Time (s)':<10}")
    print("-" * 105)
    print(f"{res_no_mlp['name']:<45} | {res_no_mlp['parameters']:<10,} | {res_no_mlp['final_val_loss']:<10.4f} | {res_no_mlp['final_val_ppl']:<10.2f} | {res_no_mlp['total_time_s']:<10.1f}s")
    print(f"{res_with_mlp['name']:<45} | {res_with_mlp['parameters']:<10,} | {res_with_mlp['final_val_loss']:<10.4f} | {res_with_mlp['final_val_ppl']:<10.2f} | {res_with_mlp['total_time_s']:<10.1f}s")
    print("=" * 105)

    delta_loss = res_no_mlp["final_val_loss"] - res_with_mlp["final_val_loss"]
    delta_ppl = res_no_mlp["final_val_ppl"] - res_with_mlp["final_val_ppl"]
    print(f"\n>>> With-MLP Improvement over No-MLP: Delta Val Loss = {delta_loss:+.4f}, Delta Val PPL = {delta_ppl:+.2f}")

    print("\n--- SAMPLE GENERATION (No-MLP) ---")
    print(res_no_mlp["sample_text"][:250])
    print("\n--- SAMPLE GENERATION (With-MLP) ---")
    print(res_with_mlp["sample_text"][:250])

    results = {
        "dataset": "TinyShakespeare",
        "seq_len": seq_len,
        "d_model": d_model,
        "total_steps": total_steps,
        "no_mlp": res_no_mlp,
        "with_mlp": res_with_mlp,
        "delta_val_loss": round(delta_loss, 4),
        "delta_val_ppl": round(delta_ppl, 2),
    }

    os.makedirs("/models", exist_ok=True)
    with open("/models/s3_014_shootout_results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    return results


@app.local_entrypoint()
def main():
    print("\n>>> Launching S3-014 TinyShakespeare Shootout (MLP vs. No-MLP in FEN-SubQ) on Modal A10G...")
    res = run_tinyshakespeare_shootout.remote(
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
    )
    print("\n>>> Completed S3-014 Shootout Successfully!\n")

    os.makedirs("season3/results", exist_ok=True)
    with open("season3/results/s3_014_tinyshakespeare_mlp_shootout.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results to season3/results/s3_014_tinyshakespeare_mlp_shootout.json")
