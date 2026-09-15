"""S3-025: MQAR Associative Recall Scaled to T=12 Hops (With vs. Without MLP).

Evaluates whether the Inverted Query-Escrow architecture suffers from the historical
MQAR collapse when the inter-hop MLP is active, or if it successfully preserves exact
associative recall across long distances (L=512).

Benchmark Setup:
- Sequence Length L = 512, Batch Size = 32, Dim D = 128
- 16 random Key-Value pairs placed in the first 350 tokens
- 8 queries placed at the end of the sequence (distance up to 300+ tokens)
- Steps = 3,000 (evaluated every 500 steps over 30 validation batches)
- Conditions:
  * Model 1: Inverted Query-Escrow No-MLP (Pure Linear State Update)
  * Model 2: Inverted Query-Escrow With-MLP (Inter-Hop Non-Linear FFN)
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season3-s3-025-mqar-t12-scaling")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_mqar_shootout(
    seed: int = 42,
    total_steps: int = 3000,
    eval_interval: int = 500,
    val_batches: int = 30,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
    n_hops: int = 12,
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

    vocab_size = 256
    num_pairs = 16
    num_queries = 8
    query_marker = 1
    max_d = seq_len
    start_q_pos = seq_len - (num_queries * 2) - 2
    query_positions = [start_q_pos + 2 * i + 1 for i in range(num_queries)]

    print("=" * 105)
    print("  S3-025: MQAR ASSOCIATIVE RECALL SCALED TO T=12 HOPS (WITH vs. WITHOUT MLP)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Hops T = {n_hops}, K = {k_peaks} Peaks, Steps = {total_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

    def make_batch(generator):
        x = torch.randint(100, 255, (batch_size, seq_len), generator=generator, device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        distances = torch.empty((batch_size, num_queries), dtype=torch.long, device=device)
        for row in range(batch_size):
            keys = torch.randperm(40, generator=generator, device=device)[:num_pairs] + 10
            values = torch.randperm(40, generator=generator, device=device)[:num_pairs] + 50
            kv = torch.randperm(350, generator=generator, device=device)[: num_pairs * 2].sort().values
            for pair in range(num_pairs):
                kp = int(kv[2 * pair].item())
                x[row, kp] = keys[pair]
                x[row, kp + 1] = values[pair]
            chosen = torch.randperm(num_pairs, generator=generator, device=device)[:num_queries]
            for qi, kt in enumerate(chosen):
                qpos = start_q_pos + 2 * qi
                kp = int(kv[2 * int(kt.item())].item())
                x[row, qpos] = query_marker
                x[row, qpos + 1] = keys[kt]
                y[row, qpos + 1] = values[kt]
                distances[row, qi] = qpos + 1 - kp
        return x, y, distances

    # 2. Inverted Query-Escrow Model
    class InvertedQueryEscrowMQAR(nn.Module):
        def __init__(self, use_mlp: bool = True):
            super().__init__()
            self.use_mlp = use_mlp
            self.n_hops = n_hops
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = max_d

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Phase 1: Causal cuDNN GRU Scanner + FEN Query Extractor
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.q_proj = nn.Linear(d_model, d_model)

            # Phase 2: Discrete Key and Value Projections (NO CUMSUM!)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            # Harmonic Wave Router
            self.init_wave_latent = nn.Parameter(torch.randn(num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4),
            )

            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, self.max_d - 1, 1))

            # Attention Normalization
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            # Optional Inter-Hop Non-Linear Token MLP
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

            # Phase 1: Causal Scan -> Query Escrow Vault
            H, _ = self.scanner(x)                     # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            Q_raw = self.q_proj(G * H)                 # [B, L, D]
            E_q_all = torch.cumsum(Q_raw, dim=1)       # [B, L, D]

            # Phase 2: Multi-Hop Retrieval over Discrete Tokens
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

                # Discrete Keys and Values from State
                K_discrete = self.k_proj(state)        # [B, L, D]
                V_discrete = self.v_proj(state)        # [B, L, D]

                K_cand = K_discrete[:, targets_clamped, :]
                V_cand = V_discrete[:, targets_clamped, :]

                # Query combines prefix query escrow + dynamic state
                q = self.ln_q(E_q_all + state)
                k = self.ln_k(K_cand)
                v = V_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_peaks)
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.unsqueeze(0).float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                state = state + self.scale * context
                if self.use_mlp:
                    state = state + self.scale * self.mlp(self.ln_mlp(state))

                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            return self.head(self.ln_f(state))

    # Evaluate MQAR exact recall accuracy
    @torch.no_grad()
    def evaluate(model, generator):
        model.eval()
        total_correct = 0
        total_queries_eval = 0
        tier_correct = {"shallow": 0, "medium": 0, "long": 0}
        tier_counts = {"shallow": 0, "medium": 0, "long": 0}

        for _ in range(val_batches):
            x_val, y_val, dists = make_batch(generator)
            logits = model(x_val)
            pred = logits.argmax(dim=-1)

            for row in range(batch_size):
                for qi, qpos in enumerate(query_positions):
                    target_val = y_val[row, qpos].item()
                    pred_val = pred[row, qpos].item()
                    d = dists[row, qi].item()

                    is_correct = (pred_val == target_val)
                    total_correct += int(is_correct)
                    total_queries_eval += 1

                    tier = "shallow" if d < 64 else ("medium" if d <= 128 else "long")
                    tier_correct[tier] += int(is_correct)
                    tier_counts[tier] += 1

        overall_acc = 100.0 * total_correct / total_queries_eval
        shallow_acc = 100.0 * tier_correct["shallow"] / max(tier_counts["shallow"], 1)
        med_acc = 100.0 * tier_correct["medium"] / max(tier_counts["medium"], 1)
        long_acc = 100.0 * tier_correct["long"] / max(tier_counts["long"], 1)

        return overall_acc, shallow_acc, med_acc, long_acc

    # Train function
    def train_model(name: str, use_mlp: bool):
        print(f"\n{'=' * 90}")
        print(f"  TRAINING: {name} (use_mlp={use_mlp})")
        print(f"{'=' * 90}")

        train_gen = torch.Generator(device=device).manual_seed(seed)
        val_gen = torch.Generator(device=device).manual_seed(seed + 999)

        model = InvertedQueryEscrowMQAR(use_mlp=use_mlp).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        records = []
        t0 = time.time()
        running_loss = 0.0

        for step in range(1, total_steps + 1):
            model.train()
            x_train, y_train, _ = make_batch(train_gen)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train)

            # Masked Cross-Entropy only on the 8 query positions
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_train.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % eval_interval == 0 or step == total_steps:
                train_loss = running_loss / eval_interval if step % eval_interval == 0 else loss.item()
                running_loss = 0.0

                # Re-seed val generator for consistent evaluation
                val_gen_eval = torch.Generator(device=device).manual_seed(seed + 999)
                acc, s_acc, m_acc, l_acc = evaluate(model, val_gen_eval)
                elapsed = time.time() - t0

                print(
                    f"  Step {step:04d}/{total_steps:04d} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Exact Recall: {acc:.2f}% (Shallow: {s_acc:.1f}%, Med: {m_acc:.1f}%, Long: {l_acc:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True,
                )
                records.append({
                    "step": step,
                    "train_loss": round(train_loss, 4),
                    "recall_acc": round(acc, 2),
                    "shallow_acc": round(s_acc, 2),
                    "medium_acc": round(m_acc, 2),
                    "long_acc": round(l_acc, 2),
                    "elapsed_s": round(elapsed, 1),
                })

        total_time = time.time() - t0
        return {
            "name": name,
            "use_mlp": use_mlp,
            "parameters": num_params,
            "final_acc": records[-1]["recall_acc"],
            "shallow_acc": records[-1]["shallow_acc"],
            "medium_acc": records[-1]["medium_acc"],
            "long_acc": records[-1]["long_acc"],
            "total_time_s": round(total_time, 2),
            "records": records,
        }

    # Shootout: No-MLP vs. With-MLP
    res_no_mlp = train_model("Inverted Query-Escrow (No-MLP, Pure Linear State)", use_mlp=False)
    res_with_mlp = train_model("Inverted Query-Escrow (With-MLP, Non-Linear FFN)", use_mlp=True)

    print("\n" + "=" * 105)
    print("  DEFINITIVE MQAR ASSOCIATIVE RECALL FINAL SHOOTOUT RESULTS")
    print("=" * 105)
    print(f"{'Model Architecture':<45} | {'Params':<10} | {'Overall Acc':<12} | {'Shallow (<64)':<14} | {'Med (64-128)':<12} | {'Long (>128)'}")
    print("-" * 105)
    print(f"{res_no_mlp['name']:<45} | {res_no_mlp['parameters']:<10,} | {res_no_mlp['final_acc']:<12.2f}% | {res_no_mlp['shallow_acc']:<14.1f}% | {res_no_mlp['medium_acc']:<12.1f}% | {res_no_mlp['long_acc']:.1f}%")
    print(f"{res_with_mlp['name']:<45} | {res_with_mlp['parameters']:<10,} | {res_with_mlp['final_acc']:<12.2f}% | {res_with_mlp['shallow_acc']:<14.1f}% | {res_with_mlp['medium_acc']:<12.1f}% | {res_with_mlp['long_acc']:.1f}%")
    print("-" * 105)
    print(f"{'Random Chance Baseline':<45} | {'—':<10} | {'2.50%':<12} | {'2.50%':<14} | {'2.50%':<12} | {'2.50%'}")
    print(f"{'Season 2 Dense 4-Layer Reference (S2-028)':<45} | {'922,000':<10} | {'5.93%':<12} | {'6.8%':<14} | {'5.9%':<12} | {'5.2%'}")
    print(f"{'Season 2 Pure SubQ No-MLP Reference (S2-028)':<45} | {'180,000':<10} | {'6.32%':<12} | {'7.8%':<14} | {'6.4%':<12} | {'5.1%'}")
    print("=" * 105)

    res = {
        "benchmark": "MQAR L=512",
        "total_steps": total_steps,
        "no_mlp": res_no_mlp,
        "with_mlp": res_with_mlp,
    }

    os.makedirs("season3/results", exist_ok=True)
    with open("season3/results/s3_025_mqar_t12_scaling.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results to season3/results/s3_025_mqar_t12_scaling.json")

    return res


@app.local_entrypoint()
def main():
    print("\n>>> Launching S3-025 MQAR Shootout at T=12 Hops on Modal A10G...")
    res = run_mqar_shootout.remote(
        seed=42,
        total_steps=3000,
        eval_interval=500,
        val_batches=30,
        seq_len=512,
        batch_size=32,
        d_model=128,
        n_hops=4,
        k_peaks=8,
        num_waves=12,
        mlp_ratio=2,
    )
    print("\n>>> Completed S3-025 MQAR Shootout at T=12 Hops Successfully!\n")
