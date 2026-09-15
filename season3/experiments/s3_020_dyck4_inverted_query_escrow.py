"""S3-020: Dyck-4 Deep Bracket Benchmark on Inverted Query-Escrow FEN-SubQ.

Evaluates the project's new Inverted Query-Escrow architecture (cumulative query escrow
E_q + discrete token keys/values + harmonic wave router) on the canonical Dyck-4 deep
bracket benchmark across nesting depths 1 to 30+.

Benchmark Setup (Exact Season 2 Parity):
- Sequence Length L = 256, Batch Size = 32, Vocab Size = 16
- 4 Bracket Pairs: (1->2), (3->4), (5->6), (7->8)
- Nesting Depth: Up to 30+ levels
- Metric: Exact Top-1 prediction on closing bracket positions across:
    * Overall Accuracy
    * Tier 1 (Shallow: 1-5)
    * Tier 2 (Medium: 6-15)
    * Tier 3 (Deep: 16-30+)
- Training: 2,000 steps, AdamW (lr=1e-3, cosine decay to 1e-4, weight_decay=1e-2)
- Historical Baselines:
    * Random Guess Baseline: 25.00%
    * Season 2 SubQ (Fixed Offsets, 218k params): 75.87%
    * Season 2 Dense 4-Layer Transformer (828k params): 94.76%
"""

import json
import math
import os
import random
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season3-s3-020-dyck4-inverted-query-escrow")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_dyck4_benchmark(
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 250,
    val_batches: int = 50,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    d_mlp: int = 512,
    n_hops: int = 4,
    k_peaks: int = 8,
    num_waves: int = 12,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 16
    open_to_close = {1: 2, 3: 4, 5: 6, 7: 8}
    open_brackets = [1, 3, 5, 7]
    max_d = seq_len

    print("=" * 100)
    print("  S3-020: DYCK-4 DEEP BRACKET BENCHMARK (INVERTED QUERY-ESCROW FEN-SUBQ)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Hops T = {n_hops}, K = {k_peaks} Peaks, Steps = {total_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    # 1. Exact Canonical Dyck-4 Sequence Generator from Season 2 (s2_029)
    def generate_dyck_sequence(target_len=256, max_depth=30, rng=None):
        _rand = rng.random if rng is not None else random.random
        _choice = rng.choice if rng is not None else random.choice

        tokens = []
        stack = []
        depth_at_pos = []

        while len(tokens) < target_len - 1:
            curr_depth = len(stack)
            p_close = 0.0 if curr_depth == 0 else (0.45 if curr_depth < max_depth else 0.90)

            if len(tokens) + curr_depth >= target_len - 1:
                p_close = 1.0

            if _rand() < p_close and curr_depth > 0:
                last_open = stack.pop()
                expected_close = open_to_close[last_open]
                tokens.append(expected_close)
                depth_at_pos.append(curr_depth)
            else:
                b = _choice(open_brackets)
                stack.append(b)
                tokens.append(b)
                depth_at_pos.append(len(stack))

        while stack and len(tokens) < target_len:
            last_open = stack.pop()
            tokens.append(open_to_close[last_open])
            depth_at_pos.append(len(stack) + 1)

        while len(tokens) < target_len:
            tokens.append(0)
            depth_at_pos.append(0)

        return tokens[:target_len], depth_at_pos[:target_len]

    def generate_dyck_batch(batch_size, rng):
        x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        depths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        for b in range(batch_size):
            tokens, depth_list = generate_dyck_sequence(seq_len, max_depth=30, rng=rng)
            x[b] = torch.tensor(tokens, device=device)
            depths[b] = torch.tensor(depth_list, device=device)

            for t in range(seq_len - 1):
                next_tok = tokens[t + 1]
                if next_tok in [2, 4, 6, 8]:
                    y[b, t] = next_tok

        return x, y, depths

    # 2. Inverted Query-Escrow FEN-SubQ Model
    class InvertedQueryEscrowDyck4(nn.Module):
        def __init__(self):
            super().__init__()
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

            # Phase 2: Discrete Key and Value Projections (NO CUMSUM)
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

            # Inter-Hop Non-Linear Token MLP
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )

            # Output Head
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, return_stats: bool = False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Phase 1: Causal Scan -> Cumulative Query Escrow Vault
            H, _ = self.scanner(x)                     # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            Q_raw = self.q_proj(G * H)                 # [B, L, D]
            E_q_all = torch.cumsum(Q_raw, dim=1)       # [B, L, D]

            # Phase 2: Multi-Hop Retrieval over Discrete Tokens
            state = H
            curr_wave = self.init_wave_latent
            pos_grid = torch.arange(L, device=idx.device).unsqueeze(1)
            hop_offsets = []
            hop_entropies = []

            for hop in range(self.n_hops):
                params = curr_wave.view(self.num_waves, 4)
                amp = torch.tanh(params[:, 0]).view(1, 1, self.num_waves)
                omega = (F.softplus(params[:, 1]).view(1, 1, self.num_waves) * self.base_freqs)
                phi = (params[:, 2] * math.pi).view(1, 1, self.num_waves)
                decay = (F.softplus(params[:, 3]) * 0.05).view(1, 1, self.num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).squeeze(0)  # [max_d - 1]

                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros(1, dtype=torch.long, device=idx.device)
                zero_val = torch.zeros(1, dtype=torch.float, device=idx.device)

                active_offsets = torch.cat([zero_off, past_offsets])
                peak_vals = torch.cat([zero_val, topk_vals])

                if return_stats:
                    hop_offsets.append(active_offsets.detach().cpu().tolist())

                targets = pos_grid - active_offsets.unsqueeze(0)  # [L, K]
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                # Discrete Keys and Values from State
                K_discrete = self.k_proj(state)        # [B, L, D]
                V_discrete = self.v_proj(state)        # [B, L, D]

                K_cand = K_discrete[:, targets_clamped, :]  # [B, L, K, D]
                V_cand = V_discrete[:, targets_clamped, :]  # [B, L, K, D]

                # Query combines prefix query escrow + dynamic state
                q = self.ln_q(E_q_all + state)
                k = self.ln_k(K_cand)
                v = V_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_peaks)
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.unsqueeze(0).float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

                if return_stats:
                    ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
                    hop_entropies.append(round(ent, 3))

                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                state = state + self.scale * context
                state = state + self.scale * self.mlp(self.ln_mlp(state))

                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            logits = self.head(self.ln_f(state))
            if return_stats:
                return logits, {
                    "offsets": hop_offsets,
                    "entropies": hop_entropies,
                    "gate": float(G.detach().mean().item()),
                }
            return logits

    @torch.no_grad()
    def evaluate(model, rng):
        model.eval()
        tiers = {
            "tier1_shallow_1_5": [1, 6],
            "tier2_medium_6_15": [6, 16],
            "tier3_deep_16_30": [16, 100],
        }
        tier_correct = {k: 0 for k in tiers}
        tier_total = {k: 0 for k in tiers}
        total_correct = 0
        total_eval = 0

        for _ in range(val_batches):
            x_val, y_val, depths_val = generate_dyck_batch(batch_size, rng)
            logits = model(x_val)
            pred = logits.argmax(dim=-1)

            mask = (y_val != -100)
            correct_mask = (pred == y_val) & mask

            total_correct += int(correct_mask.sum().item())
            total_eval += int(mask.sum().item())

            for tier_name, (lo, hi) in tiers.items():
                tier_pos = mask & (depths_val >= lo) & (depths_val < hi)
                tier_total[tier_name] += int(tier_pos.sum().item())
                tier_correct[tier_name] += int((correct_mask & tier_pos).sum().item())

        overall_acc = 100.0 * total_correct / max(1, total_eval)
        tier_acc = {
            k: 100.0 * tier_correct[k] / max(1, tier_total[k]) if tier_total[k] > 0 else 0.0
            for k in tiers
        }

        return {
            "overall_acc": overall_acc,
            "tier_acc": tier_acc,
            "total_eval": total_eval,
            "tier_total": tier_total,
        }

    # Instantiate Model
    train_rng = random.Random(seed + 1000)
    val_rng = random.Random(seed + 5000)

    model = InvertedQueryEscrowDyck4().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model: Inverted Query-Escrow FEN-SubQ (Dyck-4)")
    print(f"  Total Parameters: {param_count:,}")
    print("-" * 100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-4
    )

    start_time = time.time()
    step_records = []

    for step in range(1, total_steps + 1):
        model.train()
        x_b, y_b, _ = generate_dyck_batch(batch_size, train_rng)
        optimizer.zero_grad(set_to_none=True)

        is_eval_step = (step % eval_interval == 0 or step == total_steps)
        if is_eval_step:
            logits, stats = model(x_b, return_stats=True)
        else:
            logits = model(x_b, return_stats=False)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_b.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if is_eval_step:
            metrics = evaluate(model, val_rng)
            elapsed = time.time() - start_time
            t1 = metrics["tier_acc"]["tier1_shallow_1_5"]
            t2 = metrics["tier_acc"]["tier2_medium_6_15"]
            t3 = metrics["tier_acc"]["tier3_deep_16_30"]
            print(
                f"  Step {step:04d}/{total_steps} | Loss: {loss.item():.4f} | "
                f"Overall Acc: {metrics['overall_acc']:.2f}% | "
                f"Shallow (1-5): {t1:.1f}% | Med (6-15): {t2:.1f}% | Deep (16-30): {t3:.1f}% | "
                f"Elapsed: {elapsed:.1f}s"
            )
            print(f"    Hop 1 Offsets: {stats['offsets'][0]} | Attn Entropy: {stats['entropies'][0]}")
            print(f"    Hop {n_hops} Offsets: {stats['offsets'][-1]} | Attn Entropy: {stats['entropies'][-1]}")

            step_records.append({
                "step": step,
                "loss": float(loss.item()),
                "overall_acc": metrics["overall_acc"],
                "tier_acc": metrics["tier_acc"],
                "offsets": stats["offsets"],
                "entropies": stats["entropies"],
                "gate": stats["gate"],
                "elapsed": elapsed,
            })

    final_metrics = evaluate(model, val_rng)
    total_time = time.time() - start_time
    print("=" * 100)
    print(f"  FINAL RESULT: Inverted Query-Escrow FEN-SubQ on Dyck-4")
    print(f"  Overall Accuracy: {final_metrics['overall_acc']:.2f}%")
    for t_name, t_acc in final_metrics["tier_acc"].items():
        print(f"    {t_name}: {t_acc:.2f}%")
    print(f"  Comparison vs Baselines:")
    print(f"    Random Guess: 25.00%")
    print(f"    Season 2 SubQ (Fixed Offsets, 218k): 75.87%")
    print(f"    Season 2 Dense 4-Layer Transformer (828k): 94.76%")
    print(f"  Total Time: {total_time:.1f}s")
    print("=" * 100)

    return {
        "model": "Inverted Query-Escrow FEN-SubQ",
        "parameters": param_count,
        "final_overall_acc": final_metrics["overall_acc"],
        "final_tier_acc": final_metrics["tier_acc"],
        "total_eval_tokens": final_metrics["total_eval"],
        "step_records": step_records,
        "total_time": total_time,
    }


@app.local_entrypoint()
def main():
    print("Launching Dyck-4 Deep Bracket benchmark with Inverted Query-Escrow on Modal A10G...")
    result = run_dyck4_benchmark.remote()

    os.makedirs("season3/results", exist_ok=True)
    out_path = "season3/results/s3_020_dyck4_inverted_query_escrow.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] Experiment results saved to {out_path}")
