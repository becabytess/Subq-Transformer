"""S3-029: State-Conditioned Per-Token Wave Router on Dyck-4 Bracket Grammar (T=4 Hops).

Scientific Hypothesis:
In S3-020 and S3-024, closing brackets at different stack depths (depth 3 vs depth 25)
were forced to share the same broadcast harmonic wave offsets, requiring 12 hops of passive
transitive diffusion to resolve deep nesting (93.08% at T=12 vs 91.95% at T=4).

With State-Conditioned Per-Token Waves:
  [A_i, omega_i, phi_i, lambda_i] = WaveHead(LN(s_i^(t)))
Each token emits its personal continuous wave. Because the running state s_i^(t) already carries
the stack counter from the causal GRU scan, closing brackets can directly emit the wavelengths
and lookback distances tailored to their specific nesting depth.

Benchmark:
- Dyck-4 Bracket Grammar (L=256, 4 pairs, depths 1 to 30+, 2,000 steps, batch size 32)
- Hops T = 4, Peaks K = 8, Dim D = 128, MLP = 512
- Reference Champions:
  * S3-020 (Global Wave, T=4 Hops): 89.96% Overall | 91.32% Shallow | 86.90% Med | 91.95% Deep
  * S3-024 (Global Wave, T=12 Hops): 90.85% Overall | 91.13% Shallow | 88.01% Med | 93.08% Deep
  * Dense 4L-Transformer (828k params): 94.76% Overall | 94.80% Deep
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
app = modal.App("season3-s3-029-dyck4-per-token-waves")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_dyck4_per_token_benchmark(
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

    print("=" * 105)
    print("  S3-029: STATE-CONDITIONED PER-TOKEN WAVE ROUTER ON DYCK-4 (T=4 HOPS)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Hops T = {n_hops}, K = {k_peaks} Peaks, Steps = {total_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

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

    # 2. State-Conditioned Per-Token Wave Inverted Query-Escrow Model
    class StateConditionedPerTokenDyck4(nn.Module):
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

            # Phase 2: Discrete Key and Value Projections
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            # State-Conditioned Per-Token Wave Emission Head
            self.ln_wave = nn.LayerNorm(d_model)
            self.wave_head = nn.Linear(d_model, num_waves * 4)

            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves, 1))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, 1, self.max_d - 1))

            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, return_dynamics: bool = False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            H, _ = self.scanner(x)
            G = torch.sigmoid(self.gate(H))
            Q_raw = self.q_proj(G * H)
            E_q_all = torch.cumsum(Q_raw, dim=1)

            state = H
            pos_grid = torch.arange(L, device=idx.device).view(1, L, 1)

            diagnostics = {
                "offset_std_per_token": [],
                "mean_attended_distances": [],
                "hop_entropies": [],
            }

            for hop in range(self.n_hops):
                wave_params = self.wave_head(self.ln_wave(state)).view(B, L, self.num_waves, 4)
                amp = torch.tanh(wave_params[..., 0]).unsqueeze(-1)
                omega = F.softplus(wave_params[..., 1]).unsqueeze(-1) * self.base_freqs
                phi = (wave_params[..., 2] * math.pi).unsqueeze(-1)
                decay = (F.softplus(wave_params[..., 3]) * 0.05).unsqueeze(-1)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=2)

                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros((B, L, 1), dtype=torch.long, device=idx.device)
                zero_val = torch.zeros((B, L, 1), dtype=torch.float, device=idx.device)

                active_offsets = torch.cat([zero_off, past_offsets], dim=-1)
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)

                # Strictly causal lookbacks
                targets = pos_grid - active_offsets
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                K_discrete = self.k_proj(state)
                V_discrete = self.v_proj(state)

                idx_gather = targets_clamped.view(B, L * self.k_peaks, 1).expand(-1, -1, d_model)
                K_cand = torch.gather(K_discrete, 1, idx_gather).view(B, L, self.k_peaks, d_model)
                V_cand = torch.gather(V_discrete, 1, idx_gather).view(B, L, self.k_peaks, d_model)

                q = self.ln_q(E_q_all + state)
                k = self.ln_k(K_cand)
                v = V_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals
                scores = scores.masked_fill(~valid_mask, -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                next_state = state + self.scale * context
                next_state = next_state + self.scale * self.mlp(self.ln_mlp(next_state))
                state = next_state

                if return_dynamics:
                    std_offsets = active_offsets[:, :, 1:].float().std(dim=1).mean().item()
                    diagnostics["offset_std_per_token"].append(std_offsets)
                    mean_dist = (weights * active_offsets.float()).sum(dim=-1).mean().item()
                    diagnostics["mean_attended_distances"].append(mean_dist)
                    ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
                    diagnostics["hop_entropies"].append(ent)

            logits = self.head(self.ln_f(state))
            if return_dynamics:
                return logits, diagnostics
            return logits

    # 3. Evaluation Function
    @torch.inference_mode()
    def evaluate(model, rng):
        model.eval()
        total_correct = 0
        total_eval = 0
        tier_correct = {"shallow": 0, "medium": 0, "deep": 0}
        tier_counts = {"shallow": 0, "medium": 0, "deep": 0}

        for _ in range(val_batches):
            x_val, y_val, depths = generate_dyck_batch(batch_size, rng)
            logits = model(x_val)
            pred = logits.argmax(dim=-1)

            mask = (y_val != -100)
            corr = (pred == y_val) & mask

            total_correct += corr.sum().item()
            total_eval += mask.sum().item()

            for b in range(batch_size):
                row_mask = mask[b]
                if not row_mask.any():
                    continue
                row_corr = corr[b, row_mask]
                row_depths = depths[b, row_mask]

                for c, d in zip(row_corr, row_depths):
                    val = int(c.item())
                    dep = int(d.item())
                    tier = "shallow" if dep <= 5 else ("medium" if dep <= 15 else "deep")
                    tier_correct[tier] += val
                    tier_counts[tier] += 1

        overall = 100.0 * total_correct / max(total_eval, 1)
        shallow = 100.0 * tier_correct["shallow"] / max(tier_counts["shallow"], 1)
        med = 100.0 * tier_correct["medium"] / max(tier_counts["medium"], 1)
        deep = 100.0 * tier_correct["deep"] / max(tier_counts["deep"], 1)

        return overall, shallow, med, deep

    # 4. Training
    train_rng = random.Random(seed)
    val_rng = random.Random(seed + 999)

    model = StateConditionedPerTokenDyck4().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    records = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _ = generate_dyck_batch(batch_size, train_rng)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == total_steps:
            elapsed = time.time() - t0
            overall, shallow, med, deep = evaluate(model, val_rng)
            print(
                f"  Step {step:04d}/{total_steps} | Loss: {loss.item():.4f} | "
                f"Acc: {overall:.2f}% (Shallow: {shallow:.2f}%, Med: {med:.2f}%, Deep: {deep:.2f}%) | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True
            )
            records.append({
                "step": step,
                "loss": round(loss.item(), 4),
                "overall_acc": round(overall, 2),
                "shallow_acc": round(shallow, 2),
                "med_acc": round(med, 2),
                "deep_acc": round(deep, 2),
                "elapsed_s": round(elapsed, 1),
            })

    # Validation diagnostics
    with torch.no_grad():
        x_val, _, _ = generate_dyck_batch(batch_size, val_rng)
        _, diag = model(x_val, return_dynamics=True)

    result = {
        "study": "S3-029",
        "description": "State-Conditioned Per-Token Wave Router on Dyck-4 at T=4 Hops",
        "seq_len": seq_len,
        "n_hops": n_hops,
        "k_peaks": k_peaks,
        "num_params": num_params,
        "total_time_s": round(time.time() - t0, 1),
        "final_overall_acc": records[-1]["overall_acc"],
        "final_shallow_acc": records[-1]["shallow_acc"],
        "final_med_acc": records[-1]["med_acc"],
        "final_deep_acc": records[-1]["deep_acc"],
        "records": records,
        "diagnostics": diag,
    }

    return result


@app.local_entrypoint()
def main():
    print(">>> Launching S3-029: State-Conditioned Per-Token Waves on Dyck-4 (T=4 Hops) on Modal A10G...")
    res = run_dyck4_per_token_benchmark.remote()
    print("\n" + "=" * 105)
    print(f"  COMPLETED S3-029: Final Overall Acc = {res['final_overall_acc']:.2f}%")
    print(f"  Shallow (1-5): {res['final_shallow_acc']:.2f}% | Medium (6-15): {res['final_med_acc']:.2f}% | Deep (16-30+): {res['final_deep_acc']:.2f}%")
    print(f"  Total Time: {res['total_time_s']}s")
    print("=" * 105)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "s3_029_dyck4_per_token_waves.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Saved result to {out_path}]")
