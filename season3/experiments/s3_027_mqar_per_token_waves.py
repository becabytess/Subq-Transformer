"""S3-027: State-Conditioned Per-Token Wave Router on MQAR Associative Recall (T=4 Hops).

Scientific Hypothesis:
In S3-018 and S3-025, Inverted Query-Escrow failed MQAR (~2.5% random accuracy) because
a single global wave router emitted the same set of 8 peak offsets for all tokens in the sequence,
making it mathematically impossible to retrieve from 8 distinct arbitrary key positions simultaneously.

With State-Conditioned Per-Token Waves:
  [A_i, omega_i, phi_i, lambda_i] = WaveHead(LN(s_i^(t)))
Each query token emits its own personal wave, allowing each of the 8 query positions to independently
point its continuous harmonic radar at its specific key-value pair in the prefix.

Benchmark:
- Sequence Length L = 512, Batch Size = 32, Dim D = 128
- 16 random Key-Value pairs placed in the first 350 tokens
- 8 queries placed at the end of the sequence
- Hops T = 4, K = 8 Peaks, Total Steps = 3,000
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season3-s3-027-mqar-per-token-waves")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_mqar_per_token_experiment(
    seed: int = 42,
    total_steps: int = 3000,
    eval_interval: int = 250,
    val_batches: int = 30,
    seq_len: int = 512,
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

    vocab_size = 256
    num_pairs = 16
    num_queries = 8
    query_marker = 1
    max_d = seq_len
    start_q_pos = seq_len - (num_queries * 2) - 2
    query_positions = [start_q_pos + 2 * i + 1 for i in range(num_queries)]

    print("=" * 105)
    print("  S3-027: STATE-CONDITIONED PER-TOKEN WAVE ROUTER ON MQAR (T=4 HOPS)")
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

    # State-Conditioned Per-Token Wave Inverted Query-Escrow Model
    class StateConditionedPerTokenWaveMQAR(nn.Module):
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

            mlp_dim = d_model * mlp_ratio
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model),
            )

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, return_dynamics: bool = False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Phase 1: Causal Scan -> Query Escrow Vault
            H, _ = self.scanner(x)
            G = torch.sigmoid(self.gate(H))
            Q_raw = self.q_proj(G * H)
            E_q_all = torch.cumsum(Q_raw, dim=1)

            # Phase 2: Multi-Hop Retrieval with Per-Token State-Conditioned Waves
            state = H
            pos_grid = torch.arange(L, device=idx.device).view(1, L, 1)

            diagnostics = {
                "offset_std_per_token": [],
                "mean_attended_distances": [],
            }

            for hop in range(self.n_hops):
                wave_params = self.wave_head(self.ln_wave(state)).view(B, L, self.num_waves, 4)
                amp = torch.tanh(wave_params[..., 0]).unsqueeze(-1)
                omega = (F.softplus(wave_params[..., 1]).unsqueeze(-1) * self.base_freqs)
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

                targets = pos_grid - active_offsets
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                K_discrete = self.k_proj(state)
                V_discrete = self.v_proj(state)

                batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(B, L, self.k_peaks)
                K_cand = K_discrete[batch_idx, targets_clamped]
                V_cand = V_discrete[batch_idx, targets_clamped]

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

            logits = self.head(self.ln_f(state))
            if return_dynamics:
                return logits, diagnostics
            return logits

    # Evaluation
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

    # Training
    train_gen = torch.Generator(device=device).manual_seed(seed)
    val_gen = torch.Generator(device=device).manual_seed(seed + 999)

    model = StateConditionedPerTokenWaveMQAR().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    records = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _ = make_batch(train_gen)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == total_steps:
            elapsed = time.time() - t0
            overall, shallow, med, long = evaluate(model, val_gen)
            print(
                f"  Step {step:04d}/{total_steps} | Loss: {loss.item():.4f} | "
                f"Exact Recall: {overall:.2f}% (Shallow: {shallow:.1f}%, Med: {med:.1f}%, Long: {long:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s"
            )
            records.append({
                "step": step,
                "loss": round(loss.item(), 4),
                "overall_acc": round(overall, 2),
                "shallow_acc": round(shallow, 2),
                "med_acc": round(med, 2),
                "long_acc": round(long, 2),
                "elapsed_s": round(elapsed, 1),
            })

    with torch.no_grad():
        x_val, _, _ = make_batch(val_gen)
        _, diag = model(x_val, return_dynamics=True)

    result = {
        "study": "S3-027",
        "description": "State-Conditioned Per-Token Wave Router on MQAR at T=4 Hops",
        "seq_len": seq_len,
        "n_hops": n_hops,
        "k_peaks": k_peaks,
        "num_params": num_params,
        "total_time_s": round(time.time() - t0, 1),
        "final_overall_acc": records[-1]["overall_acc"],
        "final_shallow_acc": records[-1]["shallow_acc"],
        "final_med_acc": records[-1]["med_acc"],
        "final_long_acc": records[-1]["long_acc"],
        "records": records,
        "diagnostics": diag,
    }

    return result


@app.local_entrypoint()
def main():
    print(">>> Launching S3-027: State-Conditioned Per-Token Waves on MQAR (T=4 Hops) on Modal A10G...")
    res = run_mqar_per_token_experiment.remote()
    print("\n" + "=" * 105)
    print(f"  COMPLETED S3-027: Final Exact Recall = {res['final_overall_acc']:.2f}%")
    print(f"  Shallow: {res['final_shallow_acc']:.1f}% | Medium: {res['final_med_acc']:.1f}% | Long: {res['final_long_acc']:.1f}%")
    print(f"  Total Time: {res['total_time_s']}s")
    print("=" * 105)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "s3_027_mqar_per_token_waves.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Saved result to {out_path}]")
