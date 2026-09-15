"""S3-019: Fast Pure Feature-Escrow Network (roll_nodep) on Multi-Query Associative Recall (MQAR).

Evaluates whether Pure FEN alone—using cuDNN temporal scan to generate hidden states H,
parallel speculative feature extraction V and roll gate gamma, and the non-commutative
channel-roll conveyor belt (torch.roll)—can solve Multi-Query Associative Recall (L=512).

Architecture:
  - Temporal Scanner: H, _ = GRU(x)  [cuDNN fused kernel]
  - Parallel Extraction: G = sigmoid(gate(H)), V = v_proj(G * H)
  - Parallel Roll Gate: gamma = sigmoid(roll_gate(H))
  - Channel-Roll Conveyor: E_t = (1 - gamma_t) * E_{t-1} + gamma_t * roll(E_{t-1}, 1, -1) + V_t
  - Joint Readout Head: logits = Head([H, E])

Benchmark:
  - Sequence Length L = 512, Dim D = 128, 16 pairs, 8 queries
  - Steps: 3,000 (AdamW lr=1e-3, cosine decay)
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
app = modal.App("season3-s3-019-pure-fen-mqar")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_pure_fen_mqar(
    seed: int = 42,
    total_steps: int = 3000,
    eval_interval: int = 500,
    val_batches: int = 30,
    seq_len: int = 512,
    batch_size: int = 32,
    d_model: int = 128,
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
    start_q_pos = seq_len - (num_queries * 2) - 2
    query_positions = [start_q_pos + 2 * i + 1 for i in range(num_queries)]

    print("=" * 95)
    print("  S3-019: FAST PURE FEATURE-ESCROW NETWORK (roll_nodep) ON MQAR ASSOCIATIVE RECALL")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, 16 Pairs, 8 Queries, Steps = {total_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

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

    # Fast Pure FEN roll_nodep Model
    class FastPureFENRollNoDep(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Built-in fast cuDNN temporal scanner
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)

            # Parallel FEN feature extraction & roll gate
            self.gate = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.roll_gate = nn.Linear(d_model, 1)

            # Joint Readout Head over [H, E]
            self.head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size),
            )

        def forward(self, idx, return_stats: bool = False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # 1. Fast built-in cuDNN temporal scan
            H, _ = self.scanner(x)  # [B, L, D]

            # 2. Parallel FEN extraction across all tokens simultaneously
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            V = self.v_proj(G * H)                     # [B, L, D]
            gamma = torch.sigmoid(self.roll_gate(H))   # [B, L, 1]

            # 3. Escrow Channel-Roll conveyor belt (pure tensor ops, zero weights in loop)
            E = torch.zeros(B, self.d_model, device=idx.device)
            E_list = []
            for t in range(L):
                E = (1.0 - gamma[:, t]) * E + gamma[:, t] * torch.roll(E, shifts=1, dims=-1) + V[:, t]
                E_list.append(E)

            E_seq = torch.stack(E_list, dim=1)  # [B, L, D]

            # 4. Joint Decision Head over [H, E]
            logits = self.head(torch.cat([H, E_seq], dim=-1))

            if return_stats:
                stats = {
                    "gamma": float(gamma.mean().item()),
                    "gate": float(G.mean().item()),
                    "pipe_norm": float(H.detach().norm(dim=-1).mean().item()),
                    "escrow_norm": float(E.detach().norm(dim=-1).mean().item()),
                }
                return logits, stats
            return logits

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

                    if d < 64:
                        tier_counts["shallow"] += 1
                        if is_correct:
                            tier_correct["shallow"] += 1
                    elif d <= 128:
                        tier_counts["medium"] += 1
                        if is_correct:
                            tier_correct["medium"] += 1
                    else:
                        tier_counts["long"] += 1
                        if is_correct:
                            tier_correct["long"] += 1

        acc = 100.0 * total_correct / max(1, total_queries_eval)
        shallow_acc = 100.0 * tier_correct["shallow"] / max(1, tier_counts["shallow"]) if tier_counts["shallow"] > 0 else 0.0
        med_acc = 100.0 * tier_correct["medium"] / max(1, tier_counts["medium"]) if tier_counts["medium"] > 0 else 0.0
        long_acc = 100.0 * tier_correct["long"] / max(1, tier_counts["long"]) if tier_counts["long"] > 0 else 0.0

        return {
            "acc": acc,
            "shallow": shallow_acc,
            "medium": med_acc,
            "long": long_acc,
            "total_eval": total_queries_eval,
        }

    # Instantiate Model
    train_gen = torch.Generator(device=device).manual_seed(seed)
    val_gen = torch.Generator(device=device).manual_seed(seed + 1000)

    model = FastPureFENRollNoDep().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model: Fast Pure FEN (roll_nodep)")
    print(f"  Total Parameters: {param_count:,}")
    print("-" * 95)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_lr(step):
        if step < 100:
            return 1e-3 * (step + 1) / 100
        progress = (step - 100) / max(1, total_steps - 100)
        return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1.0 + math.cos(math.pi * progress))

    start_time = time.time()
    step_records = []

    for step in range(1, total_steps + 1):
        model.train()
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        x_b, y_b, _ = make_batch(train_gen)
        optimizer.zero_grad()

        is_eval_step = (step % eval_interval == 0 or step == total_steps)
        if is_eval_step:
            logits, stats = model(x_b, return_stats=True)
        else:
            logits = model(x_b, return_stats=False)

        loss = F.cross_entropy(logits.view(-1, vocab_size), y_b.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_eval_step:
            metrics = evaluate(model, val_gen)
            elapsed = time.time() - start_time
            print(
                f"  Step {step:04d}/{total_steps} | Loss: {loss.item():.4f} | "
                f"Exact Recall: {metrics['acc']:.2f}% "
                f"(Shallow: {metrics['shallow']:.1f}%, Med: {metrics['medium']:.1f}%, Long: {metrics['long']:.1f}%) | "
                f"Gate: {stats['gate']:.3f}, Gamma: {stats['gamma']:.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            step_records.append({
                "step": step,
                "loss": float(loss.item()),
                "acc": metrics["acc"],
                "shallow": metrics["shallow"],
                "medium": metrics["medium"],
                "long": metrics["long"],
                "gate": stats["gate"],
                "gamma": stats["gamma"],
                "pipe_norm": stats["pipe_norm"],
                "escrow_norm": stats["escrow_norm"],
                "elapsed": elapsed,
            })

    final_metrics = evaluate(model, val_gen)
    total_time = time.time() - start_time
    print("=" * 95)
    print(f"  FINAL RESULT: Fast Pure FEN (roll_nodep)")
    print(f"  Exact Recall: {final_metrics['acc']:.2f}%")
    print(f"  Distance Breakdown: Shallow={final_metrics['shallow']:.2f}%, Medium={final_metrics['medium']:.2f}%, Long={final_metrics['long']:.2f}%")
    print(f"  Total Time: {total_time:.1f}s")
    print("=" * 95)

    return {
        "model": "Fast Pure FEN (roll_nodep)",
        "param_count": param_count,
        "final_acc": final_metrics["acc"],
        "final_metrics": final_metrics,
        "step_records": step_records,
        "total_time": total_time,
    }


@app.local_entrypoint()
def main():
    print("Launching Fast Pure FEN (roll_nodep) MQAR associative recall experiment on Modal A10G...")
    result = run_pure_fen_mqar.remote()

    os.makedirs("season3/results", exist_ok=True)
    out_path = "season3/results/s3_019_pure_fen_mqar.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] Experiment results saved to {out_path}")
