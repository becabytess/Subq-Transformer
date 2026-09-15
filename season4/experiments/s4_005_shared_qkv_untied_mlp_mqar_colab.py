"""
====================================================================================================
Study S4-005: Full 6,000-Step Delayed Convergence Test for Physical Multi-Layer SubQ
====================================================================================================
Hypothesis:
In S2-028 Delay4+Aggressive MLPs, recall remained at 2.58% (random chance) for 4,000 steps,
and then suddenly snapped into an aligned minimum at Step 5000 (3.69%) and Step 6000 (4.90%)!
Because deeper models have more non-linear degrees of freedom, they require the full 6,000-step
cosine annealing schedule to co-adapt and converge.

This script tests whether the 2-Physical-Layer SubQ (where NOTHING is tied: separate QKV and
separate MLPs per layer, 494k params) also breaks out in the second half of training!

Model Tested:
- subq_2layer_4hops_clean_physical (494,080 parameters)
  * Layer 1: Independent (W_q1, W_k1, W_v1) -> 4 optical hops (`state = context`, no W_o) -> MLP 1
  * Layer 2: Independent (W_q2, W_k2, W_v2) -> 4 optical hops (`state = context`, no W_o) -> MLP 2
  * Full 6,000 steps with CosineAnnealingLR (1e-3 to 1e-4)

Task: Corrected Multi-Query Associative Recall (MQAR)
- Sequence Length: 512, 16 pairs, 8 queries
- Random Exact-Answer Baseline: 2.50%
====================================================================================================
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

seq_len = 512
batch_size = 32
n_heads = 4
d_model = 128
d_mlp = 512
vocab_size = 256
num_pairs = 16
num_queries = 8
total_steps = 6000
eval_interval = 1000
query_marker = 1

offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]
start_q_pos = seq_len - (num_queries * 2) - 2


def make_batch(generator, dev):
    x = torch.randint(100, 255, (batch_size, seq_len), generator=generator, device="cpu")
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device="cpu")
    distances = torch.empty((batch_size, num_queries), dtype=torch.long, device="cpu")

    for row in range(batch_size):
        keys = torch.randperm(40, generator=generator)[:num_pairs] + 10
        values = torch.randperm(40, generator=generator)[:num_pairs] + 50
        kv = torch.randperm(350, generator=generator)[: num_pairs * 2].sort().values

        for pair in range(num_pairs):
            kp = int(kv[2 * pair].item())
            x[row, kp] = keys[pair]
            x[row, kp + 1] = values[pair]

        chosen = torch.randperm(num_pairs, generator=generator)[:num_queries]
        for qi, kt in enumerate(chosen):
            qpos = start_q_pos + 2 * qi
            kp = int(kv[2 * int(kt.item())].item())
            x[row, qpos] = query_marker
            x[row, qpos + 1] = keys[kt]
            y[row, qpos + 1] = values[kt]
            distances[row, qi] = qpos + 1 - kp

    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), distances.to(dev, non_blocking=True)


# ------------------------------------------------------------------------------
# 2-Physical-Layer SubQ (Completely Untied QKV, Completely Untied MLPs)
# ------------------------------------------------------------------------------
class SubQ_2Layer_Clean_Physical(nn.Module):
    def __init__(self, num_layers: int = 2, hops_per_layer: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.hops_per_layer = hops_per_layer

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)

        # Completely independent physical layers (NOTHING TIED)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_attn": nn.LayerNorm(d_model),
                "q": nn.Linear(d_model, d_model, bias=False),
                "k": nn.Linear(d_model, d_model, bias=False),
                "v": nn.Linear(d_model, d_model, bias=False),
                # NO W_o projection!
                "ln_mlp": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, d_mlp),
                    nn.GELU(),
                    nn.Linear(d_mlp, d_model),
                ),
            }) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
        valid = torch.zeros((seq_len, len(offsets)), dtype=torch.bool)
        for p in range(seq_len):
            for i, off in enumerate(offsets):
                src = p - off
                if src >= 0:
                    targets[p, i] = src
                    valid[p, i] = True
        self.register_buffer("targets", targets)
        self.register_buffer("valid", valid)
        self.scale = 1.0 / math.sqrt(d_model // n_heads)

    def forward(self, idx):
        bsz, length = idx.shape
        state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
        indices = self.targets[:length]
        valid_mask = self.valid[:length].view(1, 1, length, len(offsets))

        for layer in self.layers:
            # 4 linear optical transport hops per layer
            for _ in range(self.hops_per_layer):
                z = layer["ln_attn"](state)
                q = layer["q"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = layer["k"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = layer["v"](z).view(bsz, length, n_heads, -1).transpose(1, 2)

                scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * self.scale
                weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                # Pure optical replacement, NO W_o, NO additive residual
                state = context.transpose(1, 2).contiguous().view(bsz, length, d_model)

            # 1 MLP executed at the end of each physical layer
            state = state + layer["mlp"](layer["ln_mlp"](state))

        return self.head(self.ln_f(state))


# ------------------------------------------------------------------------------
# Evaluation & Training Routine
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mqar(model, num_batches=20, eval_seed=2026):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(eval_seed)
    total_correct = total_queries = 0
    bins = {"128-255": [0, 0], "256-383": [0, 0], "384-511": [0, 0]}

    for _ in range(num_batches):
        x, y, distances = make_batch(gen, device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        mask = y != -100
        correct = (preds == y) & mask
        total_correct += correct.sum().item()
        total_queries += mask.sum().item()

        for b in range(batch_size):
            for q in range(num_queries):
                qpos = start_q_pos + 2 * q + 1
                dist = distances[b, q].item()
                is_corr = (preds[b, qpos] == y[b, qpos]).item()
                if 128 <= dist <= 255:
                    bins["128-255"][1] += 1
                    if is_corr: bins["128-255"][0] += 1
                elif 256 <= dist <= 383:
                    bins["256-383"][1] += 1
                    if is_corr: bins["256-383"][0] += 1
                elif 384 <= dist <= 511:
                    bins["384-511"][1] += 1
                    if is_corr: bins["384-511"][0] += 1

    mean_acc = (total_correct / max(1, total_queries)) * 100.0
    bin_accs = {k: (v[0] / max(1, v[1])) * 100.0 for k, v in bins.items()}
    return mean_acc, bin_accs


def train_model(model_name: str, model_builder):
    print("\n" + "=" * 90)
    print(f"  STARTING TRAINING: {model_name} (FULL 6,000 STEPS)")
    print("=" * 90)

    model = model_builder().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
    train_gen = torch.Generator(device="cpu").manual_seed(seed + 100)
    start_time = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        x, y, _ = make_batch(train_gen, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        if step % eval_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            mean_acc, bin_accs = evaluate_mqar(model)
            model.train()
            print(f"  [{model_name}] Step {step:4d}/{total_steps} | Loss: {loss.item():.4f} | "
                  f"Recall: {mean_acc:5.2f}% | "
                  f"[128-255: {bin_accs['128-255']:4.1f}% | 256-383: {bin_accs['256-383']:4.1f}% | 384-511: {bin_accs['384-511']:4.1f}%] | "
                  f"Time: {elapsed:.1f}s")

    final_acc, final_bins = evaluate_mqar(model, num_batches=50)
    print(f"\n>>> [{model_name}] FINAL SCORE: {final_acc:.2f}% "
          f"(128-255: {final_bins['128-255']:.2f}%, 256-383: {final_bins['256-383']:.2f}%, 384-511: {final_bins['384-511']:.2f}%)\n")

    return {"params": param_count, "final_acc": final_acc, "bins": final_bins, "time_seconds": time.time() - start_time}


# ------------------------------------------------------------------------------
# Main Benchmark Queue: Runs Pure Untied Physical Multi-Layer SubQ
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    experiments = {
        "subq_2layer_4hops_clean_physical": lambda: SubQ_2Layer_Clean_Physical(num_layers=2, hops_per_layer=4),
    }

    models_to_test = [
        "subq_2layer_4hops_clean_physical",
    ]

    all_results = {}
    for name in models_to_test:
        res = train_model(name, experiments[name])
        all_results[name] = res

    print("\n" + "=" * 95)
    print("  FINAL SCORECARD")
    print("=" * 95)
    print(f"{'Model Name':<34} | {'Params':<10} | {'Overall':<9} | {'128-255':<9} | {'256-383':<9} | {'384-511':<9}")
    print("-" * 95)
    for name, r in all_results.items():
        print(f"{name:<34} | {r['params']:<10,d} | {r['final_acc']:5.2f}%   | {r['bins']['128-255']:5.2f}%   | {r['bins']['256-383']:5.2f}%   | {r['bins']['384-511']:5.2f}%")
    print("=" * 95)
