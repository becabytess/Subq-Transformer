"""
====================================================================================================
Study S4-003: Multi-Physical-Layer SubQ with Fixed Residuals on Corrected MQAR
====================================================================================================
Investigates the Multi-Layer SubQ Residual Hypothesis:
Does fixing the residual connections (Pre-LN additive highways instead of hard state replacement)
and giving each layer a sufficient linear transport window (e.g. 4 or 8 hops before the MLP)
rescue Multi-Physical-Layer SubQ from the 2.60% collapse observed in S2-028?

Task: Corrected Multi-Query Associative Recall (MQAR)
- Sequence Length: 512
- 16 independently randomized (key, value) pairs
- 8 late queries with randomized placement
- Random Exact-Answer Baseline: 2.50% (1/40 uniform random chance)

Models Evaluated:
1. dense_1layer: Standard 1-layer Dense Transformer (Historical reference: 5.45%)
2. subq_1layer_8hops: 1 physical layer, 8 linear hops with residual -> 1 final MLP (Historical: 5.79% - 6.32%)
3. subq_4layer_2hops_broken_control: Replicating S2-028's broken replacement update (state = context) -> Expected 2.60%
4. subq_4layer_2hops_fixed_residuals: 4 physical layers, 2 hops each with TRUE Pre-LN additive residuals
5. subq_2layer_4hops_fixed_residuals: 2 physical layers, 4 hops each (full global reach before MLP) with TRUE residuals
6. subq_2layer_8hops_fixed_residuals: 2 physical layers, 8 hops each with TRUE residuals

Designed to be completely self-contained for single-click execution in Google Colab (GPU).
====================================================================================================
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# ------------------------------------------------------------------------------
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
total_steps = 6000  # Set to 3000 for faster screening, 6000 matches S2-028 exactly
eval_interval = 1000
query_marker = 1

# Complete K=9 basis (reaches all distances 0..512 within 8 hops):
offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]
start_q_pos = seq_len - (num_queries * 2) - 2

# ------------------------------------------------------------------------------
# 2. Fast Vectorized MQAR Batch Generator
# ------------------------------------------------------------------------------
def make_batch(generator, dev):
    # Base fill with random noise tokens [100, 254]
    x = torch.randint(100, 255, (batch_size, seq_len), generator=generator, device="cpu")
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device="cpu")
    distances = torch.empty((batch_size, num_queries), dtype=torch.long, device="cpu")

    for row in range(batch_size):
        # 16 random keys from [10, 49], 16 random values from [50, 89]
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
# 3. Model Architectures
# ------------------------------------------------------------------------------

# Reference 1: Standard 1-Layer Dense Transformer
class Dense1Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)))
        self.scale = 1.0 / math.sqrt(d_model // n_heads)

    def forward(self, idx):
        bsz, length = idx.shape
        state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
        causal_mask = self.causal[:length, :length]

        z = self.ln_attn(state)
        q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
        k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
        v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~causal_mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        msg = self.o((weights @ v).transpose(1, 2).contiguous().view(bsz, length, d_model))

        state = state + msg
        state = state + self.mlp(self.ln_mlp(state))
        return self.head(self.ln_f(state))


# Modular SubQ Multi-Layer Engine supporting both broken and fixed residual dynamics
class SubQDepthModel(nn.Module):
    def __init__(self, num_layers: int, hops_per_layer: int, broken_residuals: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.hops_per_layer = hops_per_layer
        self.broken_residuals = broken_residuals

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_attn": nn.LayerNorm(d_model),
                "q": nn.Linear(d_model, d_model, bias=False),
                "k": nn.Linear(d_model, d_model, bias=False),
                "v": nn.Linear(d_model, d_model, bias=False),
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

        # Precompute causal candidate index buffer for Complete K=9 basis
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
        self.inv_sqrt_H = 1.0 / math.sqrt(self.hops_per_layer)

    def forward(self, idx):
        bsz, length = idx.shape
        state = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
        indices = self.targets[:length]
        valid_mask = self.valid[:length].view(1, 1, length, len(offsets))

        for layer in self.layers:
            # Loop over linear transport hops for this layer
            s = state
            for _ in range(self.hops_per_layer):
                z = layer["ln_attn"](s)
                q = layer["q"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                k = layer["k"](z).view(bsz, length, n_heads, -1).transpose(1, 2)
                v = layer["v"](z).view(bsz, length, n_heads, -1).transpose(1, 2)

                # Direct tensor gather along candidates: [bsz, n_heads, length, K, head_dim]
                scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * self.scale
                weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
                context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
                context = context.transpose(1, 2).contiguous().view(bsz, length, d_model)

                if self.broken_residuals:
                    # S2-028 Flaw: Hard state replacement! Wipes out input state!
                    s = context
                else:
                    # Fixed Pre-LN additive highway:
                    s = s + self.inv_sqrt_H * context

            # 1 MLP executed at the end of the layer
            if self.broken_residuals:
                # S2-028 Flaw: State only adds MLP to the replaced context, no outer skip
                state = s + layer["mlp"](layer["ln_mlp"](s))
            else:
                # Fixed Pre-LN additive residual:
                s = s + layer["mlp"](layer["ln_mlp"](s))
                state = s  # Identity highway from layer input is preserved via s

        return self.head(self.ln_f(state))


# ------------------------------------------------------------------------------
# 4. Evaluation Function (Distance-Binned Metrics)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mqar(model, num_batches=20, eval_seed=2026):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(eval_seed)
    
    total_correct = 0
    total_queries = 0
    bins = {
        "128-255": [0, 0],
        "256-383": [0, 0],
        "384-511": [0, 0],
    }

    for _ in range(num_batches):
        x, y, distances = make_batch(gen, device)
        logits = model(x)
        preds = logits.argmax(dim=-1)

        mask = y != -100
        correct = (preds == y) & mask
        
        total_correct += correct.sum().item()
        total_queries += mask.sum().item()

        # Distance bin profiling
        for b in range(x.shape[0]):
            for qi in range(num_queries):
                qpos = start_q_pos + 2 * qi + 1
                dist = distances[b, qi].item()
                is_corr = (preds[b, qpos] == y[b, qpos]).item()

                if 128 <= dist < 256:
                    bins["128-255"][0] += int(is_corr)
                    bins["128-255"][1] += 1
                elif 256 <= dist < 384:
                    bins["256-383"][0] += int(is_corr)
                    bins["256-383"][1] += 1
                elif 384 <= dist <= 512:
                    bins["384-511"][0] += int(is_corr)
                    bins["384-511"][1] += 1

    mean_acc = (total_correct / max(1, total_queries)) * 100.0
    bin_accs = {
        k: (v[0] / max(1, v[1])) * 100.0 for k, v in bins.items()
    }
    return mean_acc, bin_accs


# ------------------------------------------------------------------------------
# 5. Training Engine
# ------------------------------------------------------------------------------
def train_model(model_name: str, build_fn):
    print("\n" + "=" * 90)
    print(f"  STARTING TRAINING: {model_name}")
    print("=" * 90)

    torch.manual_seed(seed)
    model = build_fn().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Trainable Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Cosine annealing down to 1e-5
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    train_gen = torch.Generator(device="cpu").manual_seed(seed + 100)
    start_time = time.time()

    model.train()
    for step in range(1, total_steps + 1):
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

    return {
        "params": param_count,
        "final_acc": final_acc,
        "bins": final_bins,
        "time_seconds": time.time() - start_time
    }


# ------------------------------------------------------------------------------
# 6. Main Execution Loop
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    experiments = {
        # 1. Reference 1-Layer Dense Transformer
        "dense_1layer": lambda: Dense1Layer(),

        # 2. SubQ 1-Layer, 8 Linear Hops -> 1 Final MLP (The Proven 5.8% Baseline)
        "subq_1layer_8hops": lambda: SubQDepthModel(num_layers=1, hops_per_layer=8, broken_residuals=False),

        # 3. SubQ 4-Layer, 2 Hops per Layer (S2-028 Broken Control: Expected Collapse to ~2.60%)
        "subq_4layer_2hops_broken_control": lambda: SubQDepthModel(num_layers=4, hops_per_layer=2, broken_residuals=True),

        # 4. SubQ 4-Layer, 2 Hops per Layer (Fixed Pre-LN Additive Residuals)
        "subq_4layer_2hops_fixed_residuals": lambda: SubQDepthModel(num_layers=4, hops_per_layer=2, broken_residuals=False),

        # 5. SubQ 2-Layer, 4 Hops per Layer (Full Receptive Field Before Each MLP!)
        "subq_2layer_4hops_fixed_residuals": lambda: SubQDepthModel(num_layers=2, hops_per_layer=4, broken_residuals=False),

        # 6. SubQ 2-Layer, 8 Hops per Layer (Deep Macro Transport + Multi-Layer)
        "subq_2layer_8hops_fixed_residuals": lambda: SubQDepthModel(num_layers=2, hops_per_layer=8, broken_residuals=False),
    }

    # Execution queue: New fixed models run FIRST so you can stop early!
    models_to_test = [
        "subq_4layer_2hops_fixed_residuals",  # <-- NEW: 4-Layer SubQ with fixed Pre-LN residuals! (Runs 1st)
        "subq_2layer_4hops_fixed_residuals",  # <-- NEW: 2-Layer SubQ (4 hops = full reach before MLP) (Runs 2nd)
        "subq_2layer_8hops_fixed_residuals",  # <-- NEW: 2-Layer SubQ (8 hops per layer) (Runs 3rd)
        "subq_4layer_2hops_broken_control",   # Control: S2-028 broken replacement (Runs 4th)
        "subq_1layer_8hops",                  # Control: 1-Layer baseline (Runs 5th)
        "dense_1layer",                       # Control: Dense baseline (Runs 6th)
    ]

    all_results = {}
    for name in models_to_test:
        res = train_model(name, experiments[name])
        all_results[name] = res

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print("  FINAL S4-003 EXPERIMENT SCORECARD (MQAR Sequence Length 512, 16 Pairs)")
    print("=" * 95)
    print(f"{'Model Name':<36} | {'Params':<10} | {'Overall':<9} | {'128-255':<9} | {'256-383':<9} | {'384-511':<9}")
    print("-" * 95)
    for name, r in all_results.items():
        print(f"{name:<36} | {r['params']:<10,d} | {r['final_acc']:5.2f}%   | {r['bins']['128-255']:5.2f}%   | {r['bins']['256-383']:5.2f}%   | {r['bins']['384-511']:5.2f}%")
    print("=" * 95)
    print("Random Chance Baseline: 2.50%")
