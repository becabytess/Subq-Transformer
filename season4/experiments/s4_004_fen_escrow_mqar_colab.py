"""
====================================================================================================
Study S4-004: Post-MLP Feature Escrow Network (FEN) SubQ on Corrected MQAR
====================================================================================================
Implements the exact Post-MLP FEN Escrow architecture from Study 75 / Season 3:
1. Recurrent Proposal: Working state executes Attention + Per-Hop 4x MLP (s_prop)
2. FEN Intestinal Extraction: Gated absorption from processed state:
   D_escrow = sigmoid(W_gate(s_prop)) * s_prop
   v_escrow = W_v(D_escrow)
3. FEN Channel-Roll Conveyor Belt:
   E = (1 - gamma) * E + gamma * Roll(E, shift=1) + v_escrow
4. Working State Progression:
   - No Depletion: s = s_prop
   - With Depletion: s = (1 - g_escrow) * s_prop
5. Joint Decision Readout: Head([LN(s_final), LN(E_final)])

Task: Corrected Multi-Query Associative Recall (MQAR)
- Sequence Length: 512
- 16 independently randomized (key, value) pairs
- 8 late queries with randomized placement
- Random Exact-Answer Baseline: 2.50% (1/40 uniform random chance)

Models Evaluated:
1. fen_post_mlp_roll_nodep:   Exact Study 75 FEN Post-MLP Escrow (Deplete OFF) (Runs 1st)
2. fen_post_mlp_roll_deplete: Exact Study 75 FEN Post-MLP Escrow (Deplete ON)  (Runs 2nd)
3. subq_4hops_linear_control: Linear multi-hop transport -> 1 Final MLP (Proven Baseline ~5.5%)
4. subq_4hops_unshielded_mlp: Per-Hop 4x MLP, NO Escrow (Unshielded Control ~2.5%)
5. dense_1layer:              Standard 1-Layer Dense Transformer Baseline (~5.45%)

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
total_steps = 6000  # 6000 matches S2-028 & S4-003 exactly
eval_interval = 1000
query_marker = 1

# Complete K=9 basis (reaches all distances 0..512 within 4 to 8 hops):
offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]
start_q_pos = seq_len - (num_queries * 2) - 2

# ------------------------------------------------------------------------------
# 2. Fast Vectorized MQAR Batch Generator
# ------------------------------------------------------------------------------
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
# 3. Model Architectures
# ------------------------------------------------------------------------------

# Reference Baseline: Standard 1-Layer Dense Transformer
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


# EXACT Post-MLP Feature Escrow Network (FEN) Architecture (Study 75 / Season 3)
class SubQ_FEN_PostMLP_Model(nn.Module):
    def __init__(self, n_hops: int = 4, deplete: bool = False, roll_shift: int = 1):
        super().__init__()
        self.n_hops = n_hops
        self.deplete = deplete
        self.roll_shift = roll_shift
        self.inv_sqrt_H = 1.0 / math.sqrt(n_hops)

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)

        # Multi-Hop Attention
        self.ln_attn = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Per-Hop 4x MLP
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

        # Exact FEN Post-MLP Escrow Modules (Study 75)
        self.fen_gate = nn.Linear(d_model, d_model)
        self.fen_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.fen_roll_gate = nn.Linear(d_model, 1)

        self.ln_s = nn.LayerNorm(d_model)
        self.ln_e = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * 2, vocab_size, bias=False)

        # Precompute causal candidate buffer
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
        s = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
        indices = self.targets[:length]
        valid_mask = self.valid[:length].view(1, 1, length, len(offsets))
        E = torch.zeros_like(s)

        for hop in range(self.n_hops):
            # 1. Multi-Hop Attention
            z = self.ln_attn(s)
            q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)

            scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * self.scale
            weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
            raw_context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
            c = raw_context.transpose(1, 2).contiguous().view(bsz, length, d_model)
            msg = self.o(c)

            # 2. Recurrent Proposal (Attention + MLP) - Step 6 of Study 75
            s_prop = s + self.inv_sqrt_H * msg
            s_prop = s_prop + self.inv_sqrt_H * self.mlp(self.ln_mlp(s_prop))

            # 3. FEN Feature Escrow Absorption - Step 7 of Study 75
            g_escrow = torch.sigmoid(self.fen_gate(s_prop))
            D_escrow = g_escrow * s_prop
            v_escrow = self.fen_v_proj(D_escrow)

            if self.deplete:
                s = (1.0 - g_escrow) * s_prop
            else:
                s = s_prop

            # Channel-Roll Escrow Vault Update
            gamma = torch.sigmoid(self.fen_roll_gate(s_prop))  # [B, L, 1]
            E = (1.0 - gamma) * E + gamma * torch.roll(E, shifts=self.roll_shift, dims=-1) + v_escrow

        # Joint Readout Head
        return self.head(torch.cat([self.ln_s(s), self.ln_e(E)], dim=-1))


# Control Models: Unshielded & Linear
class SubQ_Control_Model(nn.Module):
    def __init__(self, n_hops: int = 4, mlp_interval: int = 1):
        super().__init__()
        self.n_hops = n_hops
        self.mlp_interval = mlp_interval
        self.inv_sqrt_H = 1.0 / math.sqrt(n_hops)

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
        self.ln_s = nn.LayerNorm(d_model)
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
        s = self.tok(idx) + self.pos(torch.arange(length, device=idx.device).unsqueeze(0))
        indices = self.targets[:length]
        valid_mask = self.valid[:length].view(1, 1, length, len(offsets))

        for hop in range(self.n_hops):
            z = self.ln_attn(s)
            q = self.q(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            k = self.k(z).view(bsz, length, n_heads, -1).transpose(1, 2)
            v = self.v(z).view(bsz, length, n_heads, -1).transpose(1, 2)

            scores = (q.unsqueeze(3) * k[:, :, indices, :]).sum(-1) * self.scale
            weights = F.softmax(scores.masked_fill(~valid_mask, -1e4), dim=-1)
            raw_context = (weights.unsqueeze(-1) * v[:, :, indices, :]).sum(3)
            c = raw_context.transpose(1, 2).contiguous().view(bsz, length, d_model)
            s = s + self.inv_sqrt_H * self.o(c)

            if self.mlp_interval == 1:
                s = s + self.inv_sqrt_H * self.mlp(self.ln_mlp(s))

        if self.mlp_interval == 0:
            s = s + self.mlp(self.ln_mlp(s))

        return self.head(self.ln_s(s))


# ------------------------------------------------------------------------------
# 4. Evaluation Function
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
    bin_accs = {
        k: (v[0] / max(1, v[1])) * 100.0 for k, v in bins.items()
    }
    return mean_acc, bin_accs


# ------------------------------------------------------------------------------
# 5. Training Routine
# ------------------------------------------------------------------------------
def train_model(model_name: str, model_builder):
    print("\n" + "=" * 90)
    print(f"  STARTING TRAINING: {model_name}")
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

    return {
        "params": param_count,
        "final_acc": final_acc,
        "bins": final_bins,
        "time_seconds": time.time() - start_time
    }


# ------------------------------------------------------------------------------
# 6. Main Benchmark Queue
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    experiments = {
        # 1. EXACT FEN Post-MLP Escrow (Deplete OFF: roll_nodep) (Runs 1st)
        "fen_post_mlp_roll_nodep": lambda: SubQ_FEN_PostMLP_Model(n_hops=4, deplete=False, roll_shift=1),

        # 2. EXACT FEN Post-MLP Escrow (Deplete ON: roll_dep) (Runs 2nd)
        "fen_post_mlp_roll_deplete": lambda: SubQ_FEN_PostMLP_Model(n_hops=4, deplete=True, roll_shift=1),

        # 3. CONTROL: Linear 4 Hops -> 1 Final MLP (Proven Linear Baseline ~5.5%)
        "subq_4hops_linear_control": lambda: SubQ_Control_Model(n_hops=4, mlp_interval=0),

        # 4. CONTROL: Per-Hop 4x MLP, NO Escrow (Unshielded Control ~2.5%)
        "subq_4hops_unshielded_mlp": lambda: SubQ_Control_Model(n_hops=4, mlp_interval=1),

        # 5. CONTROL: 1-Layer Dense Transformer Baseline (Historical ~5.45%)
        "dense_1layer": lambda: Dense1Layer(),
    }

    # Queue runs the exact Post-MLP FEN models first!
    models_to_test = [
        "fen_post_mlp_roll_nodep",     # Study 75 Post-MLP Escrow (deplete OFF)
        "fen_post_mlp_roll_deplete",   # Study 75 Post-MLP Escrow (deplete ON)
        "subq_4hops_linear_control",   # Linear Control
        "subq_4hops_unshielded_mlp",   # Unshielded Control
        "dense_1layer",                # Dense Baseline
    ]

    all_results = {}
    for name in models_to_test:
        res = train_model(name, experiments[name])
        all_results[name] = res

    # Final Summary Scorecard
    print("\n" + "=" * 95)
    print("  FINAL S4-004 EXPERIMENT SCORECARD (MQAR Sequence Length 512, 16 Pairs, 8 Queries)")
    print("=" * 95)
    print(f"{'Model Name':<34} | {'Params':<10} | {'Overall':<9} | {'128-255':<9} | {'256-383':<9} | {'384-511':<9}")
    print("-" * 95)
    for name, r in all_results.items():
        print(f"{name:<34} | {r['params']:<10,d} | {r['final_acc']:5.2f}%   | {r['bins']['128-255']:5.2f}%   | {r['bins']['256-383']:5.2f}%   | {r['bins']['384-511']:5.2f}%")
    print("=" * 95)
    print("Random Chance Baseline: 2.50%")
