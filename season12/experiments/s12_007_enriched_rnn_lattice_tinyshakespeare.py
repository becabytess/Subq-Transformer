"""
===================================================================================================
STUDY S12-007: THE ENRICHED RECURRENT LATTICE (PARAMETER-HANDICAPPED RIGOROUS TEST)
Task: TinyShakespeare Character-Level LM | L = 64, T = 64
Target Comparison: S12-002 Canonical RNN had 190,208 parameters (PPL 5.53).

RIGOROUS SCIENTIFIC CONTROLS:
1. Parameter Deficit:
   - S12-002 had 190,208 parameters (d_mlp = 512).
   - In S12-007, we ELIMINATE the 32k W_gate matrix entirely!
   - Gating is governed strictly by a channel-wise vector gate_logit of 128 scalar parameters.
   - We set d_mlp = 480 so the total model has 182,112 parameters (~8,100 FEWER parameters than S12-002!).
2. Guaranteed Canonical RNN Backbone:
   h_rnn = tanh(W_h * h_prev + W_x * x_own) (Identical to S12-002).
3. Zero-Initialized Self-Enrichment:
   g = sigmoid(gate_logit) (initialized to -3.0, so g ~ 0.047 at start).
   h_i^(t) = (1 - g) * h_rnn + g * h_i^(t-1)
4. Scientific Hypothesis:
   If S12-007 STILL beats S12-002 (5.53 PPL) while carrying 8,100 FEWER parameters, 
   the architectural superiority of vertical self-enrichment is proven beyond any parameter bias.
===================================================================================================
"""

import math
import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================================================
# 1. DATASET SETUP: TINYSHAKESPEARE (EXACT SAME HARNESS)
# =================================================================================================
def load_tinyshakespeare():
    data_dir = "data"
    data_path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading TinyShakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, stoi, itos


# =================================================================================================
# 2. PARAMETER-HANDICAPPED ENRICHED LATTICE MODEL
# =================================================================================================
class EnrichedRecurrentLattice(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=480):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T

        # Token & Position Embeddings (Identical to S12-002)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # Canonical RNN Backbone (Identical to S12-002)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=True)

        # ZERO-MATRIX Channel-wise Gate: Only 128 scalar parameters!
        # NO W_gate matrix. Total parameter count is strictly LOWER than S12-002!
        self.gate_logit = nn.Parameter(torch.full((d_model,), -3.0))

        # Output Head: d_mlp=480 to keep total parameters strictly below 190,208
        self.ln_mlp = nn.LayerNorm(d_model)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        nn.init.normal_(self.head.weight, 0.0, 0.02)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        dev = idx.device
        pos = torch.arange(0, L, device=dev).unsqueeze(0)
        x_raw = self.tok(idx) + self.pos(pos)
        x_own = self.ln_in(x_raw)

        h = x_own.clone()
        trajectory = [] if return_diagnostics else None
        gates_list = [] if return_diagnostics else None

        # Channel-wise gate vector (1, 1, d_model)
        g = torch.sigmoid(self.gate_logit).unsqueeze(0).unsqueeze(0)

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())

            # 1. Untouchable Canonical RNN Backbone
            h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h_rnn = torch.tanh(self.W_h(h_prev) + self.W_x(x_own))

            # 2. Pure Parameter-Free Channel-Wise Convex Blend
            h = (1.0 - g) * h_rnn + g * h

            if return_diagnostics:
                gates_list.append(g.squeeze().detach())

        if return_diagnostics:
            trajectory.append(h.detach())

        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))

        if return_diagnostics:
            return logits, trajectory, gates_list
        return logits


# =================================================================================================
# 3. SCIENTIFIC DIAGNOSTICS
# =================================================================================================
@torch.no_grad()
def analyze_field_velocity(trajectory):
    T = len(trajectory) - 1
    velocities = []
    for t in range(1, T + 1):
        diff = trajectory[t] - trajectory[t - 1]
        v = diff.norm(dim=-1).mean().item()
        velocities.append(v)
    return velocities


def analyze_gradient_reach(model, eval_x, target_pos=63):
    model.eval()
    B, L = eval_x.shape
    pos = torch.arange(0, L, device=eval_x.device).unsqueeze(0)

    x_emb = model.tok(eval_x) + model.pos(pos)
    x_emb = model.ln_in(x_emb)
    x_emb.retain_grad()
    x_emb.requires_grad_(True)

    h = x_emb
    x_own = x_emb
    g = torch.sigmoid(model.gate_logit).unsqueeze(0).unsqueeze(0)

    for hop in range(model.T):
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        h_rnn = torch.tanh(model.W_h(h_prev) + model.W_x(x_own))
        h = (1.0 - g) * h_rnn + g * h

    out = h + model.final_mlp(model.ln_mlp(h))
    logits = model.head(model.ln_f(out))

    dummy_target = torch.randint(0, model.vocab_size, (B,), device=eval_x.device)
    loss_target = F.cross_entropy(logits[:, target_pos, :], dummy_target)
    loss_target.backward()

    grad_norms = x_emb.grad.norm(dim=-1).mean(dim=0)
    return grad_norms


# =================================================================================================
# 4. TRAINING & EVALUATION HARNESS (EXACT SEED & BATCH SIZE AS S12-002)
# =================================================================================================
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate_loss(model, data, batch_size, seq_len, device, eval_iters=20):
    model.eval()
    total_loss = 0.0
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    return total_loss / eval_iters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Study S12-007 (Parameter-Handicapped Rigorous Test) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    model = EnrichedRecurrentLattice(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        T=T,
        d_mlp=480,  # Explicitly reduced to guarantee lower parameter count than S12-002
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    s12_002_params = 190208
    delta = total_params - s12_002_params
    print(f"Parameters: {total_params:,} (vs S12-002: {s12_002_params:,} | Delta: {delta:+,} params)")
    print(f"SeqLen: {seq_len} | Hops (T): {T} | Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch_size = 64
    max_steps = 1500
    eval_interval = 250

    start_time = time.time()
    print(f"Training Parameter-Handicapped Enriched Lattice ({max_steps} steps, T={T} hops)...")

    for step in range(1, max_steps + 1):
        model.train()
        x, y = get_batch(train_data, batch_size, seq_len, device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            val_loss = evaluate_loss(model, val_data, batch_size, seq_len, device)
            ppl = math.exp(min(val_loss, 20.0))
            elapsed = time.time() - start_time
            print(
                f"  Step {step:4d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | PPL: {ppl:5.2f} | Elapsed: {elapsed:5.1f}s"
            )

    print("\nExecuting Rigorous Comparative Diagnostics...\n")

    # 1. Relaxation Velocity Profile
    eval_x, _ = get_batch(val_data, batch_size=32, seq_len=seq_len, device=device)
    model.eval()
    _, trajectory, gates_list = model(eval_x, return_diagnostics=True)
    velocities = analyze_field_velocity(trajectory)

    learned_gate = torch.sigmoid(model.gate_logit)
    mean_g = learned_gate.mean().item()
    min_g = learned_gate.min().item()
    max_g = learned_gate.max().item()

    print("=" * 95)
    print("🔬 TEST 1: FIELD VELOCITY PROFILE")
    print("=" * 95)
    print(f"{'Hop (t)':<10} | {'Field Velocity ||dh||':<25} | {'Learned Channel Gate Mean':<25}")
    print("-" * 95)
    for t_idx in range(T):
        hop = t_idx + 1
        if hop <= 5 or hop % 5 == 0 or hop == T:
            v = velocities[t_idx]
            print(f"Hop {hop:2d}     | {v:18.4f}        | {mean_g:18.4f}")
    print("=" * 95)

    # 2. Gradient Reach Test
    target_pos = 63
    grad_norms = analyze_gradient_reach(model, eval_x, target_pos=target_pos)

    print("\n" + "=" * 95)
    print("🔬 TEST 2: GRADIENT AUDIBILITY HORIZON")
    print("=" * 95)
    print(f"{'Distance (d)':<16} | {'Source Position (63-d)':<25} | {'Gradient Norm ||dL_63 / dx||':<30}")
    print("-" * 95)
    distances = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60]
    for d in distances:
        src_pos = target_pos - d
        g_norm = grad_norms[src_pos].item()
        print(f"d = {d:2d} tokens     | Pos {src_pos:2d}                    | {g_norm:18.6e}")

    print("-" * 95)
    deaf_threshold = 1e-6
    active_distances = [d for d in distances if grad_norms[target_pos - d].item() > deaf_threshold]
    max_audible = max(active_distances) if active_distances else 0
    print(f">> Critical Audibility Horizon: Distance d = {max_audible} tokens")
    print("=" * 95)

    # 3. Channel Gate Distribution
    print("\n" + "=" * 95)
    print("🔬 TEST 3: CHANNEL-WISE GATE SPECIALIZATION VERDICT")
    print("=" * 95)
    print(f"Mean Gate across 128 channels: {mean_g:.4f}")
    print(f"Min Gate (most conservative):  {min_g:.4f}")
    print(f"Max Gate (most receptive):     {max_g:.4f}")
    print(f"Total Model Parameters:        {total_params:,} (S12-002 had {s12_002_params:,})")
    print("=" * 95)


if __name__ == "__main__":
    main()
