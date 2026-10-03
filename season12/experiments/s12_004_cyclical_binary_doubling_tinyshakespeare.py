"""
===================================================================================================
STUDY S12-004: CYCLICAL BINARY DOUBLING LATTICE ON TINYSHAKESPEARE
Adding Depth by Looping the 6-Hop Schedule:
Cycle Structure: Repeating [1, 2, 4, 8, 16, 32] across N Cycles
Total Hops = num_cycles * 6 (e.g., 3 Cycles = 18 Hops)
Update: h[i]^(t) = tanh( W_self * h[i]^(t-1) + W_neighbor * h[i - offset]^(t-1) + W_x * x[i] )

KEY ADVANTAGES:
1. Re-visits offset = 1 at the start of every cycle with globally enriched context
2. Boosts nonlinear depth from 6 to 18 layers while remaining 3.5x faster than 64 hops
3. Tests whether repeated multi-scale relaxation pushes perplexity down towards 5.5
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
# 1. DATASET SETUP: TINYSHAKESPEARE
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
# 2. CYCLICAL BINARY DOUBLING LATTICE ARCHITECTURE
# =================================================================================================
class CyclicalBinaryDoublingLattice(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, d_mlp=512, num_cycles=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_cycles = num_cycles

        # Base schedule: log2(64) = 6 hops -> [1, 2, 4, 8, 16, 32]
        self.T_base = int(math.ceil(math.log2(seq_len)))
        self.base_offsets = [2**t for t in range(self.T_base)]

        # Total hops = num_cycles * 6 (e.g. 3 * 6 = 18 hops)
        self.total_hops = self.num_cycles * self.T_base

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # Shared recurrent fusion cell
        self.W_self = nn.Linear(d_model, d_model, bias=False)
        self.W_neighbor = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=True)

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

    def forward(self, idx, return_trajectory=False):
        B, L = idx.shape
        dev = idx.device
        pos = torch.arange(0, L, device=dev).unsqueeze(0)
        x_raw = self.tok(idx) + self.pos(pos)
        x_own = self.ln_in(x_raw)

        h = x_own.clone()
        trajectory = [] if return_trajectory else None

        # Outer loop over cycles (e.g. 3 cycles)
        for cycle in range(self.num_cycles):
            for hop in range(self.T_base):
                if return_trajectory:
                    trajectory.append(h.detach())

                offset = self.base_offsets[hop]
                h_neighbor = F.pad(h[:, :-offset, :], (0, 0, offset, 0))

                # Fused update
                h = torch.tanh(self.W_self(h) + self.W_neighbor(h_neighbor) + self.W_x(x_own))

        if return_trajectory:
            trajectory.append(h.detach())

        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))

        if return_trajectory:
            return logits, trajectory
        return logits


# =================================================================================================
# 3. DIAGNOSTICS: RELAXATION PROFILE & GRADIENT REACH
# =================================================================================================
@torch.no_grad()
def analyze_cyclical_relaxation(model, eval_x):
    model.eval()
    B, L = eval_x.shape
    _, trajectory = model(eval_x, return_trajectory=True)
    total_hops = len(trajectory) - 1

    print("\n" + "=" * 95)
    print(f"🔬 TEST 1: CYCLICAL RELAXATION PROFILE across {model.num_cycles} Cycles ({total_hops} Hops Total)")
    print("=" * 95)
    print(f"{'Global Hop':<12} | {'Cycle':<8} | {'Lookback (2^t)':<16} | {'Mean Velocity':<15} | {'Pos 60 Velocity':<16}")
    print("-" * 75)

    hop_counter = 0
    for cycle in range(model.num_cycles):
        for h in range(model.T_base):
            hop_counter += 1
            prev = trajectory[hop_counter - 1]
            curr = trajectory[hop_counter]
            delta = curr - prev
            mean_vel = delta.norm(dim=-1).mean().item()
            v60 = delta[:, 60, :].norm(dim=-1).mean().item()
            offset = model.base_offsets[h]
            print(f"Hop {hop_counter:2d}      | Cycle {cycle+1} | offset = {offset:2d}         | {mean_vel:12.4f}  | {v60:12.4f}")

    print("-" * 75)


def test_gradient_reach_cyclical(model, eval_x, eval_y):
    """
    Measures the influence of past tokens at distance d on the loss at position 63.
    """
    model.eval()
    B, L = eval_x.shape
    dev = eval_x.device

    pos = torch.arange(0, L, device=dev).unsqueeze(0)
    embeds = model.tok(eval_x).detach().requires_grad_(True)
    x_raw = embeds + model.pos(pos)
    x_own = model.ln_in(x_raw)

    h = x_own.clone()
    for cycle in range(model.num_cycles):
        for hop in range(model.T_base):
            offset = model.base_offsets[hop]
            h_neighbor = F.pad(h[:, :-offset, :], (0, 0, offset, 0))
            h = torch.tanh(model.W_self(h) + model.W_neighbor(h_neighbor) + model.W_x(x_own))

    out = h + model.final_mlp(model.ln_mlp(h))
    logits = model.head(model.ln_f(out))

    target_pos = 63
    loss_at_63 = F.cross_entropy(logits[:, target_pos], eval_y[:, target_pos])
    loss_at_63.backward()

    grad_norms = embeds.grad.norm(dim=-1).mean(dim=0)

    print("\n" + "=" * 95)
    print("🔬 TEST 2: GRADIENT REACH TEST ON TOKEN 63 (Influence across Distance d)")
    print("=" * 95)
    print(f"{'Distance (d)':<16} | {'Source Position (63-d)':<25} | {'Gradient Norm ||dL_63 / dx||':<30}")
    print("-" * 95)

    distances = [1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48, 56, 60]
    for d in distances:
        src_pos = target_pos - d
        g_norm = grad_norms[src_pos].item()
        print(f"d = {d:2d} tokens     | Pos {src_pos:2d}                    | {g_norm:18.6e}")

    print("-" * 95)
    g_at_1 = grad_norms[target_pos - 1].item()
    g_at_60 = grad_norms[target_pos - 60].item()
    print(f">> Gradient sensitivity at d = 1 (Immediate neighbor): {g_at_1:.6e}")
    print(f">> Gradient sensitivity at d = 60 (Distant context):    {g_at_60:.6e}")
    print("=" * 95)


# =================================================================================================
# 4. TRAINING HARNESS
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
    print(f"Starting Study S12-004 on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    num_cycles = 3  # 3 cycles of [1, 2, 4, 8, 16, 32] = 18 total hops

    model = CyclicalBinaryDoublingLattice(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        d_mlp=512,
        num_cycles=num_cycles,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} | SeqLen: {seq_len} | Cycles: {num_cycles} | Total Hops: {model.total_hops} | Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\nTraining Cyclical Binary Doubling Lattice (1500 steps, 18 hops)...")
    total_steps = 1500
    eval_interval = 250
    start_time = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        x, y = get_batch(train_data, batch_size=32, seq_len=seq_len, device=device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == total_steps:
            elapsed = time.time() - start_time
            val_loss = evaluate_loss(model, val_data, batch_size=32, seq_len=seq_len, device=device)
            ppl = math.exp(min(val_loss, 20))
            print(f"  Step {step:4d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | PPL: {ppl:5.2f} | Elapsed: {elapsed:5.1f}s")

    # Run Diagnostics
    print("\nExecuting Cyclical Relaxation & Gradient Reach Tests...")
    eval_x, eval_y = get_batch(val_data, batch_size=16, seq_len=seq_len, device=device)
    analyze_cyclical_relaxation(model, eval_x)
    test_gradient_reach_cyclical(model, eval_x, eval_y)


if __name__ == "__main__":
    main()
