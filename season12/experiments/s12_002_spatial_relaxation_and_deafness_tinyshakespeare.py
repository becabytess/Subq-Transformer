"""
===================================================================================================
STUDY S12-002: SPATIAL RELAXATION & THE DEAFNESS HORIZON TEST
Task: TinyShakespeare Character-Level LM | L = 64, T = 64
Architecture: Canonical Recurrent Lattice RNN (K=1, Offset=1)

RESEARCH QUESTIONS:
1. Does convergence speed depend on position depth i?
   - Do late tokens (i=60) keep relaxing longer than early tokens (i=10)?
   - Or does the entire lattice freeze at Hop 20 regardless of sequence position?
2. Informational Satiation vs. Architectural Deafness:
   - Does token 63 stop updating because it "has seen enough" (Satiation)?
   - Or because the recurrent contraction gamma^t killed the gradient (Deafness)?
   - Test: Gradient Influence Curve || d(Loss_63) / d(Input_63-d) || across distances d=1..60.
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
# 1. DATASET SETUP: TINYSHAKESPEARE (AUTO-DOWNLOADED ON COLAB)
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
# 2. CANONICAL RECURRENT LATTICE RNN (K=1, OFFSET=1)
# =================================================================================================
class RecurrentLatticeRNN(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_h = nn.Linear(d_model, d_model, bias=False)
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

        for hop in range(self.T):
            if return_trajectory:
                trajectory.append(h.detach())

            h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_h(h_prev) + self.W_x(x_own))

        if return_trajectory:
            trajectory.append(h.detach())

        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))

        if return_trajectory:
            return logits, trajectory
        return logits


# =================================================================================================
# 3. SPATIAL VELOCITY & DEAFNESS DIAGNOSTICS
# =================================================================================================
@torch.no_grad()
def analyze_spatial_velocity(model, eval_x):
    """
    Measures velocity ||h_i^(t) - h_i^(t-1)|| for specific position slices i in {10, 25, 40, 60}.
    """
    model.eval()
    B, L = eval_x.shape
    _, trajectory = model(eval_x, return_trajectory=True)
    T = len(trajectory) - 1

    positions = [10, 25, 40, 60]
    sample_hops = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 64]

    # Matrix: [len(sample_hops), len(positions)]
    grid = {pos: [] for pos in positions}

    for t in range(1, T + 1):
        prev = trajectory[t - 1]
        curr = trajectory[t]
        delta = curr - prev  # [B, L, d_model]

        for pos in positions:
            vel_pos = delta[:, pos, :].norm(dim=-1).mean().item()
            grid[pos].append(vel_pos)

    print("\n" + "=" * 95)
    print("🔬 TEST 1: SPATIAL RELAXATION GRID (Position vs. Hop Velocity)")
    print("Does late token (i=60) keep relaxing longer than early tokens (i=10)?")
    print("=" * 95)
    print(f"{'Hop':<8} | {'Pos 10 (early)':<18} | {'Pos 25 (mid-early)':<20} | {'Pos 40 (mid-late)':<20} | {'Pos 60 (late)':<18}")
    print("-" * 95)

    for h_idx in sample_hops:
        v10 = grid[10][h_idx - 1]
        v25 = grid[25][h_idx - 1]
        v40 = grid[40][h_idx - 1]
        v60 = grid[60][h_idx - 1]
        print(f"Hop {h_idx:2d}  | {v10:14.4f}   | {v25:16.4f}     | {v40:16.4f}     | {v60:14.4f}")

    print("-" * 95)


def test_gradient_deafness_horizon(model, eval_x, eval_y):
    """
    Measures the influence of past token inputs at distance d on the loss at position 63:
    || d(Loss_63) / d(Embed_63-d) ||
    Proves whether the network physically 'hears' tokens at distances d=1..60.
    """
    model.eval()
    B, L = eval_x.shape
    dev = eval_x.device

    pos = torch.arange(0, L, device=dev).unsqueeze(0)
    embeds = model.tok(eval_x).detach().requires_grad_(True)
    x_raw = embeds + model.pos(pos)
    x_own = model.ln_in(x_raw)

    h = x_own.clone()
    for hop in range(model.T):
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        h = torch.tanh(model.W_h(h_prev) + model.W_x(x_own))

    out = h + model.final_mlp(model.ln_mlp(h))
    logits = model.head(model.ln_f(out))

    target_pos = 63
    loss_at_63 = F.cross_entropy(logits[:, target_pos], eval_y[:, target_pos])
    loss_at_63.backward()

    # Inspect gradient magnitudes at each distance d back from target_pos
    grad_norms = embeds.grad.norm(dim=-1).mean(dim=0)  # [L]

    print("\n" + "=" * 95)
    print("🔬 TEST 2: GRADIENT DEAFNESS HORIZON (Influence of Distance d on Token 63)")
    print("Is the model Satiated (chose to ignore) or Deaf (zero mathematical gradient)?")
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
    if max_audible < 25:
        print(">> [VERDICT: ARCHITECTURAL DEAFNESS] Gradient vanishes exponentially into zero past ~20 tokens.")
        print("   The network does not 'choose' to ignore distant tokens; it physically cannot hear them!")
    else:
        print(">> [VERDICT: LONG-RANGE AWARENESS] The network maintains active gradient sensitivity past distance 25.")
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
    print(f"Starting Study S12-002 on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    model = RecurrentLatticeRNN(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        T=T,
        d_mlp=512,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} | SeqLen (L): {seq_len} | Hops (T): {T} | Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\nTraining Canonical Recurrent Lattice RNN (1500 steps)...")
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
    print("\nExecuting Spatial Relaxation & Gradient Deafness Tests...")
    eval_x, eval_y = get_batch(val_data, batch_size=16, seq_len=seq_len, device=device)
    analyze_spatial_velocity(model, eval_x)
    test_gradient_deafness_horizon(model, eval_x, eval_y)


if __name__ == "__main__":
    main()
