"""
===================================================================================================
STUDY S12-013: DUAL CROSS-SYMMETRIC RECURRENT LATTICE (4-WAY PAIRWISE BALANCE)
Task: TinyShakespeare Character-Level LM | L = 64, T = 64
Target Comparison: S12-002 (PPL 5.53, 190,208 params), S12-008 (PPL 5.45, 190,144 params).

CORE ARCHITECTURAL CONCEPT:
In S12-002, the recurrence was inherently asymmetric across the pair (i-1, i):
- Left token (i-1) contributed its hidden state h_{i-1}.
- Right token (i) contributed its raw observation x_i.
- The right hidden state h_i and the left raw observation x_{i-1} were both excluded.

Study S12-013 establishes full pairwise cross-symmetry:
Between the adjacent pair (i-1, i), both tokens contribute BOTH their hidden states and their raw tokens:
1. Left hidden state:   h_{i-1}^(t-1)  [Forward Context Wave]
2. Right hidden state:  h_i^(t-1)      [Self-Temporal Memory]
3. Left raw token:      x_{i-1}        [Left Observation Anchor]
4. Right raw token:     x_i            [Right Observation Anchor]

The 4-Way Cross-Symmetric Recurrence:
    h_i^(t) = tanh( W_hl * h_{i-1}^(t-1) + W_hr * h_i^(t-1) + W_xl * x_{i-1} + W_xr * x_i )

SCIENTIFIC CONTROLS:
- Total Parameters: 190,080 (Strictly -128 FEWER parameters than S12-002: 190,208!).
- d_mlp = 384 maintains strict parameter parity.
- 100% Causal: All inputs come strictly from positions <= i.
- Diagnostics measure the 4-way weight balance (The Tetrad Law across the 4 forces).
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
# 2. DUAL CROSS-SYMMETRIC RECURRENT LATTICE MODEL
# =================================================================================================
class DualCrossSymmetricLattice(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=384):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T

        # Token & Position Embeddings (Identical to S12-002)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # 4-Way Cross-Symmetric Recurrence Matrices
        self.W_hl = nn.Linear(d_model, d_model, bias=False)  # Left Hidden
        self.W_hr = nn.Linear(d_model, d_model, bias=False)  # Right Hidden
        self.W_xl = nn.Linear(d_model, d_model, bias=False)  # Left Raw Token
        self.W_xr = nn.Linear(d_model, d_model, bias=True)   # Right Raw Token

        # Output MLP & Head (Budget-matched to stay <= 190,208)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initializations
        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hr.weight, 0.0, 0.01)  # small initial self persistence
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xr.weight, 0.0, 0.02)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        dev = idx.device
        pos = torch.arange(0, L, device=dev).unsqueeze(0)

        # Raw Token Observations
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())

            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))

            # 4-Way Cross-Symmetric update
            h = torch.tanh(
                self.W_hl(h_left) + 
                self.W_hr(h) + 
                self.W_xl(x_left) + 
                self.W_xr(x_own)
            )

        if return_diagnostics:
            trajectory.append(h.detach())

        # Pointwise Output Synthesis
        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))

        if return_diagnostics:
            return logits, trajectory
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
    dev = eval_x.device
    pos = torch.arange(0, L, device=dev).unsqueeze(0)

    x_emb = model.tok(eval_x) + model.pos(pos)
    x_emb = model.ln_in(x_emb)
    x_emb.retain_grad()
    x_emb.requires_grad_(True)

    x_own = x_emb
    x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))
    h = x_own

    for hop in range(model.T):
        h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        h = torch.tanh(
            model.W_hl(h_left) + 
            model.W_hr(h) + 
            model.W_xl(x_left) + 
            model.W_xr(x_own)
        )

    out = h + model.final_mlp(model.ln_mlp(h))
    logits = model.head(model.ln_f(out))

    dummy_target = torch.randint(0, model.vocab_size, (B,), device=dev)
    loss_target = F.cross_entropy(logits[:, target_pos, :], dummy_target)
    loss_target.backward()

    grad_norms = x_emb.grad.norm(dim=-1).mean(dim=0)
    return grad_norms


# =================================================================================================
# 4. TRAINING & EVALUATION HARNESS (EXACT SAME SEED & STEPS AS S12-002)
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
    print(f"Starting Study S12-013 (Dual Cross-Symmetric Lattice) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    model = DualCrossSymmetricLattice(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        T=T,
        d_mlp=384,  # Budget-matched: 190,080 params (<= 190,208)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    s12_002_params = 190208
    delta = total_params - s12_002_params
    print(f"Parameters: {total_params:,} (vs S12-002: {s12_002_params:,} | Delta: {delta:+,} params)")
    print(f"SeqLen: {seq_len} | Hops (T): {T} | Hidden Dim: 128 | Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch_size = 64
    max_steps = 1500
    eval_interval = 250

    start_time = time.time()
    print(f"Training Dual Cross-Symmetric Lattice ({max_steps} steps, T={T} hops)...")

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

    print("\nExecuting Dual Cross-Symmetric Scientific Diagnostics...\n")

    # 1. Relaxation Velocity Profile
    eval_x, _ = get_batch(val_data, batch_size=32, seq_len=seq_len, device=device)
    model.eval()
    _, trajectory = model(eval_x, return_diagnostics=True)
    velocities = analyze_field_velocity(trajectory)

    print("=" * 95)
    print("🔬 TEST 1: DUAL CROSS-SYMMETRIC FIELD VELOCITY PROFILE")
    print("=" * 95)
    print(f"{'Hop (t)':<10} | {'Field Velocity ||dh||':<25} | {'Physical State':<40}")
    print("-" * 95)
    for t_idx in range(T):
        hop = t_idx + 1
        if hop <= 5 or hop % 5 == 0 or hop == T:
            v = velocities[t_idx]
            status = "Dynamic Intake" if v > 1.0 else ("Settling State" if v > 0.05 else "Frozen Attractor")
            print(f"Hop {hop:2d}     | {v:18.4f}        | {status}")
    print("=" * 95)

    # 2. Gradient Reach Test
    target_pos = 63
    grad_norms = analyze_gradient_reach(model, eval_x, target_pos=target_pos)

    print("\n" + "=" * 95)
    print("🔬 TEST 2: GRADIENT AUDIBILITY HORIZON (Cross-Symmetric Reach)")
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

    # 3. 4-Way Tetrad Force Balance
    print("\n" + "=" * 95)
    print("🔬 TEST 3: THE 4-WAY TETRAD FORCE BALANCE")
    print("=" * 95)
    norm_hl = model.W_hl.weight.norm().item()
    norm_hr = model.W_hr.weight.norm().item()
    norm_xl = model.W_xl.weight.norm().item()
    norm_xr = model.W_xr.weight.norm().item()
    total_w = norm_hl + norm_hr + norm_xl + norm_xr

    p_hl = (norm_hl / total_w) * 100
    p_hr = (norm_hr / total_w) * 100
    p_xl = (norm_xl / total_w) * 100
    p_xr = (norm_xr / total_w) * 100
    exit_norm = trajectory[-1].norm(dim=-1).mean().item()

    print(f"1. Left Hidden State  ||W_hl|| (Context wave):    {norm_hl:.4f} ({p_hl:5.1f}%)")
    print(f"2. Right Hidden State ||W_hr|| (Self-memory):     {norm_hr:.4f} ({p_hr:5.1f}%)")
    print(f"3. Left Raw Token     ||W_xl|| (Left anchor):     {norm_xl:.4f} ({p_xl:5.1f}%)")
    print(f"4. Right Raw Token    ||W_xr|| (Right anchor):    {norm_xr:.4f} ({p_xr:5.1f}%)")
    print(f"Exit State Norm ||h|| at Hop 64:                  {exit_norm:.4f}")
    print(f"Total Parameters:                                 {total_params:,} (Budget: {s12_002_params:,})")
    print("=" * 95)


if __name__ == "__main__":
    main()
