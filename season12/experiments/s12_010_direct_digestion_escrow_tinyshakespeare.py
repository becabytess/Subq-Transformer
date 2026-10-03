"""
===================================================================================================
STUDY S12-010: PURE CANONICAL RNN + DIRECT DIGESTION ESCROW
Task: TinyShakespeare Character-Level LM | L = 64, T = 64
Target Comparison: S12-002 (PPL 5.53, 190,208 params) and S12-009 (PPL 5.41, 187,948 params).

CORE ARCHITECTURAL CONCEPT:
Instead of forcing all 3 ingredients (neighbor, self, raw token) into the same non-linear RNN tube,
and instead of using complex gating networks to extract nutrients:
1. The Main Tube (Canonical RNN):
   Runs pure, untouched sequential recurrence over raw tokens:
       h_i^(t) = tanh( W_h * h_{i-1}^(t-1) + W_x * x_i )
   Zero self-state jamming. Pure sequential fidelity.
2. The Self-State IS the Digested Food:
   The hidden state h_i is treated directly as the digested nutrient.
   NO extraction gates. NO value projection matrices.
   It diffuses directly into the Causal Rolled Escrow Accumulator:
       escrow_i = (1 - alpha) * Roll(escrow_{i-1}) + alpha * h_i
3. The Decision Synthesis:
   The decision head synthesizes the raw sequential RNN state + the holographic rolled superposition:
       logits_i = Head( [ h_i ; escrow_i ] )

SCIENTIFIC CONTROLS:
- Parameters: Budgeted to 189,908 parameters (Strictly -300 FEWER parameters than S12-002: 190,208!).
- Zero gating overhead: Eliminates W_eg and W_ev matrices.
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
# 2. DIRECT DIGESTION ESCROW LATTICE MODEL
# =================================================================================================
class DirectDigestionEscrowLattice(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=340):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T

        # Token & Position Embeddings (Identical to S12-002)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # 1. The Pure Untouched Canonical RNN Backbone (Line-for-line S12-002)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=True)

        # 2. Causal Rolled Escrow Accumulator Parameter (Channel-wise mix rate)
        # NO extraction gate matrix! NO value projection matrix!
        # The hidden state h IS the nutrient itself.
        self.alpha_logit = nn.Parameter(torch.zeros(d_model))

        # 3. Output Head (Synthesizes Canonical State + Escrow Superposition)
        d_fused = d_model + d_model  # 128 + 128 = 256
        self.ln_mlp = nn.LayerNorm(d_fused)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_fused, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initializations
        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.normal_(self.W_h.weight, 0.0, 0.02)
        nn.init.normal_(self.W_x.weight, 0.0, 0.02)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        dev = idx.device
        pos = torch.arange(0, L, device=dev).unsqueeze(0)
        x_raw = self.tok(idx) + self.pos(pos)
        x_own = self.ln_in(x_raw)

        # Track 1: Pure Canonical RNN Recurrence (Untouched Digestive Tube)
        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())

            h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_h(h_prev) + self.W_x(x_own))

        if return_diagnostics:
            trajectory.append(h.detach())

        # Track 2: Direct Digestion Escrow (The hidden state h directly enters the bloodstream)
        alpha = torch.sigmoid(self.alpha_logit).unsqueeze(0)  # (1, d_model)
        escrow = torch.zeros(B, self.d_model, device=dev)
        escrow_list = []

        for i in range(L):
            h_i = h[:, i, :]  # Local digestion at position i (pure h, zero gating!)
            rolled_prev = torch.roll(escrow, shifts=1, dims=-1)  # Holographic circular shift
            escrow = (1.0 - alpha) * rolled_prev + alpha * h_i
            escrow_list.append(escrow)

        escrow_seq = torch.stack(escrow_list, dim=1)  # (B, L, d_model)

        # Track 3: The Decision Synthesis
        fused = torch.cat([h, escrow_seq], dim=-1)  # (B, L, 256)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))

        if return_diagnostics:
            return logits, trajectory, escrow_seq
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

    h = x_emb
    x_own = x_emb

    for hop in range(model.T):
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        h = torch.tanh(model.W_h(h_prev) + model.W_x(x_own))

    alpha = torch.sigmoid(model.alpha_logit).unsqueeze(0)
    escrow = torch.zeros(B, model.d_model, device=dev)
    escrow_list = []

    for i in range(L):
        h_i = h[:, i, :]
        rolled_prev = torch.roll(escrow, shifts=1, dims=-1)
        escrow = (1.0 - alpha) * rolled_prev + alpha * h_i
        escrow_list.append(escrow)

    escrow_seq = torch.stack(escrow_list, dim=1)

    fused = torch.cat([h, escrow_seq], dim=-1)
    out = h + model.final_mlp(model.ln_mlp(fused))
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
    print(f"Starting Study S12-010 (Direct Digestion Escrow) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    model = DirectDigestionEscrowLattice(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        T=T,
        d_mlp=340,  # Strictly <= S12-002
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    s12_002_params = 190208
    delta = total_params - s12_002_params
    print(f"Parameters: {total_params:,} (vs S12-002: {s12_002_params:,} | Delta: {delta:+,} params)")
    print(f"SeqLen: {seq_len} | Hops (T): {T} | Escrow Dim: 128 | Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch_size = 64
    max_steps = 1500
    eval_interval = 250

    start_time = time.time()
    print(f"Training Direct Digestion Escrow Lattice ({max_steps} steps, T={T} hops)...")

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

    print("\nExecuting Direct Digestion Escrow Scientific Diagnostics...\n")

    # 1. Relaxation Velocity Profile
    eval_x, _ = get_batch(val_data, batch_size=32, seq_len=seq_len, device=device)
    model.eval()
    _, trajectory, escrow_seq = model(eval_x, return_diagnostics=True)
    velocities = analyze_field_velocity(trajectory)

    print("=" * 95)
    print("🔬 TEST 1: PURE CANONICAL TUBE FIELD VELOCITY PROFILE")
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
    print("🔬 TEST 2: GRADIENT AUDIBILITY HORIZON (Direct Digestion Reach)")
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

    # 3. Escrow Diagnostics
    print("\n" + "=" * 95)
    print("🔬 TEST 3: DIRECT DIGESTION ESCROW DYNAMICS")
    print("=" * 95)
    alpha = torch.sigmoid(model.alpha_logit)
    mean_alpha = alpha.mean().item()
    min_alpha = alpha.min().item()
    max_alpha = alpha.max().item()
    mean_h_norm = trajectory[-1].norm(dim=-1).mean().item()
    mean_escrow_norm = escrow_seq.norm(dim=-1).mean().item()

    print(f"Mean Alpha (Direct h vs Rolled History blend): {mean_alpha:.4f}")
    print(f"Min Alpha (Most historical / slowest decay):    {min_alpha:.4f}")
    print(f"Max Alpha (Most immediate / fast intake):       {max_alpha:.4f}")
    print(f"Active State Norm ||h|| at Exit:                {mean_h_norm:.4f}")
    print(f"Direct Escrow Norm ||escrow|| at Exit:          {mean_escrow_norm:.4f}")
    print(f"Total Parameters:                               {total_params:,} (Budget: {s12_002_params:,})")
    print("=" * 95)


if __name__ == "__main__":
    main()
