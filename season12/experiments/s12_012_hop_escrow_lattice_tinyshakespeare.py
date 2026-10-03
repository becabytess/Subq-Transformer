"""
===================================================================================================
STUDY S12-012: HOP-ESCROW LATTICE (VERTICAL DIGESTION OF THE 3RD TERM)
Task: TinyShakespeare Character-Level LM | L = 64, T = 64
Target Comparison: S12-002 (PPL 5.53, 190,208 params), S12-008 (PPL 5.45), S12-011 (PPL 5.41).

CORE ARCHITECTURAL CONCEPT:
In Study S12-008 (3-Way Fusion), we identified 3 distinct information streams at every cell:
1. h_prev: incoming spatial neighbor state from the left (diagonal bucket brigade)
2. x_own: current raw token observation
3. h: the token's own previous hidden state

In S12-008, we tried cramming all 3 into one non-linear bottle:
    h = tanh( W_neighbor * h_prev + W_self * h + W_raw * x_own )
In S12-002, the 3rd term (h) was completely ditched and overwritten at every hop.

Study S12-012 implements the true Feature-Escrow principle on the 3rd term:
- The main digestive tube runs cleanly on the incoming neighbor and raw token:
      h_next = tanh( W_h * h_{i-1}^(t-1) + W_x * x_i )
- The 3rd term (h_i^(t-1)) is NOT discarded and NOT crammed into tanh.
  It is treated directly as the digested nutrient and absorbed into the Escrow Vault:
      gamma = sigmoid( W_roll * h )
      escrow = (1 - gamma) * escrow + gamma * Roll(escrow) + h
  Each token's escrow register accumulates the multi-scale progression of n-grams (1-gram, 2-gram...
  up to 64-gram) across all hops, while the main tube relaxes smoothly into equilibrium.

SCIENTIFIC CONTROLS:
- Total Parameters: 189,909 (Strictly -299 FEWER parameters than S12-002: 190,208!).
- Identical seed, sequence length (L=64), hops (T=64), batch size (64), and AdamW optimizer.
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
# 2. HOP-ESCROW LATTICE MODEL
# =================================================================================================
class HopEscrowLattice(nn.Module):
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

        # 1. Pure Untouched Canonical RNN Backbone (2-term: neighbor + raw)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=True)

        # 2. Canonical FEN Dynamic Roll Gate on the 3rd term (self state h)
        self.roll_gate = nn.Linear(d_model, 1, bias=True)

        # 3. Output Head (Synthesizes Canonical State + Vertical Hop Escrow)
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
        nn.init.normal_(self.roll_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        dev = idx.device
        pos = torch.arange(0, L, device=dev).unsqueeze(0)
        x_raw = self.tok(idx) + self.pos(pos)
        x_own = self.ln_in(x_raw)

        h = x_own.clone()
        escrow = torch.zeros(B, L, self.d_model, device=dev)
        trajectory = [] if return_diagnostics else None
        gammas = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())

            # 1. Absorption of 3rd term (current self-state h) into vertical escrow
            gamma = torch.sigmoid(self.roll_gate(h))  # (B, L, 1)
            if return_diagnostics:
                gammas.append(gamma.detach())

            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma) * escrow + gamma * rolled + h

            # 2. Pure 2-term canonical RNN step along diagonal wavefront
            h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_h(h_prev) + self.W_x(x_own))

        if return_diagnostics:
            trajectory.append(h.detach())

        # Decision Synthesis: combines final relaxed tube state + vertical multi-scale escrow
        fused = torch.cat([h, escrow], dim=-1)  # (B, L, 256)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))

        if return_diagnostics:
            gamma_seq = torch.stack(gammas, dim=1)  # (B, T, L, 1)
            return logits, trajectory, escrow, gamma_seq
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
    escrow = torch.zeros(B, L, model.d_model, device=dev)

    for hop in range(model.T):
        gamma = torch.sigmoid(model.roll_gate(h))
        rolled = torch.roll(escrow, shifts=1, dims=-1)
        escrow = (1.0 - gamma) * escrow + gamma * rolled + h

        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        h = torch.tanh(model.W_h(h_prev) + model.W_x(x_own))

    fused = torch.cat([h, escrow], dim=-1)
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
    print(f"Starting Study S12-012 (Hop-Escrow Lattice) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    model = HopEscrowLattice(
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
    print(f"Training Hop-Escrow Lattice ({max_steps} steps, T={T} hops)...")

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

    print("\nExecuting Hop-Escrow Scientific Diagnostics...\n")

    # 1. Relaxation Velocity Profile
    eval_x, _ = get_batch(val_data, batch_size=32, seq_len=seq_len, device=device)
    model.eval()
    _, trajectory, escrow, gamma_seq = model(eval_x, return_diagnostics=True)
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
    print("🔬 TEST 2: GRADIENT AUDIBILITY HORIZON (Hop-Escrow Reach)")
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

    # 3. Dynamic Roll Gate & Escrow Diagnostics
    print("\n" + "=" * 95)
    print("🔬 TEST 3: VERTICAL HOP-ESCROW DYNAMICS")
    print("=" * 95)
    mean_gamma = gamma_seq.mean().item()
    min_gamma = gamma_seq.min().item()
    max_gamma = gamma_seq.max().item()
    std_gamma = gamma_seq.std().item()
    mean_h_norm = trajectory[-1].norm(dim=-1).mean().item()
    mean_escrow_norm = escrow.norm(dim=-1).mean().item()

    print(f"Dynamic Roll Gate gamma (mean across hops/tokens): {mean_gamma:.4f}")
    print(f"Dynamic Roll Gate gamma (min):                     {min_gamma:.4f}")
    print(f"Dynamic Roll Gate gamma (max):                     {max_gamma:.4f}")
    print(f"Dynamic Roll Gate gamma (std):                     {std_gamma:.4f}")
    print(f"Active State Norm ||h|| at Exit:                   {mean_h_norm:.4f}")
    print(f"Vertical Hop-Escrow Norm ||escrow|| at Exit:       {mean_escrow_norm:.4f}")
    print(f"Total Parameters:                                  {total_params:,} (Budget: {s12_002_params:,})")
    print("=" * 95)


if __name__ == "__main__":
    main()
