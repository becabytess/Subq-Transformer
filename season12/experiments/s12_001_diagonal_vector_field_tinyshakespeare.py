"""
===================================================================================================
STUDY S12-001: BEHAVIORAL ANALYSIS OF THE RECURRENT LATTICE (K=1, OFFSET=1)
Task: TinyShakespeare Language Modeling | L = 64, T = 64
Architecture: Exact Elman RNN on the Parallel Diagonal Lattice
Update: h[i]^(t) = tanh( W_h * h[i-1]^(t-1) + W_x * x[i] + b )

OBJECTIVE: BEHAVIORAL FIELD DYNAMICS (NOT A CANDIDATE SHOOTOUT)
1. Vector Field Velocity: ||h^(t) - h^(t-1)|| across hops 1..64 (Does it relax into an attractor?)
2. Net Direction & Global Flow: Cosine alignment with the sequence-wide mean drift vector
3. Local Attractor Analysis: Do specific token classes (vowels, spaces, punctuation) cluster in state space?
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
# 2. THE RECURRENT LATTICE MODEL (THE EXACT RNN FORMULATION)
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

        # Standard Elman RNN Cell: h_next = tanh(W_h * h_prev + W_x * x_own + b)
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
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))

        h = x_own.clone()
        trajectory = [] if return_trajectory else None

        # Fully vectorized hop loop across parallel diagonals
        for hop in range(self.T):
            if return_trajectory:
                trajectory.append(h.detach())

            # Vectorized right shift: token i pulls from token i-1 for all i in parallel
            h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))

            # Standard RNN update across all tokens simultaneously
            h = torch.tanh(self.W_h(h_prev) + self.W_x(x_own))

        if return_trajectory:
            trajectory.append(h.detach())

        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))

        if return_trajectory:
            return logits, trajectory
        return logits


# =================================================================================================
# 3. BEHAVIORAL FIELD & ATTRACTOR DIAGNOSTIC ENGINE
# =================================================================================================
@torch.no_grad()
def deep_behavioral_analysis(model, eval_x, itos):
    """
    Exhaustively examines:
    1. Velocity profile v(t): is the field relaxing toward fixed-point attractors?
    2. Net direction alignment: are token steps coherent with a global flow?
    3. Token attractor clustering: do specific tokens relax into distinct state clusters?
    """
    model.eval()
    B, L = eval_x.shape
    logits, trajectory = model(eval_x, return_trajectory=True)
    T = len(trajectory) - 1

    print("\n" + "=" * 95)
    print("🔬 DEEP BEHAVIORAL ANALYSIS: THE STATE VECTOR FIELD & ATTRACTORS")
    print("=" * 95)

    # 1. FIELD VELOCITY & GLOBAL COHERENCE
    print(f"\n[1. FIELD RELAXATION DYNAMICS across {T} Hops]")
    print(f"{'Hop':<6} | {'Field Velocity (Mean Step Size)':<32} | {'Net Direction Alignment':<25}")
    print("-" * 75)

    hop_samples = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    vels = []
    aligns = []

    for t in range(1, T + 1):
        prev = trajectory[t - 1]
        curr = trajectory[t]
        delta = curr - prev  # [B, L, d_model]

        vel = delta.norm(dim=-1).mean().item()
        vels.append(vel)

        # Sequence-level mean drift vector: [B, 1, d_model]
        mean_drift = delta.mean(dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(delta, mean_drift, dim=-1).mean().item()
        aligns.append(cos_sim)

        if t in hop_samples:
            print(f"Hop {t:2d} | {vel:8.4f}                         | {cos_sim:8.4f}")

    vel_drop = ((vels[0] - vels[-1]) / vels[0]) * 100.0
    print("-" * 75)
    print(f">> Total Velocity Drop: {vel_drop:.1f}%")
    if vel_drop > 40:
        print(">> [VERDICT: ATTRACTOR RELAXATION DETECTED] The field decelerates significantly, settling into a steady state.")
    else:
        print(">> [VERDICT: CONTINUOUS FLOW] The field maintains constant dynamic velocity.")

    # 2. LOCAL TOKEN ATTRACTOR CLUSTERING
    # Let's inspect the final states h^(T) of specific token characters from sample 0
    sample_tokens = eval_x[0]  # [L]
    final_states = trajectory[-1][0]  # [L, d_model]

    print("\n[2. TOKEN ATTRACTOR ANALYSIS: How Characters Cluster in State Space]")
    # Group token representations by character
    char_states = {}
    for idx, tok_id in enumerate(sample_tokens.tolist()):
        ch = itos[tok_id]
        if ch not in char_states:
            char_states[ch] = []
        char_states[ch].append(final_states[idx])

    # Compute intra-character self-similarity vs inter-character distance
    print(f"{'Character':<12} | {'Occurrences':<12} | {'Internal State Consistency (Cosine)':<35}")
    print("-" * 65)
    frequent_chars = sorted(char_states.keys(), key=lambda c: len(char_states[c]), reverse=True)[:6]

    for ch in frequent_chars:
        states = torch.stack(char_states[ch])
        display_ch = repr(ch)
        if len(states) > 1:
            # Pairwise cosine similarity among occurrences of the same character
            norm_s = F.normalize(states, dim=-1)
            sim_matrix = torch.mm(norm_s, norm_s.T)
            # Average off-diagonal
            n = len(states)
            off_diag = (sim_matrix.sum() - n) / (n * (n - 1))
            print(f"{display_ch:<12} | {n:<12} | {off_diag.item():.4f}")
        else:
            print(f"{display_ch:<12} | {1:<12} | (Single occurrence)")

    print("=" * 95)


# =================================================================================================
# 4. TRAINING & EVALUATION HARNESS
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
    print(f"Starting Study S12-001 on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, itos = load_tinyshakespeare()
    seq_len = 64
    T = 64

    # Single Canonical Model: The Exact Elman RNN on the Parallel Diagonal Lattice
    print("\nINITIALIZING: The Canonical Recurrent Lattice RNN (K=1, Offset=1)")
    model = RecurrentLatticeRNN(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        T=T,
        d_mlp=512,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} | Sequence Length (L): {seq_len} | Hops (T): {T} | Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\nTraining on TinyShakespeare...")
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
            print(f"  Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:6.2f} | Elapsed: {elapsed:5.1f}s")

    # Deep Behavioral Analysis
    print("\nRunning Behavioral Field Analysis on Trained Model...")
    eval_x, _ = get_batch(val_data, batch_size=16, seq_len=seq_len, device=device)
    deep_behavioral_analysis(model, eval_x, itos)


if __name__ == "__main__":
    main()
