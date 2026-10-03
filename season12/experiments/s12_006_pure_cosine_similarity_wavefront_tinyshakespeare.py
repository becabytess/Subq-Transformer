"""
===================================================================================================
STUDY S12-006: PURE PARAMETER-FREE COSINE SIMILARITY WAVEFRONT
Task: TinyShakespeare Character-Level LM | L = 64, T = 32
Architecture: Continuous Stride-1 Wavefront with Pure Geometric Cosine Gating

CORE EXPERIMENTAL QUESTION:
Can the model navigate recurrence using raw, parameter-free geometric cosine similarity?
- In S12-005, W_q and W_k were 32,768 learned parameters that could artificially manufacture similarity.
- In S12-006, W_q and W_k are completely ELIMINATED.
- Similarity is the literal geometric angle between adjacent state vectors:
      cos_sim(h_i, h_{i-1}) = (h_i . h_{i-1}) / (||h_i|| * ||h_{i-1}|| + eps)
- Gate is modulated solely by 2 learnable scalars (scale and bias):
      gate = sigmoid(scale * cos_sim + bias)
      h_i^(t) = (1 - gate) * h_i^(t-1) + gate * W_v(h_{i-1}^(t-1))

DIAGNOSTIC TARGETS:
1. Does true geometric cosine similarity naturally increase as hops t increase (50% -> 95% overlap)?
2. Does the model achieve competitive perplexity without any learned query/key projections?
3. Does raw cosine similarity distinguish word boundaries (spaces/punctuation) from continuous letters?
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
# 2. PURE COSINE SIMILARITY WAVEFRONT LATTICE MODEL
# =================================================================================================
class PureCosineWavefrontLattice(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=32, d_mlp=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T

        # Token & Position Embeddings
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # Value projection (feature rotation)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Pure Geometric Gate Scalars (ONLY 2 LEARNABLE SCALARS FOR GATING!)
        # W_q and W_k are 100% ELIMINATED.
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

        # Output Head
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
        h = self.ln_in(x_raw)

        trajectory = [] if return_diagnostics else None
        gates_list = [] if return_diagnostics else None
        cosines_list = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())

            # Immediate neighbor (offset = 1, continuous contiguous flow)
            h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))

            # PURE PARAMETER-FREE COSINE SIMILARITY: strictly in [-1, 1]
            # (h_i . h_{i-1}) / (||h_i|| * ||h_{i-1}|| + eps)
            cos_sim = F.cosine_similarity(h, h_prev, dim=-1, eps=1e-8)  # (B, L)
            
            # Scalar modulated gate
            gate = torch.sigmoid(self.gate_scale * cos_sim + self.gate_bias).unsqueeze(-1)  # (B, L, 1)

            # Value projection and convex blend
            v_prev = self.W_v(h_prev)
            h = (1.0 - gate) * h + gate * v_prev

            if return_diagnostics:
                gates_list.append(gate.squeeze(-1).detach())
                cosines_list.append(cos_sim.detach())

        if return_diagnostics:
            trajectory.append(h.detach())

        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))

        if return_diagnostics:
            return logits, trajectory, gates_list, cosines_list
        return logits


# =================================================================================================
# 3. DIAGNOSTICS & VERIFICATION HARNESS
# =================================================================================================
@torch.no_grad()
def analyze_wavefront_velocity(trajectory):
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

    for hop in range(model.T):
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        cos_sim = F.cosine_similarity(h, h_prev, dim=-1, eps=1e-8)
        gate = torch.sigmoid(model.gate_scale * cos_sim + model.gate_bias).unsqueeze(-1)
        v_prev = model.W_v(h_prev)
        h = (1.0 - gate) * h + gate * v_prev

    out = h + model.final_mlp(model.ln_mlp(h))
    logits = model.head(model.ln_f(out))

    dummy_target = torch.randint(0, model.vocab_size, (B,), device=eval_x.device)
    loss_target = F.cross_entropy(logits[:, target_pos, :], dummy_target)
    loss_target.backward()

    grad_norms = x_emb.grad.norm(dim=-1).mean(dim=0)
    return grad_norms


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
    print(f"Starting Study S12-006 on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, stoi, itos = load_tinyshakespeare()
    seq_len = 64
    T = 32

    model = PureCosineWavefrontLattice(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        T=T,
        d_mlp=512,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (W_q and W_k completely removed!) | SeqLen: {seq_len} | Hops (T): {T} | Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch_size = 64
    max_steps = 1500
    eval_interval = 250

    start_time = time.time()
    print(f"Training Pure Cosine Wavefront Lattice ({max_steps} steps, T={T} contiguous hops)...")

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

    print("\nExecuting Pure Cosine Diagnostics & Scientific Tests...\n")

    # 1. Relaxation Velocity Profile
    eval_x, _ = get_batch(val_data, batch_size=32, seq_len=seq_len, device=device)
    model.eval()
    _, trajectory, gates_list, cosines_list = model(eval_x, return_diagnostics=True)
    velocities = analyze_wavefront_velocity(trajectory)

    print("=" * 95)
    print("🔬 TEST 1: PURE COSINE FIELD VELOCITY & COSINE EVOLUTION ACROSS HOPS")
    print("=" * 95)
    print(f"{'Hop (t)':<10} | {'Field Velocity ||dh||':<25} | {'Mean Raw Cosine Sim':<25} | {'Mean Gate':<15}")
    print("-" * 95)
    for t_idx in range(T):
        hop = t_idx + 1
        if hop <= 5 or hop % 5 == 0 or hop == T:
            v = velocities[t_idx]
            mean_cos = cosines_list[t_idx].mean().item()
            mean_g = gates_list[t_idx].mean().item()
            print(f"Hop {hop:2d}     | {v:18.4f}        | {mean_cos:18.4f}        | {mean_g:10.4f}")
    print("=" * 95)

    # 2. Gradient Reach Test
    target_pos = 63
    grad_norms = analyze_gradient_reach(model, eval_x, target_pos=target_pos)

    print("\n" + "=" * 95)
    print("🔬 TEST 2: GRADIENT AUDIBILITY HORIZON (Pure Cosine Gated Reach)")
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

    # 3. Content-Aware Gating Analysis: Space/Boundary vs Char
    print("\n" + "=" * 95)
    print("🔬 TEST 3: RAW GEOMETRIC COSINE & GATING AT SYNTACTIC BOUNDARIES")
    print("Does raw geometric cosine naturally drop at word boundaries?")
    print("=" * 95)
    sample_tokens = eval_x[0].cpu().tolist()
    sample_text = [itos[tok] for tok in sample_tokens]
    final_cosines = cosines_list[-1][0].cpu().tolist()
    final_gates = gates_list[-1][0].cpu().tolist()

    space_cos = []
    char_cos = []
    space_g = []
    char_g = []
    for char, cos_val, g_val in zip(sample_text, final_cosines, final_gates):
        if char in [" ", "\n", ".", ",", ":", ";", "!"]:
            space_cos.append(cos_val)
            space_g.append(g_val)
        else:
            char_cos.append(cos_val)
            char_g.append(g_val)

    mean_space_cos = sum(space_cos) / max(len(space_cos), 1)
    mean_char_cos = sum(char_cos) / max(len(char_cos), 1)
    mean_space_g = sum(space_g) / max(len(space_g), 1)
    mean_char_g = sum(char_g) / max(len(char_g), 1)

    print(f"Raw Cosine Sim at Boundaries: {mean_space_cos:.4f}  |  Gate at Boundaries: {mean_space_g:.4f}")
    print(f"Raw Cosine Sim at Characters: {mean_char_cos:.4f}  |  Gate at Characters: {mean_char_g:.4f}")
    print(f"Learned Gate Scale Scalar:    {model.gate_scale.item():.4f}")
    print(f"Learned Gate Bias Scalar:     {model.gate_bias.item():.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
