"""
===================================================================================================
STUDY S12-016: THE TITANS DUEL (OPTIMIZED FLAT SUM VS OPTIMIZED TWO-STREAM)
Task: TinyShakespeare Character-Level LM | L = 64, T = 64 | 1500 steps each
Hardware Baseline: S12-002 (190,208 params, PPL 5.53)

MODELS IN THE DUEL (100% Mathematically Identical to S12-013 & S12-014 Cand 2):
1. Model 0: Flat Sum (S12-013 Incumbent Champion)
   h = tanh( W_hl * h_{i-1} + W_hr * h_i + (W_xl * x_{i-1} + W_xr * x_i) )
   Static observation sum precomputed once outside the 64-hop loop!
   [190,080 params | delta: -128 | d_mlp: 384]

2. Model 1: Two-Stream Gated Predictor-Corrector (S12-014 Cand 2 Gradient Champion)
   v_fwd = tanh( W_hf * h_{i-1} + W_xf * x_i )
   v_loc = tanh( W_hl * h_i + W_xl * x_{i-1} )
   g = sigmoid( W_g * [v_fwd ; v_loc] )
   h = g * v_fwd + (1 - g) * v_loc
   Static observation biases precomputed once outside the 64-hop loop!
   Full dynamic learned per-channel gating preserved with zero approximation!
   [190,208 params | delta: 0 | d_mlp: 256]
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
# 1. DATASET SETUP
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
# 2. EXACT ARCHITECTURES WITH PRECOMPUTED STATIC OBSERVATIONS
# =================================================================================================

# --- Model 0: Flat Sum (S12-013 Exact Math, Precomputed x) ---
class Model0_FlatSum(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=384):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Model 0: Flat Sum (S12-013)"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_hl = nn.Linear(d_model, d_model, bias=False)
        self.W_hr = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=False)
        self.W_xr = nn.Linear(d_model, d_model, bias=True)

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
        nn.init.normal_(self.W_hl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hr.weight, 0.0, 0.01)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xr.weight, 0.0, 0.02)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        # Mathematically identical: precompute static observation terms ONCE outside 64 hops
        x_proj = self.W_xl(x_left) + self.W_xr(x_own)

        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(
                self.W_hl(h_left) + 
                self.W_hr(h) + 
                x_proj
            )

        if return_diagnostics:
            trajectory.append(h.detach())
        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Model 1: Two-Stream Gated (S12-014 Cand 2 Exact Math, Precomputed x) ---
class Model1_TwoStreamGated(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Model 1: Two-Stream Gated (S12-014)"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_fwd = nn.Linear(d_model * 2, d_model, bias=True)
        self.W_loc = nn.Linear(d_model * 2, d_model, bias=True)
        self.W_g = nn.Linear(d_model * 2, d_model, bias=True)

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
        nn.init.normal_(self.W_fwd.weight, 0.0, 0.02)
        nn.init.normal_(self.W_loc.weight, 0.0, 0.02)
        nn.init.normal_(self.W_g.weight, 0.0, 0.02)
        nn.init.constant_(self.W_g.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        D = self.d_model
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        # Mathematically identical block-matrix decomposition:
        # W_fwd * [h_left; x_own] = W_fwd[:, :D] * h_left + (W_fwd[:, D:] * x_own + bias)
        x_fwd_bias = F.linear(x_own, self.W_fwd.weight[:, D:], self.W_fwd.bias)
        x_loc_bias = F.linear(x_left, self.W_loc.weight[:, D:], self.W_loc.bias)
        W_fwd_h = self.W_fwd.weight[:, :D]
        W_loc_h = self.W_loc.weight[:, :D]

        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            v_fwd = torch.tanh(F.linear(h_left, W_fwd_h) + x_fwd_bias)
            v_loc = torch.tanh(F.linear(h, W_loc_h) + x_loc_bias)
            g = torch.sigmoid(self.W_g(torch.cat([v_fwd, v_loc], dim=-1)))
            h = g * v_fwd + (1.0 - g) * v_loc

        if return_diagnostics:
            trajectory.append(h.detach())
        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# =================================================================================================
# 3. TRAINING & EVALUATION HARNESS
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
    D = model.d_model

    pos = torch.arange(0, L, device=dev).unsqueeze(0)
    tok_emb = model.tok(eval_x)
    pos_emb = model.pos(pos)
    x_emb = tok_emb + pos_emb
    x_emb.retain_grad()

    x_own = model.ln_in(x_emb)
    x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

    if isinstance(model, Model0_FlatSum):
        x_proj = model.W_xl(x_left) + model.W_xr(x_own)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hl(h_left) + model.W_hr(h) + x_proj)
    elif isinstance(model, Model1_TwoStreamGated):
        x_fwd_bias = F.linear(x_own, model.W_fwd.weight[:, D:], model.W_fwd.bias)
        x_loc_bias = F.linear(x_left, model.W_loc.weight[:, D:], model.W_loc.bias)
        W_fwd_h = model.W_fwd.weight[:, :D]
        W_loc_h = model.W_loc.weight[:, :D]
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            v_fwd = torch.tanh(F.linear(h_left, W_fwd_h) + x_fwd_bias)
            v_loc = torch.tanh(F.linear(h, W_loc_h) + x_loc_bias)
            g = torch.sigmoid(model.W_g(torch.cat([v_fwd, v_loc], dim=-1)))
            h = g * v_fwd + (1.0 - g) * v_loc

    out = h + model.final_mlp(model.ln_mlp(h))
    logits = model.head(model.ln_f(out))

    dummy_target = torch.randint(0, model.vocab_size, (B,), device=dev)
    loss_target = F.cross_entropy(logits[:, target_pos, :], dummy_target)
    loss_target.backward()

    grad_norms = x_emb.grad.norm(dim=-1).mean(dim=0)
    return grad_norms


def train_candidate(cand_cls, cand_name, train_data, val_data, vocab_size, seq_len, T, device, max_steps=1500):
    print("\n" + "=" * 95)
    print(f"🚀 TRAINING {cand_name.upper()}")
    print("=" * 95)

    torch.manual_seed(42)
    model = cand_cls(vocab_size=vocab_size, seq_len=seq_len, d_model=128, T=T).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    s12_002_params = 190208
    delta = total_params - s12_002_params
    print(f"Parameters: {total_params:,} (vs S12-002: {s12_002_params:,} | Delta: {delta:+,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    batch_size = 64
    eval_interval = 250

    start_time = time.time()
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

    elapsed_total = time.time() - start_time

    # Diagnostics
    eval_x, _ = get_batch(val_data, batch_size=32, seq_len=seq_len, device=device)
    model.eval()
    _, trajectory = model(eval_x, return_diagnostics=True)
    velocities = analyze_field_velocity(trajectory)
    grad_norms = analyze_gradient_reach(model, eval_x, target_pos=63)

    g_d1 = grad_norms[62].item()
    g_d5 = grad_norms[58].item()
    g_d20 = grad_norms[43].item()
    g_d60 = grad_norms[3].item()

    v_hop1 = velocities[0]
    v_hop20 = velocities[min(19, len(velocities) - 1)]

    # Final evaluation on 50 batches
    val_loss_final = evaluate_loss(model, val_data, batch_size, seq_len, device, eval_iters=50)
    ppl_final = math.exp(min(val_loss_final, 20.0))

    return {
        "name": cand_name,
        "params": total_params,
        "val_loss": val_loss_final,
        "ppl": ppl_final,
        "time": elapsed_total,
        "v_hop1": v_hop1,
        "v_hop20": v_hop20,
        "g_d1": g_d1,
        "g_d5": g_d5,
        "g_d20": g_d20,
        "g_d60": g_d60,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Study S12-016 (The Titans Duel: Flat Sum vs Two-Stream) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    candidates = [
        (Model0_FlatSum, "Model 0: Flat Sum (S12-013 Optimized)"),
        (Model1_TwoStreamGated, "Model 1: Two-Stream Gated (S12-014 Optimized)"),
    ]

    results = []
    for cand_cls, cand_name in candidates:
        res = train_candidate(
            cand_cls, cand_name, train_data, val_data, vocab_size, seq_len, T, device, max_steps=1500
        )
        results.append(res)

    print("\n" + "=" * 125)
    print("🏆 FINAL TOURNAMENT LEADERBOARD: S12-016 TITANS DUEL")
    print("=" * 125)
    header = (
        f"{'Model':<35} | {'Params':<10} | {'Val Loss':<9} | {'Val PPL':<8} | {'Time':<6} | "
        f"{'Grad d=1':<10} | {'Grad d=20':<10} | {'Grad d=60':<10}"
    )
    print(header)
    print("-" * 125)

    sorted_res = sorted(results, key=lambda r: r["val_loss"])
    for r in sorted_res:
        line = (
            f"{r['name']:<35} | {r['params']:<10,d} | {r['val_loss']:<9.4f} | {r['ppl']:<8.2f} | {r['time']:<5.1f}s | "
            f"{r['g_d1']:<10.2e} | {r['g_d20']:<10.2e} | {r['g_d60']:<10.2e}"
        )
        print(line)
    print("=" * 125)


if __name__ == "__main__":
    main()
