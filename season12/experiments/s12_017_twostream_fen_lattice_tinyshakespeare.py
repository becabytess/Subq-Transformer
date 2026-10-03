"""
===================================================================================================
STUDY S12-017: TWO-STREAM FEATURE ESCROW NETWORK (FEN) TOURNAMENT
Task: TinyShakespeare Character-Level LM | L = 64, T = 64 | 1500 steps each
Baseline Reference: S12-002 (190,208 params, PPL 5.53)
Incumbent Champion: S12-013 Flat Sum (190,080 params, PPL 5.26, Val Loss 1.6605)
FEN Reference: S12-011 Canonical FEN (189,909 params, PPL 5.41, Grad d=60: 1.42e-3)

THE ARCHITECTURAL BLUEPRINT:
1. Phase 1: 2D Parallel Lattice (Across T=64 hops, zero spatial loops)
   - Track 1 (Forward Wave h_fwd): h_fwd(i) = tanh( W_hf * h_fwd(i-1) + W_xf * x_i )
     Ditches own state, keeping the clean diagonal causal RNN hand-off.
   - Track 2 (Backward Counter-Wave h_bwd): h_bwd(i) = tanh( W_v * h_fwd(i) + W_xl * x_{i-1} )
     Stores the local backward verification check in parallel across all tokens.

2. Phase 2: Post-Lattice Escrow Accumulation (Runs ONCE after all hops finish)
   - Sweeps horizontally from token 0 to 63 (strictly left-to-right for airtight causality).
   - Candidate 1 (Backward Escrow): h_fwd is kept local; h_bwd is accumulated into Escrow E_bwd.
   - Candidate 2 (Dual Escrow): Both h_fwd and h_bwd are accumulated into E_fwd and E_bwd.
   - Candidate 3 (Forward Escrow): h_bwd is kept local; h_fwd is accumulated into Escrow E_fwd.
   - Candidate 0 (Flat Sum Baseline): S12-013 4-way sum benchmark (1.6605).

All models strictly <= 190,208 parameters.
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
# 2. TOURNAMENT ARCHITECTURES
# =================================================================================================

# --- Candidate 0: Flat Sum (S12-013 Baseline Champion) ---
class Cand0_FlatSum(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=384):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 0: Flat Sum (S12-013)"

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

        # Static terms precomputed once outside loop
        x_proj = self.W_xl(x_left) + self.W_xr(x_own)
        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_hl(h_left) + self.W_hr(h) + x_proj)

        if return_diagnostics:
            trajectory.append(h.detach())
        out = h + self.final_mlp(self.ln_mlp(h))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Candidate 1: Backward Escrow (Core Idea: h_fwd + E_bwd) ---
class Cand1_BackwardEscrow(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=255):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 1: Backward Escrow (h_fwd + E_bwd)"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # Track 1: Forward Wave (steps i-1 -> i, ditches own state)
        self.W_hf = nn.Linear(d_model, d_model, bias=False)
        self.W_xf = nn.Linear(d_model, d_model, bias=True)

        # Track 2: Backward Verification (takes state at i and char from i-1)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=True)

        # FEN Roll Gate for Escrow
        self.roll_gate = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2  # 256
        self.ln_mlp = nn.LayerNorm(d_fused)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_fused, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_v.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        # Static character projections precomputed once
        x_fwd_bias = self.W_xf(x_own)
        x_loc_bias = self.W_xl(x_left)

        # Phase 1: 2D Recurrent Lattice across T=64 hops (Zero spatial loops!)
        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            # Clean diagonal forward wave: steps i-1 -> i, ditches own state
            h = torch.tanh(self.W_hf(h_left) + x_fwd_bias)

        if return_diagnostics:
            trajectory.append(h.detach())

        # Track 2: Backward Verification Check across all tokens in parallel
        v_loc = torch.tanh(self.W_v(h) + x_loc_bias)

        # Phase 2: Post-Lattice FEN Escrow Accumulation (Runs ONCE from token 0 to 63)
        escrow = torch.zeros(B, self.d_model, device=idx.device)
        escrow_list = []
        for i in range(L):
            v_i = v_loc[:, i, :]
            gamma_i = torch.sigmoid(self.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)

        escrow_seq = torch.stack(escrow_list, dim=1)  # (B, L, d_model)

        # Phase 3: Head receives [h_fwd ; E_bwd]
        fused = torch.cat([h, escrow_seq], dim=-1)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Candidate 2: Dual Escrow (Both Accumulate: E_fwd + E_bwd) ---
class Cand2_DualEscrow(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=254):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 2: Dual Escrow (E_fwd + E_bwd)"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_hf = nn.Linear(d_model, d_model, bias=False)
        self.W_xf = nn.Linear(d_model, d_model, bias=True)

        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=True)

        # Two roll gates for the two independent escrows
        self.roll_gate_fwd = nn.Linear(d_model, 1, bias=True)
        self.roll_gate_bwd = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2  # 256
        self.ln_mlp = nn.LayerNorm(d_fused)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_fused, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_v.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate_fwd.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate_fwd.bias, 0.0)
        nn.init.normal_(self.roll_gate_bwd.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate_bwd.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        x_fwd_bias = self.W_xf(x_own)
        x_loc_bias = self.W_xl(x_left)

        # Phase 1: 2D Lattice across T=64 hops
        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_hf(h_left) + x_fwd_bias)

        if return_diagnostics:
            trajectory.append(h.detach())

        v_loc = torch.tanh(self.W_v(h) + x_loc_bias)

        # Phase 2: Dual FEN Escrow Accumulation
        e_fwd = torch.zeros(B, self.d_model, device=idx.device)
        e_bwd = torch.zeros(B, self.d_model, device=idx.device)
        e_fwd_list = []
        e_bwd_list = []

        for i in range(L):
            h_i = h[:, i, :]
            v_i = v_loc[:, i, :]

            # Forward Escrow
            gf_i = torch.sigmoid(self.roll_gate_fwd(h_i))
            rolled_f = torch.roll(e_fwd, shifts=1, dims=-1)
            e_fwd = (1.0 - gf_i) * e_fwd + gf_i * rolled_f + h_i
            e_fwd_list.append(e_fwd)

            # Backward Escrow
            gb_i = torch.sigmoid(self.roll_gate_bwd(v_i))
            rolled_b = torch.roll(e_bwd, shifts=1, dims=-1)
            e_bwd = (1.0 - gb_i) * e_bwd + gb_i * rolled_b + v_i
            e_bwd_list.append(e_bwd)

        e_fwd_seq = torch.stack(e_fwd_list, dim=1)
        e_bwd_seq = torch.stack(e_bwd_list, dim=1)

        # Phase 3: Head receives [E_fwd ; E_bwd]
        fused = torch.cat([e_fwd_seq, e_bwd_seq], dim=-1)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Candidate 3: Forward Escrow (Forward Accumulates: E_fwd + h_bwd) ---
class Cand3_ForwardEscrow(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=255):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 3: Forward Escrow (E_fwd + h_bwd)"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_hf = nn.Linear(d_model, d_model, bias=False)
        self.W_xf = nn.Linear(d_model, d_model, bias=True)

        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=True)

        self.roll_gate = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2  # 256
        self.ln_mlp = nn.LayerNorm(d_fused)
        self.final_mlp = nn.Sequential(
            nn.Linear(d_fused, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_v.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        x_fwd_bias = self.W_xf(x_own)
        x_loc_bias = self.W_xl(x_left)

        # Phase 1: 2D Lattice across T=64 hops
        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_hf(h_left) + x_fwd_bias)

        if return_diagnostics:
            trajectory.append(h.detach())

        v_loc = torch.tanh(self.W_v(h) + x_loc_bias)

        # Phase 2: Forward FEN Escrow Accumulation on h
        escrow = torch.zeros(B, self.d_model, device=idx.device)
        escrow_list = []
        for i in range(L):
            h_i = h[:, i, :]
            gamma_i = torch.sigmoid(self.roll_gate(h_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + h_i
            escrow_list.append(escrow)

        escrow_seq = torch.stack(escrow_list, dim=1)

        # Phase 3: Head receives [E_fwd ; h_bwd]
        fused = torch.cat([escrow_seq, v_loc], dim=-1)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# =================================================================================================
# 3. MATHEMATICAL CAUSALITY VERIFICATION
# =================================================================================================
def verify_model_causality(model, vocab_size=65, seq_len=64, device="cpu"):
    model.eval()
    test_positions = [10, 25, 45]
    all_passed = True
    x = torch.randint(0, vocab_size, (2, seq_len), device=device)

    with torch.no_grad():
        out_base = model(x)
        for k in test_positions:
            x_pert = x.clone()
            # Mutate all future tokens strictly after position k
            x_pert[:, k + 1 :] = (x_pert[:, k + 1 :] + 1) % vocab_size
            out_pert = model(x_pert)

            diff_past = (out_base[:, : k + 1, :] - out_pert[:, : k + 1, :]).abs().max().item()
            diff_future = (out_base[:, k + 1 :, :] - out_pert[:, k + 1 :, :]).abs().max().item()

            if diff_past > 0.0 or diff_future == 0.0:
                all_passed = False
                print(f"CAUSALITY VIOLATION at k={k}! Past diff: {diff_past:.10e}")

    return all_passed


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
    tok_emb = model.tok(eval_x)
    pos_emb = model.pos(pos)
    x_emb = tok_emb + pos_emb
    x_emb.retain_grad()

    x_own = model.ln_in(x_emb)
    x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

    if isinstance(model, Cand0_FlatSum):
        x_proj = model.W_xl(x_left) + model.W_xr(x_own)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hl(h_left) + model.W_hr(h) + x_proj)
        out = h + model.final_mlp(model.ln_mlp(h))

    elif isinstance(model, Cand1_BackwardEscrow):
        x_fwd_bias = model.W_xf(x_own)
        x_loc_bias = model.W_xl(x_left)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hf(h_left) + x_fwd_bias)
        v_loc = torch.tanh(model.W_v(h) + x_loc_bias)

        escrow = torch.zeros(B, model.d_model, device=dev)
        escrow_list = []
        for i in range(L):
            v_i = v_loc[:, i, :]
            gamma_i = torch.sigmoid(model.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)
        escrow_seq = torch.stack(escrow_list, dim=1)
        fused = torch.cat([h, escrow_seq], dim=-1)
        out = h + model.final_mlp(model.ln_mlp(fused))

    elif isinstance(model, Cand2_DualEscrow):
        x_fwd_bias = model.W_xf(x_own)
        x_loc_bias = model.W_xl(x_left)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hf(h_left) + x_fwd_bias)
        v_loc = torch.tanh(model.W_v(h) + x_loc_bias)

        e_fwd = torch.zeros(B, model.d_model, device=dev)
        e_bwd = torch.zeros(B, model.d_model, device=dev)
        e_fwd_list = []
        e_bwd_list = []
        for i in range(L):
            h_i = h[:, i, :]
            v_i = v_loc[:, i, :]
            gf_i = torch.sigmoid(model.roll_gate_fwd(h_i))
            rolled_f = torch.roll(e_fwd, shifts=1, dims=-1)
            e_fwd = (1.0 - gf_i) * e_fwd + gf_i * rolled_f + h_i
            e_fwd_list.append(e_fwd)

            gb_i = torch.sigmoid(model.roll_gate_bwd(v_i))
            rolled_b = torch.roll(e_bwd, shifts=1, dims=-1)
            e_bwd = (1.0 - gb_i) * e_bwd + gb_i * rolled_b + v_i
            e_bwd_list.append(e_bwd)
        fused = torch.cat([torch.stack(e_fwd_list, dim=1), torch.stack(e_bwd_list, dim=1)], dim=-1)
        out = h + model.final_mlp(model.ln_mlp(fused))

    elif isinstance(model, Cand3_ForwardEscrow):
        x_fwd_bias = model.W_xf(x_own)
        x_loc_bias = model.W_xl(x_left)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hf(h_left) + x_fwd_bias)
        v_loc = torch.tanh(model.W_v(h) + x_loc_bias)

        escrow = torch.zeros(B, model.d_model, device=dev)
        escrow_list = []
        for i in range(L):
            h_i = h[:, i, :]
            gamma_i = torch.sigmoid(model.roll_gate(h_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + h_i
            escrow_list.append(escrow)
        escrow_seq = torch.stack(escrow_list, dim=1)
        fused = torch.cat([escrow_seq, v_loc], dim=-1)
        out = h + model.final_mlp(model.ln_mlp(fused))

    logits = model.head(model.ln_f(out))
    dummy_target = torch.randint(0, model.vocab_size, (B,), device=dev)
    loss_target = F.cross_entropy(logits[:, target_pos, :], dummy_target)
    loss_target.backward()

    grad_norms = x_emb.grad.norm(dim=-1).mean(dim=0)
    return grad_norms


def train_candidate(cand_cls, cand_name, train_data, val_data, vocab_size, seq_len, T, device, max_steps=1500):
    print("\n" + "=" * 95)
    print(f"INITIALIZING & VERIFYING {cand_name.upper()}")
    print("=" * 95)

    torch.manual_seed(42)
    model = cand_cls(vocab_size=vocab_size, seq_len=seq_len, d_model=128, T=T).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    s12_002_params = 190208
    delta = total_params - s12_002_params
    print(f"Parameters: {total_params:,} (vs S12-002: {s12_002_params:,} | Delta: {delta:+,} params)")

    causal_ok = verify_model_causality(model, vocab_size=vocab_size, seq_len=seq_len, device=device)
    if causal_ok:
        print("CAUSALITY VERIFICATION: PASSED (100% Strictly Causal. Max Past Perturbation = 0.00000000)")
    else:
        raise RuntimeError("FATAL: Causality violation detected!")

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
    print(f"Starting Study S12-017 (Two-Stream Feature Escrow Tournament) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    candidates = [
        (Cand0_FlatSum, "Cand 0: Flat Sum (S12-013 Champion)"),
        (Cand1_BackwardEscrow, "Cand 1: Backward Escrow (h_fwd + E_bwd)"),
        (Cand2_DualEscrow, "Cand 2: Dual Escrow (E_fwd + E_bwd)"),
        (Cand3_ForwardEscrow, "Cand 3: Forward Escrow (E_fwd + h_bwd)"),
    ]

    results = []
    for cand_cls, cand_name in candidates:
        res = train_candidate(
            cand_cls, cand_name, train_data, val_data, vocab_size, seq_len, T, device, max_steps=1500
        )
        results.append(res)

    print("\n" + "=" * 125)
    print("FINAL TOURNAMENT LEADERBOARD: S12-017 TWO-STREAM FEN SHOOTOUT")
    print("=" * 125)
    header = (
        f"{'Candidate':<40} | {'Params':<10} | {'Val Loss':<9} | {'Val PPL':<8} | {'Time':<6} | "
        f"{'Grad d=1':<10} | {'Grad d=20':<10} | {'Grad d=60':<10}"
    )
    print(header)
    print("-" * 125)

    sorted_res = sorted(results, key=lambda r: r["val_loss"])
    for r in sorted_res:
        line = (
            f"{r['name']:<40} | {r['params']:<10,d} | {r['val_loss']:<9.4f} | {r['ppl']:<8.2f} | {r['time']:<5.1f}s | "
            f"{r['g_d1']:<10.2e} | {r['g_d20']:<10.2e} | {r['g_d60']:<10.2e}"
        )
        print(line)
    print("=" * 125)


if __name__ == "__main__":
    main()
