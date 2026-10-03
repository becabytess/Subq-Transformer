"""
===================================================================================================
STUDY S12-018: FLAT SUM + FEATURE ESCROW NETWORK (FEN) EXTRACTORS TOURNAMENT
Task: TinyShakespeare Character-Level LM | L = 64, T = 64 | 1500 steps each
Reference Baseline (No need to re-run): S12-013 Flat Sum Champion
  - Val Loss: 1.6605 (PPL 5.26 | 50-batch: 1.6706)
  - Grad reach d=60: 1.43e-20 (vanished)
  - Time: 57.6s | Params: 190,080

THE ARCHITECTURAL BLUEPRINT:
1. Phase 1: 2D Flat Sum Recurrent Lattice across T=64 hops (Zero spatial loops!)
   - All 4 cross-symmetric forces interact at every hop:
     h = tanh( W_hl * h_left + W_hr * h + W_xl * x_left + W_xr * x_own )
   - Static terms precomputed once outside the loop for maximum GPU speed.

2. Phase 2: Post-Lattice FEN Escrow Accumulation (Runs ONCE after all hops finish)
   - Sweeps horizontally from token 0 to 63 (strictly left-to-right for airtight causality).
   - Candidate 1: FlatSum + Backward Verification Extractor (v_bwd = tanh(W_vb * h + W_xb * x_left))
   - Candidate 2: FlatSum + Canonical Gated FEN Extractor (v_fen = sigmoid(W_gate * h) * tanh(W_val * h))
   - Candidate 3: FlatSum + Linear Extractor (v_ext = tanh(W_ext * h), maximizing MLP head capacity)
   - Candidate 4: FlatSum + Dual Escrow (Both E_fwd and E_bwd accumulated and fused)

All models strictly <= 190,208 parameters.
Flat Sum baseline is preserved from prior studies and NOT re-run to save GPU compute.
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
# 2. TOURNAMENT ARCHITECTURES (ALL ON TOP OF FLAT SUM LATTICE)
# =================================================================================================

# --- Candidate 1: Flat Sum + Backward Verification Extractor ---
class Cand1_FlatSumBwdEscrow(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=169):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 1: FlatSum + Bwd Escrow"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        # 2D Flat Sum Lattice
        self.W_hl = nn.Linear(d_model, d_model, bias=False)
        self.W_hr = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=False)
        self.W_xr = nn.Linear(d_model, d_model, bias=True)

        # Backward Verification Extractor
        self.W_vb = nn.Linear(d_model, d_model, bias=False)
        self.W_xb = nn.Linear(d_model, d_model, bias=True)

        # FEN Roll Gate
        self.roll_gate = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2
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
        nn.init.normal_(self.W_hl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hr.weight, 0.0, 0.01)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xr.weight, 0.0, 0.02)
        nn.init.normal_(self.W_vb.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xb.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        # Static Flat Sum terms precomputed once outside loop
        x_proj = self.W_xl(x_left) + self.W_xr(x_own)
        x_bwd_proj = self.W_xb(x_left)

        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_hl(h_left) + self.W_hr(h) + x_proj)

        if return_diagnostics:
            trajectory.append(h.detach())

        # Post-Lattice Backward Extractor
        v_loc = torch.tanh(self.W_vb(h) + x_bwd_proj)

        # Horizontal FEN Escrow Accumulation (0 to L-1)
        escrow = torch.zeros(B, self.d_model, device=idx.device)
        escrow_list = []
        for i in range(L):
            v_i = v_loc[:, i, :]
            gamma_i = torch.sigmoid(self.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)

        escrow_seq = torch.stack(escrow_list, dim=1)
        fused = torch.cat([h, escrow_seq], dim=-1)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Candidate 2: Flat Sum + Canonical Gated FEN Extractor ---
class Cand2_FlatSumGatedFEN(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=169):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 2: FlatSum + Gated FEN"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_hl = nn.Linear(d_model, d_model, bias=False)
        self.W_hr = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=False)
        self.W_xr = nn.Linear(d_model, d_model, bias=True)

        # Canonical Gated FEN Information Extractor
        self.extract_gate = nn.Linear(d_model, d_model, bias=True)
        self.extract_val = nn.Linear(d_model, d_model, bias=False)

        self.roll_gate = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2
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
        nn.init.normal_(self.W_hl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hr.weight, 0.0, 0.01)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xr.weight, 0.0, 0.02)
        nn.init.normal_(self.extract_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.extract_gate.bias, 0.0)
        nn.init.normal_(self.extract_val.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

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

        # Canonical Gated Extraction: v = sigmoid(gate) * tanh(val)
        v_fen = torch.sigmoid(self.extract_gate(h)) * torch.tanh(self.extract_val(h))

        escrow = torch.zeros(B, self.d_model, device=idx.device)
        escrow_list = []
        for i in range(L):
            v_i = v_fen[:, i, :]
            gamma_i = torch.sigmoid(self.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)

        escrow_seq = torch.stack(escrow_list, dim=1)
        fused = torch.cat([h, escrow_seq], dim=-1)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Candidate 3: Flat Sum + Linear Extractor (Max Head Capacity) ---
class Cand3_FlatSumLinearExtractor(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=212):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 3: FlatSum + Linear Extractor"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_hl = nn.Linear(d_model, d_model, bias=False)
        self.W_hr = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=False)
        self.W_xr = nn.Linear(d_model, d_model, bias=True)

        self.W_extract = nn.Linear(d_model, d_model, bias=True)
        self.roll_gate = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2
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
        nn.init.normal_(self.W_hl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hr.weight, 0.0, 0.01)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xr.weight, 0.0, 0.02)
        nn.init.normal_(self.W_extract.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

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

        v_ext = torch.tanh(self.W_extract(h))

        escrow = torch.zeros(B, self.d_model, device=idx.device)
        escrow_list = []
        for i in range(L):
            v_i = v_ext[:, i, :]
            gamma_i = torch.sigmoid(self.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)

        escrow_seq = torch.stack(escrow_list, dim=1)
        fused = torch.cat([h, escrow_seq], dim=-1)
        out = h + self.final_mlp(self.ln_mlp(fused))
        logits = self.head(self.ln_f(out))
        if return_diagnostics:
            return logits, trajectory
        return logits


# --- Candidate 4: Flat Sum + Dual Escrow ---
class Cand4_FlatSumDualEscrow(nn.Module):
    def __init__(self, vocab_size=65, seq_len=64, d_model=128, T=64, d_mlp=127):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.T = T
        self.name = "Cand 4: FlatSum + Dual Escrow"

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)

        self.W_hl = nn.Linear(d_model, d_model, bias=False)
        self.W_hr = nn.Linear(d_model, d_model, bias=False)
        self.W_xl = nn.Linear(d_model, d_model, bias=False)
        self.W_xr = nn.Linear(d_model, d_model, bias=True)

        self.W_vf = nn.Linear(d_model, d_model, bias=False)
        self.W_vb = nn.Linear(d_model, d_model, bias=False)
        self.W_xb = nn.Linear(d_model, d_model, bias=True)

        self.roll_gate_fwd = nn.Linear(d_model, 1, bias=True)
        self.roll_gate_bwd = nn.Linear(d_model, 1, bias=True)

        d_fused = d_model * 2
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
        nn.init.normal_(self.W_hl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_hr.weight, 0.0, 0.01)
        nn.init.normal_(self.W_xl.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xr.weight, 0.0, 0.02)
        nn.init.normal_(self.W_vf.weight, 0.0, 0.02)
        nn.init.normal_(self.W_vb.weight, 0.0, 0.02)
        nn.init.normal_(self.W_xb.weight, 0.0, 0.02)
        nn.init.normal_(self.roll_gate_fwd.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate_fwd.bias, 0.0)
        nn.init.normal_(self.roll_gate_bwd.weight, 0.0, 0.02)
        nn.init.constant_(self.roll_gate_bwd.bias, 0.0)

    def forward(self, idx, return_diagnostics=False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x_own = self.ln_in(self.tok(idx) + self.pos(pos))
        x_left = F.pad(x_own[:, :-1, :], (0, 0, 1, 0))

        x_proj = self.W_xl(x_left) + self.W_xr(x_own)
        x_bwd_proj = self.W_xb(x_left)

        h = x_own.clone()
        trajectory = [] if return_diagnostics else None

        for hop in range(self.T):
            if return_diagnostics:
                trajectory.append(h.detach())
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(self.W_hl(h_left) + self.W_hr(h) + x_proj)

        if return_diagnostics:
            trajectory.append(h.detach())

        v_fwd = torch.tanh(self.W_vf(h))
        v_bwd = torch.tanh(self.W_vb(h) + x_bwd_proj)

        e_fwd = torch.zeros(B, self.d_model, device=idx.device)
        e_bwd = torch.zeros(B, self.d_model, device=idx.device)
        e_fwd_list = []
        e_bwd_list = []

        for i in range(L):
            vf_i = v_fwd[:, i, :]
            vb_i = v_bwd[:, i, :]

            gf_i = torch.sigmoid(self.roll_gate_fwd(vf_i))
            rolled_f = torch.roll(e_fwd, shifts=1, dims=-1)
            e_fwd = (1.0 - gf_i) * e_fwd + gf_i * rolled_f + vf_i
            e_fwd_list.append(e_fwd)

            gb_i = torch.sigmoid(self.roll_gate_bwd(vb_i))
            rolled_b = torch.roll(e_bwd, shifts=1, dims=-1)
            e_bwd = (1.0 - gb_i) * e_bwd + gb_i * rolled_b + vb_i
            e_bwd_list.append(e_bwd)

        fused = torch.cat([torch.stack(e_fwd_list, dim=1), torch.stack(e_bwd_list, dim=1)], dim=-1)
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

    if isinstance(model, Cand1_FlatSumBwdEscrow):
        x_proj = model.W_xl(x_left) + model.W_xr(x_own)
        x_bwd_proj = model.W_xb(x_left)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hl(h_left) + model.W_hr(h) + x_proj)
        v_loc = torch.tanh(model.W_vb(h) + x_bwd_proj)
        escrow = torch.zeros(B, model.d_model, device=dev)
        escrow_list = []
        for i in range(L):
            v_i = v_loc[:, i, :]
            gamma_i = torch.sigmoid(model.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)
        fused = torch.cat([h, torch.stack(escrow_list, dim=1)], dim=-1)
        out = h + model.final_mlp(model.ln_mlp(fused))

    elif isinstance(model, Cand2_FlatSumGatedFEN):
        x_proj = model.W_xl(x_left) + model.W_xr(x_own)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hl(h_left) + model.W_hr(h) + x_proj)
        v_fen = torch.sigmoid(model.extract_gate(h)) * torch.tanh(model.extract_val(h))
        escrow = torch.zeros(B, model.d_model, device=dev)
        escrow_list = []
        for i in range(L):
            v_i = v_fen[:, i, :]
            gamma_i = torch.sigmoid(model.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)
        fused = torch.cat([h, torch.stack(escrow_list, dim=1)], dim=-1)
        out = h + model.final_mlp(model.ln_mlp(fused))

    elif isinstance(model, Cand3_FlatSumLinearExtractor):
        x_proj = model.W_xl(x_left) + model.W_xr(x_own)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hl(h_left) + model.W_hr(h) + x_proj)
        v_ext = torch.tanh(model.W_extract(h))
        escrow = torch.zeros(B, model.d_model, device=dev)
        escrow_list = []
        for i in range(L):
            v_i = v_ext[:, i, :]
            gamma_i = torch.sigmoid(model.roll_gate(v_i))
            rolled = torch.roll(escrow, shifts=1, dims=-1)
            escrow = (1.0 - gamma_i) * escrow + gamma_i * rolled + v_i
            escrow_list.append(escrow)
        fused = torch.cat([h, torch.stack(escrow_list, dim=1)], dim=-1)
        out = h + model.final_mlp(model.ln_mlp(fused))

    elif isinstance(model, Cand4_FlatSumDualEscrow):
        x_proj = model.W_xl(x_left) + model.W_xr(x_own)
        x_bwd_proj = model.W_xb(x_left)
        h = x_own.clone()
        for hop in range(model.T):
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            h = torch.tanh(model.W_hl(h_left) + model.W_hr(h) + x_proj)
        v_fwd = torch.tanh(model.W_vf(h))
        v_bwd = torch.tanh(model.W_vb(h) + x_bwd_proj)
        e_fwd = torch.zeros(B, model.d_model, device=dev)
        e_bwd = torch.zeros(B, model.d_model, device=dev)
        e_fwd_list = []
        e_bwd_list = []
        for i in range(L):
            vf_i = v_fwd[:, i, :]
            vb_i = v_bwd[:, i, :]
            gf_i = torch.sigmoid(model.roll_gate_fwd(vf_i))
            rolled_f = torch.roll(e_fwd, shifts=1, dims=-1)
            e_fwd = (1.0 - gf_i) * e_fwd + gf_i * rolled_f + vf_i
            e_fwd_list.append(e_fwd)

            gb_i = torch.sigmoid(model.roll_gate_bwd(vb_i))
            rolled_b = torch.roll(e_bwd, shifts=1, dims=-1)
            e_bwd = (1.0 - gb_i) * e_bwd + gb_i * rolled_b + vb_i
            e_bwd_list.append(e_bwd)
        fused = torch.cat([torch.stack(e_fwd_list, dim=1), torch.stack(e_bwd_list, dim=1)], dim=-1)
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
    print(f"Starting Study S12-018 (Flat Sum + FEN Extractors Tournament) on {device}...")
    torch.manual_seed(42)

    train_data, val_data, vocab_size, _, _ = load_tinyshakespeare()
    seq_len = 64
    T = 64

    # Notice: Flat Sum baseline is firmly established and NOT re-run to save compute!
    candidates = [
        (Cand1_FlatSumBwdEscrow, "Cand 1: FlatSum + Bwd Escrow"),
        (Cand2_FlatSumGatedFEN, "Cand 2: FlatSum + Gated FEN"),
        (Cand3_FlatSumLinearExtractor, "Cand 3: FlatSum + Linear Extractor"),
        (Cand4_FlatSumDualEscrow, "Cand 4: FlatSum + Dual Escrow"),
    ]

    results = []
    for cand_cls, cand_name in candidates:
        res = train_candidate(
            cand_cls, cand_name, train_data, val_data, vocab_size, seq_len, T, device, max_steps=1500
        )
        results.append(res)

    print("\n" + "=" * 125)
    print("FINAL TOURNAMENT LEADERBOARD: S12-018 FLAT SUM + FEN EXTRACTORS")
    print("=" * 125)
    header = (
        f"{'Candidate':<40} | {'Params':<10} | {'Val Loss':<9} | {'Val PPL':<8} | {'Time':<6} | "
        f"{'Grad d=1':<10} | {'Grad d=20':<10} | {'Grad d=60':<10}"
    )
    print(header)
    print("-" * 125)
    print(
        f"{'Reference: S12-013 Flat Sum (Benchmark)':<40} | {'190,080':<10} | {'1.6706':<9} | {'5.32':<8} | {'57.6s':<6} | "
        f"{'4.63e-01':<10} | {'1.15e-04':<10} | {'1.43e-20':<10}"
    )
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
