"""S3-035: Pure FEN Roll on TinyShakespeare Causal Language Modeling.

Evaluates Pure FEN Roll with causal per-token escrow trajectory tracking [h_t, E_t]
on TinyShakespeare (L=256, d=128, 2,000 steps).

Benchmark Comparisons (Strict Parity with S3-016 & S3-034):
- S1 Study 58 (Pure SubQ, NO GRU, T=4):        Val Loss = 1.7092, Val PPL = 5.52
- S1 Study 63 (Pure SubQ, NO GRU, T=8):        Val Loss = 1.6790, Val PPL = 5.36
- S3-016 (Inverted Query Escrow + SubQ, T=4):  Val Loss = 1.6535, Val PPL = 5.23
- S3-034 (No Escrow / cuDNN GRU Alone, T=4):   Val Loss = 1.6284, Val PPL = 5.10
- S3-023 (T=12 Scaling):                       Val Loss = 1.6133, Val PPL = 5.02
"""

import json
import math
import os
import time
import urllib.request
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("season3-s3-035-pure-fen-roll-tinyshakespeare")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

# ------------------------------------------------------------------------------
# MODEL ARCHITECTURE: Pure FEN Roll for Causal Language Modeling
# ------------------------------------------------------------------------------
class PureFENRollLM(nn.Module):
    def __init__(self, vocab_size=65, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 1. Recurrent Scanner (cuDNN GRU)
        self.scanner = nn.GRU(d_model, d_model, batch_first=True)
        
        # 2. Intestinal Extraction Gating
        self.gate = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.roll_gate = nn.Linear(d_model, 1)
        
        # 3. Per-Token Joint Readout Head
        self.head = nn.Linear(d_model * 2, vocab_size)

    def forward(self, x):
        B, L = x.shape
        emb = self.embed(x)  # [B, L, d]
        
        # 1. Causal recurrent sweep
        H, _ = self.scanner(emb)  # [B, L, d]
        
        # 2. Parallel feature extraction
        G = torch.sigmoid(self.gate(H))  # [B, L, d]
        V = self.v_proj(G * H)  # [B, L, d]
        gamma = torch.sigmoid(self.roll_gate(H))  # [B, L, 1]
        
        # 3. Causal Escrow Roll Trajectory
        E = torch.zeros(B, L, self.d_model, device=x.device, dtype=H.dtype)
        curr_E = torch.zeros(B, self.d_model, device=x.device, dtype=H.dtype)
        for t in range(L):
            g = gamma[:, t]
            curr_E = (1.0 - g) * curr_E + g * torch.roll(curr_E, shifts=1, dims=-1) + V[:, t]
            E[:, t] = curr_E
            
        # 4. Readout logits for each token position
        logits = self.head(torch.cat([H, E], dim=-1))  # [B, L, vocab_size]
        return logits

# ------------------------------------------------------------------------------
# TRAINING LOGIC (Strict Parity with S3-016 & S3-034)
# ------------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_pure_fen_roll_tinyshakespeare(
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 250,
    val_batches: int = 30,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 95)
    print("  S3-035: PURE FEN ROLL ON TINYSHAKESPEARE (CAUSAL NEXT-TOKEN PREDICTION)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Steps = {total_steps}, Batch Size = {batch_size}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    # 1. Dataset Setup
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.urlopen(url)
    text = req.read().decode("utf-8")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    def get_batch(split, step_seed=None):
        if step_seed is not None:
            torch.manual_seed(step_seed)
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix])
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    model = PureFENRollLM(vocab_size=vocab_size, d_model=d_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,} (d={d_model})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    def evaluate():
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for vb in range(val_batches):
                x_val, y_val = get_batch("val", step_seed=10000 + vb)
                logits = model(x_val)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y_val.view(-1))
                total_loss += loss.item()
        val_loss = total_loss / val_batches
        val_ppl = math.exp(min(val_loss, 20.0))
        return val_loss, val_ppl

    # Initial evaluation
    val_loss_init, val_ppl_init = evaluate()
    print(f"Initial Check: Val Loss = {val_loss_init:.4f} | Val PPL = {val_ppl_init:.2f}")

    eval_history = [{"step": 0, "val_loss": val_loss_init, "val_ppl": val_ppl_init}]
    best_val_loss = val_loss_init
    best_val_ppl = val_ppl_init
    best_step = 0

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        x_train, y_train = get_batch("train")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_train.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == total_steps:
            val_loss, val_ppl = evaluate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                best_step = step

            dt = time.time() - t0
            print(
                f"Step {step:04d}/{total_steps} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                f"Best PPL: {best_val_ppl:.2f} (@step {best_step}) | Time: {dt:.1f}s"
            )

    total_time = time.time() - t0

    print("\n" + "=" * 95)
    print("  S3-035 EXPERIMENT RESULTS: PURE FEN ROLL ON TINYSHAKESPEARE")
    print("=" * 95)
    print(f"  Model Parameters:       {n_params:,}")
    print(f"  Best Val Loss:          {best_val_loss:.4f}")
    print(f"  Best Val Perplexity:    {best_val_ppl:.2f} (at step {best_step})")
    print(f"  Final Val Perplexity:   {val_ppl:.2f}")
    print(f"  Total Runtime:          {total_time:.1f}s")
    print("-" * 95)
    print("Benchmark Comparison on TinyShakespeare:")
    print("  - S1 Study 58 (Pure SubQ, NO GRU, T=4):        1.7092 Loss | 5.52 PPL")
    print("  - S1 Study 63 (Pure SubQ, NO GRU, T=8):        1.6790 Loss | 5.36 PPL")
    print("  - S3-016 (Inverted Query Escrow + SubQ, T=4):  1.6535 Loss | 5.23 PPL")
    print("  - S3-034 (cuDNN GRU Alone, NO Escrow, T=4):    1.6284 Loss | 5.10 PPL")
    print(f"  - S3-035 (Pure FEN Roll, NO SubQ/Attn):        {best_val_loss:.4f} Loss | {best_val_ppl:.2f} PPL")
    print("=" * 95)

    return {
        "model": "PureFENRollLM",
        "params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
        "best_step": best_step,
        "total_time": total_time,
        "eval_history": eval_history,
    }

@app.local_entrypoint()
def main():
    result = run_pure_fen_roll_tinyshakespeare.remote()
    print("Run completed successfully! Summary:")
    print(json.dumps({k: v for k, v in result.items() if k != "eval_history"}, indent=2))
    
    out_dir = "season3/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s3_035_pure_fen_roll_tinyshakespeare.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    run_pure_fen_roll_tinyshakespeare.local()
