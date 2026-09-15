"""S3-036: Pure GRU Baseline on TinyShakespeare (Ablation for FEN Roll).

Tests whether the 5.28 PPL achieved by Pure FEN Roll (S3-035) was genuinely aided by
the Escrow Roll trajectory E_t, or whether the cuDNN GRU hidden state alone (H)
achieves the same or better performance.

Evaluates:
1. Pure GRU (d=128, 115,777 params) - Exact backbone used in S3-035, minus Escrow E.
2. Parameter-Matched Pure GRU (d=152, 159,361 params) - Perfectly matched to S3-035's 157k params.
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

app = modal.App("season3-s3-036-pure-gru-tinyshakespeare")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy"
)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

# ------------------------------------------------------------------------------
# MODEL ARCHITECTURES: Pure GRU Baselines
# ------------------------------------------------------------------------------
class PureGRULM(nn.Module):
    def __init__(self, vocab_size=65, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.scanner = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embed(x)  # [B, L, d]
        H, _ = self.scanner(emb)  # [B, L, d]
        return self.head(H)

# ------------------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_pure_gru_experiment(
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 250,
    val_batches: int = 30,
    seq_len: int = 256,
    batch_size: int = 32,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def train_single_model(d_model, name):
        torch.manual_seed(seed)
        model = PureGRULM(vocab_size=vocab_size, d_model=d_model).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTraining {name} (d={d_model}, Params: {n_params:,})...")

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

        best_val_loss, best_val_ppl = evaluate()
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
                    f"  [{name}] Step {step:04d}/{total_steps} | Train: {loss.item():.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Best PPL: {best_val_ppl:.2f} | [{dt:.1f}s]"
                )

        return {
            "name": name,
            "d_model": d_model,
            "params": n_params,
            "best_val_loss": best_val_loss,
            "best_val_ppl": best_val_ppl,
            "best_step": best_step,
            "runtime": time.time() - t0,
        }

    # Model 1: Exact GRU Backbone from FEN (d=128)
    res_d128 = train_single_model(d_model=128, name="Pure_GRU_d128")

    # Model 2: Parameter-Matched Pure GRU (d=152 ~159k params)
    res_d152 = train_single_model(d_model=152, name="Pure_GRU_d152_ParamMatched")

    print("\n" + "=" * 95)
    print("  EXPERIMENT RESULTS: PURE GRU VS PURE FEN ROLL ON TINYSHAKESPEARE")
    print("=" * 95)
    print("Benchmark Comparison (L=256, 2,000 steps):")
    print(f"  - S3-035 (Pure FEN Roll, d=128):             1.6647 Loss | 5.28 PPL (157k params, 160s)")
    print(f"  - S3-036 (Pure GRU, d=128):                  {res_d128['best_val_loss']:.4f} Loss | {res_d128['best_val_ppl']:.2f} PPL ({res_d128['params']:,} params, {res_d128['runtime']:.1f}s)")
    print(f"  - S3-036 (Pure GRU Param-Matched, d=152):    {res_d152['best_val_loss']:.4f} Loss | {res_d152['best_val_ppl']:.2f} PPL ({res_d152['params']:,} params, {res_d152['runtime']:.1f}s)")
    print("=" * 95)

    return {"gru_d128": res_d128, "gru_d152": res_d152}

@app.local_entrypoint()
def main():
    result = run_pure_gru_experiment.remote()
    print("Completed! Summary:")
    print(json.dumps(result, indent=2))
    
    out_dir = "season3/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s3_036_pure_gru_tinyshakespeare.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_pure_gru_experiment.local()
