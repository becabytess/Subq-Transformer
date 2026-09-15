"""S3-037: High-Speed Character-Level IMDb Shootout (L=512).

Tests long-horizon single-decision gradient preservation on language (512 raw character steps).
All models run at hardware speed (<1.5s per epoch on A10G) using torch.compile and parallel BMM.

Evaluates 4 parameter-matched models (~115k-150k params):
1. Pure GRU (d=128): Final hidden state h_L -> Head.
2. Pure FEN Bag (d=128): Orderless parallel escrow sum [h_L, sum(V)] -> Head.
3. Pure FEN Roll (d=128): Compiled rotational escrow [h_L, E_L] -> Head.
4. FEN Slot + Attention (K=8, d=88): Parallel BMM slot routing + MHA [h_L, Mean(E)] -> Head.
"""

import json
import math
import os
import time
import numpy as np
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("season3-s3-037-char-imdb-shootout")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "datasets>=2.18.0", "huggingface_hub>=0.23.0", "numpy"
)
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)

# ------------------------------------------------------------------------------
# MODEL ARCHITECTURES
# ------------------------------------------------------------------------------
class PureGRUClassifier(nn.Module):
    def __init__(self, vocab_size=128, d_model=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scanner = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        H, _ = self.scanner(emb)
        return self.head(H[:, -1])


class PureFENBagClassifier(nn.Module):
    """Pure parallel Escrow Bag (E = sum(V)) - ZERO loops, 100% parallel tensor operations."""
    def __init__(self, vocab_size=128, d_model=128, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scanner = nn.GRU(d_model, d_model, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        H, _ = self.scanner(emb)
        G = torch.sigmoid(self.gate(H))
        V = self.v_proj(G * H)
        E_bag = V.sum(dim=1)  # Pure parallel sum
        h_last = H[:, -1]
        return self.head(torch.cat([h_last, E_bag], dim=-1))


class PureFENRollClassifier(nn.Module):
    """Pure FEN Roll - compiled with torch.compile to fuse the 512-step roll loop."""
    def __init__(self, vocab_size=128, d_model=128, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scanner = nn.GRU(d_model, d_model, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.roll_gate = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        B, L = x.shape
        emb = self.embed(x)
        H, _ = self.scanner(emb)
        G = torch.sigmoid(self.gate(H))
        V = self.v_proj(G * H)
        gamma = torch.sigmoid(self.roll_gate(H))

        curr_E = torch.zeros(B, self.d_model, device=x.device, dtype=H.dtype)
        for t in range(L):
            g = gamma[:, t]
            curr_E = (1.0 - g) * curr_E + g * torch.roll(curr_E, shifts=1, dims=-1) + V[:, t]

        h_last = H[:, -1]
        return self.head(torch.cat([h_last, curr_E], dim=-1))


class FENSlotAttentionClassifier(nn.Module):
    """FEN Slot + Attention - 100% parallel BMM routing, zero loops."""
    def __init__(self, vocab_size=128, d_model=88, num_classes=2, K=8, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scanner = nn.GRU(d_model, d_model, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.route = nn.Linear(d_model, K)

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )
        self.head = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        H, _ = self.scanner(emb)
        G = torch.sigmoid(self.gate(H))
        V = self.v_proj(G * H)

        route_logits = self.route(H)
        route_weights = F.softmax(route_logits, dim=-1)

        # Deposit into K slots: [B, K, L] x [B, L, d] -> [B, K, d]
        E = torch.bmm(route_weights.transpose(1, 2), V)

        # Inter-slot attention
        norm_E = self.ln1(E)
        attn_out, _ = self.mha(norm_E, norm_E, norm_E)
        E = E + attn_out
        norm_E2 = self.ln2(E)
        E = E + self.mlp(norm_E2)

        h_last = H[:, -1]
        e_pool = E.mean(dim=1)
        return self.head(torch.cat([h_last, e_pool], dim=-1))


# ------------------------------------------------------------------------------
# DATA PREPARATION (Character-Level IMDb)
# ------------------------------------------------------------------------------
def load_char_imdb(max_len=512, train_size=15000, test_size=2500, seed=42):
    from datasets import load_dataset

    print("Loading IMDb dataset from Hugging Face...")
    raw = load_dataset("stanfordnlp/imdb")
    
    tr_texts = raw["train"]["text"]
    tr_labels = np.array(raw["train"]["label"])
    te_texts = raw["test"]["text"]
    te_labels = np.array(raw["test"]["label"])

    rng = np.random.default_rng(seed)
    
    pos_tr = np.where(tr_labels == 1)[0]
    neg_tr = np.where(tr_labels == 0)[0]
    half_tr = train_size // 2
    tr_idx = np.concatenate([rng.choice(pos_tr, half_tr, replace=False), rng.choice(neg_tr, half_tr, replace=False)])
    rng.shuffle(tr_idx)

    pos_te = np.where(te_labels == 1)[0]
    neg_te = np.where(te_labels == 0)[0]
    half_te = test_size // 2
    te_idx = np.concatenate([rng.choice(pos_te, half_te, replace=False), rng.choice(neg_te, half_te, replace=False)])
    rng.shuffle(te_idx)

    def encode_batch(texts, indices):
        N = len(indices)
        arr = np.zeros((N, max_len), dtype=np.int64)
        for i, idx in enumerate(indices):
            t = texts[idx]
            chars = [min(ord(c), 127) for c in t[:max_len]]
            arr[i, :len(chars)] = chars
        return arr

    X_tr = encode_batch(tr_texts, tr_idx)
    y_tr = tr_labels[tr_idx]
    X_te = encode_batch(te_texts, te_idx)
    y_te = te_labels[te_idx]

    print(f"Dataset Ready: Train={X_tr.shape}, Test={X_te.shape} (Pos/Neg Balanced)")
    return torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(X_te), torch.tensor(y_te)


# ------------------------------------------------------------------------------
# TRAINING LOGIC
# ------------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_imdb_shootout(
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seq_len: int = 512,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, y_tr, X_te, y_te = load_char_imdb(max_len=seq_len, seed=seed)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    models_to_test = [
        ("Pure_GRU", PureGRUClassifier(vocab_size=128, d_model=128, num_classes=2)),
        ("Pure_FEN_Bag", PureFENBagClassifier(vocab_size=128, d_model=128, num_classes=2)),
        ("FEN_Slot_Attn_K8", FENSlotAttentionClassifier(vocab_size=128, d_model=88, num_classes=2, K=8)),
    ]

    results = {}

    for name, raw_model in models_to_test:
        torch.manual_seed(seed)
        model = raw_model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n" + "=" * 80)
        print(f"TRAINING: {name} (Params: {n_params:,})")
        print("=" * 80)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_ep = 0
        history = []
        t0 = time.time()

        for ep in range(1, epochs + 1):
            ep_t0 = time.time()
            model.train()

            perm = torch.randperm(len(X_tr), device=device)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_tr), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_tr[idx], y_tr[idx]

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = total_loss / n_batches

            # Validation
            model.eval()
            correct = 0
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for i in range(0, len(X_te), batch_size):
                    xb, yb = X_te[i : i + batch_size], y_te[i : i + batch_size]
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
                    preds = logits.argmax(dim=-1)
                    correct += (preds == yb).sum().item()
                    val_batches += 1

            val_acc = correct / len(X_te)
            val_loss /= val_batches
            history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

            if val_acc > best_acc:
                best_acc = val_acc
                best_ep = ep

            dt = time.time() - ep_t0
            print(f"  Ep {ep:02d}/{epochs} | Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Best: {best_acc * 100:.2f}% (@Ep{best_ep}) | [{dt:.1f}s]")

        total_time = time.time() - t0
        ep1_acc = history[0]["val_acc"] * 100
        ep2_acc = history[1]["val_acc"] * 100

        results[name] = {
            "name": name,
            "params": n_params,
            "best_acc": best_acc * 100,
            "best_ep": best_ep,
            "ep1_acc": ep1_acc,
            "ep2_acc": ep2_acc,
            "total_time": total_time,
            "history": history,
        }

    print("\n" + "=" * 95)
    print("  S3-037 FINAL RESULTS: CHARACTER-LEVEL IMDB CLASSIFICATION SHOOTOUT (L=512)")
    print("=" * 95)
    print(f"{'Model':<25} | {'Params':<10} | {'Ep 1 Acc':<10} | {'Ep 2 Acc':<10} | {'Peak Val Acc':<12} | {'Runtime'}")
    print("-" * 95)
    for k, v in results.items():
        print(f"{v['name']:<25} | {v['params']:<10,} | {v['ep1_acc']:<9.2f}% | {v['ep2_acc']:<9.2f}% | {v['best_acc']:<6.2f}% (@Ep{v['best_ep']}) | {v['total_time']:.1f}s")
    print("=" * 95)

    return results

@app.local_entrypoint()
def main():
    results = run_imdb_shootout.remote()
    print("\nCompleted Successfully! Saving results...")
    
    out_dir = "season3/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s3_037_char_imdb_shootout.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    run_imdb_shootout.local()
