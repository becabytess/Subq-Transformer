"""S2-042: Fast FEN-SubQ Fusion on High-Res CIFAR-100 (L=257 tokens).

Fully vectorized and fused implementation:
Phase 1: Fused Causal RNN (cuDNN) sequence scan -> Parallel FEN Extraction -> Fused Cumsum Escrow Vault.
Phase 2: Discrete SubQ multi-hop execution over K=9 escrow states (Zero Q, Zero K, Zero Softmax).

Tests:
1. fen_subq_rnn: Fused Tanh-RNN causal scanner + Parallel Bag Escrow + 4-hop SubQ execution.
2. fen_subq_gru: Fused GRU causal scanner + Parallel Bag Escrow + 4-hop SubQ execution.

Known Baselines for comparison (10 epochs, seed 42):
- Historical FEN (exp11 at T=256): FEN Bag = 3.1%, FEN Roll = 14.9%
- S2-030 Canonical SubQ (QKV Softmax, L=257): 33.0%
- S2-030 Dense 4-Layer ViT (All-to-All, L=257): 42.1%
"""

import json
import math
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season2-s2-042-fen-subq-cifar100")


@app.function(image=image, gpu="A10G", timeout=3600)
def run_fen_subq_experiment(
    rnn_type: str = "rnn",  # "rnn" or "gru"
    seed: int = 42,
    epochs: int = 10,
    n_hops: int = 4,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 128
    d_mlp = 384
    offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]  # K = 9 offsets
    batch_size = 128
    seq_len = 257  # 256 patches (2x2, stride 2 on 32x32) + 1 CLS token
    num_classes = 100

    print("=" * 90)
    print(f"  S2-042: Fast FEN-SubQ Fusion ViT (Scanner: {rnn_type.upper()}) on CIFAR-100")
    print(f"  Seq Len L = {seq_len}, Dim = {d_model}, Hops = {n_hops}, Offsets K = {len(offsets)}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 90)

    # 1. Dataset
    raw = load_dataset("uoft-cs/cifar100")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    class Cifar(Dataset):
        def __init__(self, split_name, transform):
            self.items, self.transform = raw[split_name], transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            return self.transform(item.get("img", item.get("image"))), item.get(
                "fine_label", item.get("label")
            )

    train_loader = DataLoader(
        Cifar("train", train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Cifar("test", test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Fast FEN-SubQ Model (Fused cuDNN + Parallel Escrow)
    class FastFENSubQViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 1.0 / math.sqrt(len(offsets))

            # Patch projection
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

            # Phase 1: Fused Causal Scanner
            if rnn_type == "gru":
                self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            else:
                self.scanner = nn.RNN(d_model, d_model, nonlinearity="tanh", batch_first=True)

            # Parallel FEN Feature Extraction from hidden states
            self.gate = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)

            # Phase 2: Discrete SubQ Multi-Hop Execution (No QKV, No Softmax)
            self.ln_h = nn.LayerNorm(d_model)
            self.ln_e = nn.LayerNorm(d_model)
            self.proj_h = nn.Linear(d_model, d_model)
            self.proj_e = nn.Linear(d_model, d_model)
            self.exec_core = nn.Linear(d_model, d_model)
            self.exec_gate = nn.Linear(d_model, d_model)
            self.exec_out = nn.Linear(d_model, d_model)

            # End-MLP
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model),
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)

            # SubQ offset lookup table
            targets = torch.zeros((seq_len, len(offsets)), dtype=torch.long)
            for pos in range(seq_len):
                for j, offset in enumerate(offsets):
                    targets[pos, j] = (pos - offset) % seq_len
            self.register_buffer("targets", targets)

        def forward(self, x):
            bsz = x.shape[0]
            # Embed image into tokens
            tokens = self.patch(x).flatten(2).transpose(1, 2)  # [B, 256, D]
            tokens = torch.cat([self.cls.expand(bsz, -1, -1), tokens], dim=1) + self.pos  # [B, 257, D]

            # -------------------------------------------------------------
            # PHASE 1: Fused Causal Scan + Parallel Escrow Vault
            # -------------------------------------------------------------
            # 1. Fused cuDNN RNN pass (zero Python loop)
            H, _ = self.scanner(tokens)  # [B, L, D]

            # 2. Parallel FEN extraction across all positions simultaneously
            G = torch.sigmoid(self.gate(H))  # [B, L, D]
            V = self.v_proj(G * H)           # [B, L, D]

            # 3. Fused parallel cumulative-sum escrow vault
            E = torch.cumsum(V, dim=1)       # [B, L, D]

            # -------------------------------------------------------------
            # PHASE 2: Discrete SubQ Multi-Hop Execution (Zero QKV, Zero Softmax)
            # -------------------------------------------------------------
            # Gather K historical escrow snapshots: [B, L, K, D]
            E_cand = E[:, self.targets, :]

            state = H
            for hop in range(n_hops):
                h_norm = self.ln_h(state)
                e_norm = self.ln_e(E_cand)

                # Joint program execution across K escrow candidates
                z_exec = self.proj_h(h_norm).unsqueeze(2) + self.proj_e(e_norm)
                f_exec = torch.tanh(self.exec_core(z_exec) + z_exec)
                g_exec = torch.sigmoid(self.exec_gate(f_exec))
                delta = self.exec_out((g_exec * f_exec).sum(dim=2)) * self.scale
                state = state + delta

            # End-MLP
            state = state + self.mlp(self.ln_mlp(state))

            # Readout from [CLS] token
            return self.head(self.ln_f(state)[:, 0])

    model = FastFENSubQViT().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epoch_records = []
    t0_train = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0_ep = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total

        # Evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                pred = logits.argmax(dim=-1)
                test_correct += (pred == labels).sum().item()
                test_total += imgs.size(0)

        test_acc = 100.0 * test_correct / test_total
        ep_time = time.time() - t0_ep

        print(
            f"  Epoch {ep:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}% | Time: {ep_time:.1f}s"
        )

        epoch_records.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_acc": round(test_acc, 2),
            "time_s": round(ep_time, 2),
        })

    total_time = time.time() - t0_train
    print("-" * 90)
    print(f"Final Test Accuracy ({rnn_type.upper()}): {epoch_records[-1]['test_acc']:.2f}% in {total_time:.1f}s")
    print("-" * 90)

    return {
        "rnn_type": rnn_type,
        "seed": seed,
        "parameters": num_params,
        "epochs": epochs,
        "n_hops": n_hops,
        "epoch_records": epoch_records,
        "final_test_acc": epoch_records[-1]["test_acc"],
        "total_time_s": round(total_time, 2),
    }


@app.local_entrypoint()
def main(scanner: str = "both"):
    scanners_to_run = ["rnn", "gru"] if scanner == "both" else [scanner]
    results = {}

    for sc in scanners_to_run:
        print(f"\n>>> Launching Fast FEN-SubQ ViT ({sc.upper()} Scanner) on Modal A10G...")
        res = run_fen_subq_experiment.remote(rnn_type=sc, seed=42, epochs=10, n_hops=4)
        results[sc] = res
        print(f">>> Completed {sc.upper()}: Final Accuracy = {res['final_test_acc']}%\n")

    # Summary table
    print("=" * 80)
    print("  S2-042 FEN-SUBQ CIFAR-100 FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Architecture':<25} | {'Params':<10} | {'Ep 1 Acc':<10} | {'Ep 10 Acc':<10} | {'Time (s)':<10}")
    print("-" * 80)
    print(f"{'S2-030 SubQ ViT (Control)':<25} | {'229,120':<10} | {'14.5%':<10} | {'33.0%':<10} | {'325s':<10}")
    print(f"{'Historical FEN Bag (P2)':<25} | {'~100,000':<10} | {'--':<10} | {'3.1%':<10} | {'--':<10}")
    print(f"{'Historical FEN Roll (P2)':<25} | {'~100,000':<10} | {'--':<10} | {'14.9%':<10} | {'--':<10}")
    print("-" * 80)
    for sc, res in results.items():
        name = f"FEN-SubQ ({sc.upper()})"
        ep1 = f"{res['epoch_records'][0]['test_acc']:.2f}%"
        ep10 = f"{res['final_test_acc']:.2f}%"
        params = f"{res['parameters']:,}"
        t = f"{res['total_time_s']:.1f}s"
        print(f"{name:<25} | {params:<10} | {ep1:<10} | {ep10:<10} | {t:<10}")
    print("=" * 80)

    # Save to json
    with open("season2/results/s2_042_fen_subq_cifar100.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to season2/results/s2_042_fen_subq_cifar100.json")
