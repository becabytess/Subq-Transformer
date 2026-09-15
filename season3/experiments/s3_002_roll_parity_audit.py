"""S2-043: FEN-SubQ with Channel-Roll Escrow & Strict Parameter Parity Shootout.

Answers two fundamental questions:
1. Does Channel-Roll Escrow improve over pure Bag Escrow on High-Res CIFAR-100?
2. At STRICT parameter parity (~229k and ~262k), does GRU scanner have a true algorithmic advantage over Tanh-RNN, or was the prior gain just capacity?

Conditions:
1. fen_subq_rnn_roll_229k: Tanh-RNN + Roll Escrow (Target: 229,120 params, matching Canonical SubQ S2-030).
2. fen_subq_rnn_roll_262k: Tanh-RNN + Roll Escrow (Target: 262,000 params, d_mlp=256).
3. fen_subq_gru_roll_262k: GRU + Roll Escrow (Target: 262,000 params, exact match to condition 2).

All conditions:
- CIFAR-100, Patch 2x2 (L=257 tokens), 10 epochs, seed 42.
- Modal A10G.
"""

import json
import math
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season2-s2-043-fen-subq-roll-parity")


@app.function(image=image, gpu="A10G", timeout=3600)
def run_roll_parity_experiment(
    condition_name: str,
    rnn_type: str = "rnn",  # "rnn" or "gru"
    d_mlp: int = 128,
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
    offsets = [0, 1, 2, 4, 8, 16, 63, 127, 128]  # K = 9 offsets
    batch_size = 128
    seq_len = 257  # 256 patches (2x2, stride 2 on 32x32) + 1 CLS token
    num_classes = 100

    print("=" * 90)
    print(f"  S2-043: FEN-SubQ Roll Parity — {condition_name}")
    print(f"  Scanner: {rnn_type.upper()}, d_mlp: {d_mlp}, Hops: {n_hops}, SeqLen: {seq_len}")
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

    # 2. FEN-SubQ Model with Channel-Roll Escrow
    class FENSubQRollViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 1.0 / math.sqrt(len(offsets))
            self.has_mlp = (d_mlp > 0)

            # Patch projection
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

            # Phase 1: Fused Causal Scanner
            if rnn_type == "gru":
                self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            else:
                self.scanner = nn.RNN(d_model, d_model, nonlinearity="tanh", batch_first=True)

            # FEN extraction + Roll Gate
            self.gate = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.roll_gate = nn.Linear(d_model, 1)

            # Phase 2: Discrete SubQ Multi-Hop FEN Readout
            self.ln_h = nn.LayerNorm(d_model)
            self.ln_e = nn.LayerNorm(d_model)
            self.proj_h = nn.Linear(d_model, d_model)
            self.proj_e = nn.Linear(d_model, d_model)
            self.exec_core = nn.Linear(d_model, d_model)
            self.exec_gate = nn.Linear(d_model, d_model)
            self.exec_out = nn.Linear(d_model, d_model)

            # End-MLP (optional depending on parameter target)
            if self.has_mlp:
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
            # PHASE 1: Fused Causal Scan + Channel-Roll Escrow
            # -------------------------------------------------------------
            H, _ = self.scanner(tokens)  # [B, L, D]

            # Feature extraction and roll gate (parallel across L)
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            V = self.v_proj(G * H)                     # [B, L, D]
            gamma = torch.sigmoid(self.roll_gate(H))   # [B, L, 1]

            # Channel-Roll Escrow Accumulation (lightweight temporal roll scan)
            E = torch.zeros(bsz, d_model, device=x.device)
            E_list = []
            for t in range(seq_len):
                g_t = gamma[:, t]
                E = (1.0 - g_t) * E + g_t * torch.roll(E, shifts=1, dims=-1) + V[:, t]
                E_list.append(E)
            E_all = torch.stack(E_list, dim=1)         # [B, L, D]

            # -------------------------------------------------------------
            # PHASE 2: Discrete SubQ Multi-Hop Readout
            # -------------------------------------------------------------
            # Gather K historical escrow snapshots: [B, L, K, D]
            E_cand = E_all[:, self.targets, :]

            state = H
            for hop in range(n_hops):
                h_norm = self.ln_h(state)
                e_norm = self.ln_e(E_cand)

                # Readout on each candidate escrow individually
                z_exec = self.proj_h(h_norm).unsqueeze(2) + self.proj_e(e_norm)
                f_exec = torch.tanh(self.exec_core(z_exec) + z_exec)
                g_exec = torch.sigmoid(self.exec_gate(f_exec))
                delta = self.exec_out((g_exec * f_exec).sum(dim=2)) * self.scale
                state = state + delta

            # End-MLP
            if self.has_mlp:
                state = state + self.mlp(self.ln_mlp(state))

            # Readout from [CLS] token
            return self.head(self.ln_f(state)[:, 0])

    model = FENSubQRollViT().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

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
    print(f"Final Test Accuracy ({condition_name}): {epoch_records[-1]['test_acc']:.2f}% in {total_time:.1f}s")
    print("-" * 90)

    return {
        "condition": condition_name,
        "rnn_type": rnn_type,
        "d_mlp": d_mlp,
        "seed": seed,
        "parameters": num_params,
        "epochs": epochs,
        "n_hops": n_hops,
        "epoch_records": epoch_records,
        "final_test_acc": epoch_records[-1]["test_acc"],
        "total_time_s": round(total_time, 2),
    }


@app.local_entrypoint()
def main():
    experiments = [
        # Condition 1: Strict 229k parity with S2-030 Canonical SubQ
        {
            "condition_name": "fen_subq_rnn_roll_229k",
            "rnn_type": "rnn",
            "d_mlp": 128,  # Target: ~229k
        },
        # Condition 2: 262k parity baseline for RNN
        {
            "condition_name": "fen_subq_rnn_roll_262k",
            "rnn_type": "rnn",
            "d_mlp": 256,  # Target: ~262k
        },
        # Condition 3: 262k parity match for GRU
        {
            "condition_name": "fen_subq_gru_roll_262k",
            "rnn_type": "gru",
            "d_mlp": 0,    # Target: ~262k (no End-MLP needed, GRU gates supply capacity)
        },
    ]

    results = {}
    for exp in experiments:
        name = exp["condition_name"]
        print(f"\n>>> Launching {name} on Modal A10G...")
        res = run_roll_parity_experiment.remote(
            condition_name=name,
            rnn_type=exp["rnn_type"],
            d_mlp=exp["d_mlp"],
            seed=42,
            epochs=10,
            n_hops=4,
        )
        results[name] = res
        print(f">>> Completed {name}: Final Accuracy = {res['final_test_acc']}%\n")

    # Summary table
    print("=" * 90)
    print("  S2-043 FEN-SUBQ CHANNEL-ROLL & STRICT PARITY RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Condition':<30} | {'Params':<10} | {'Ep 1 Acc':<10} | {'Ep 10 Acc':<10} | {'Time (s)':<10}")
    print("-" * 90)
    print(f"{'S2-030 Canonical SubQ':<30} | {'229,120':<10} | {'14.5%':<10} | {'33.0%':<10} | {'325s':<10}")
    print(f"{'S2-042 FEN-SubQ RNN (Bag)':<30} | {'295,936':<10} | {'10.51%':<10} | {'31.01%':<10} | {'413s':<10}")
    print(f"{'S2-042 FEN-SubQ GRU (Bag)':<30} | {'361,984':<10} | {'11.36%':<10} | {'35.74%':<10} | {'418s':<10}")
    print("-" * 90)
    for name, res in results.items():
        ep1 = f"{res['epoch_records'][0]['test_acc']:.2f}%"
        ep10 = f"{res['final_test_acc']:.2f}%"
        params = f"{res['parameters']:,}"
        t = f"{res['total_time_s']:.1f}s"
        print(f"{name:<30} | {params:<10} | {ep1:<10} | {ep10:<10} | {t:<10}")
    print("=" * 90)

    with open("season2/results/s2_043_fen_subq_roll_parity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to season2/results/s2_043_fen_subq_roll_parity.json")
