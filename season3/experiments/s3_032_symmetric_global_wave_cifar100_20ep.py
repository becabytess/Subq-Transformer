"""S3-032: True Symmetric Global Wave on High-Res CIFAR-100 (20 Epochs).

Tests the 20-epoch ceiling of the True Symmetric Global Wave ViT (Bilateral Spatial Radiation),
benchmarking against:
- S2-068 (Canonical SubQ ViT, 20 Epochs): 45.67%
- S3-007 (Scaled Q.K FEN-SubQ, 20 Epochs, d=192, 372k params): 49.37%
- S3-013 (Inter-Hop MLP FEN-SubQ, 20 Epochs, 520k params): 50.15%
- Standard 4-Layer Dense ViT (20 Epochs): 50.37%

Configuration:
- Dataset: High-Res CIFAR-100 (L=256, 2x2 patches, 100 classes)
- Hidden Dim: d=128, Hops T=4, Epochs=20, Batch Size=128
- Bilateral Wave Router: P=4 positive peaks (K=9 bilateral candidates: anchor 0, ±Δ_p)
- Pooling: Global Average Pooling (GAP) across all 256 patches
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season3-s3-032-symmetric-wave-20ep")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_symmetric_global_wave_20ep_experiment(
    seed: int = 42,
    epochs: int = 20,
    n_hops: int = 4,
    p_peaks: int = 4,
    num_waves: int = 12,
    d_model: int = 128,
    mlp_ratio: int = 2,
    batch_size: int = 128,
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

    seq_len = 256
    k_total = 1 + 2 * p_peaks  # 9 candidates
    num_classes = 100

    print("=" * 95)
    print("  S3-032: TRUE SYMMETRIC GLOBAL WAVE ON HIGH-RES CIFAR-100 (20 EPOCHS)")
    print(f"  Seq Len L = {seq_len} | Dim = {d_model} | Epochs = {epochs} | Hops T = {n_hops} | K = {k_total} (P = {p_peaks} pairs)")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

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

    # 2. Symmetric Global Wave ViT Architecture
    class SymmetricGlobalWaveViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_hops = n_hops
            self.p_peaks = p_peaks
            self.k_total = k_total
            self.num_waves = num_waves
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = seq_len

            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.q_proj = nn.Linear(d_model, d_model)

            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            self.init_wave_latent = nn.Parameter(torch.randn(num_waves * 4) * 0.1)
            self.wave_transition = nn.Sequential(
                nn.Linear(num_waves * 4, 32),
                nn.GELU(),
                nn.Linear(32, num_waves * 4),
            )

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, self.max_d - 1, 1))

            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            mlp_dim = d_model * mlp_ratio
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model),
            )

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)

        def forward(self, x, return_diagnostics: bool = False):
            B = x.shape[0]
            tokens = self.patch(x).flatten(2).transpose(1, 2) + self.pos

            H, _ = self.scanner(tokens)
            G = torch.sigmoid(self.gate(H))
            Q_raw = self.q_proj(G * H)
            E_q_all = torch.cumsum(Q_raw, dim=1)

            state = H
            curr_wave = self.init_wave_latent
            pos_grid = torch.arange(seq_len, device=x.device).unsqueeze(1)

            diagnostics = {
                "offsets": [],
                "wave_reaches": [],
                "attn_entropies": [],
            }

            for hop in range(self.n_hops):
                params = curr_wave.view(self.num_waves, 4)
                amp = torch.tanh(params[:, 0]).view(1, 1, self.num_waves)
                omega = (F.softplus(params[:, 1]).view(1, 1, self.num_waves) * self.base_freqs)
                phi = (params[:, 2] * math.pi).view(1, 1, self.num_waves)
                decay = (F.softplus(params[:, 3]) * 0.05).view(1, 1, self.num_waves)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=-1).squeeze(0)

                topk_vals, past_offsets = torch.topk(wave_1d, k=self.p_peaks, dim=-1)
                past_offsets = past_offsets + 1

                targets_minus = (pos_grid - past_offsets.unsqueeze(0)) % seq_len
                targets_plus = (pos_grid + past_offsets.unsqueeze(0)) % seq_len
                targets = torch.cat([pos_grid, targets_minus, targets_plus], dim=-1)

                zero_val = torch.zeros(1, dtype=torch.float, device=x.device)
                peak_vals = torch.cat([zero_val, topk_vals, topk_vals])

                K_discrete = self.k_proj(state)
                V_discrete = self.v_proj(state)

                K_cand = K_discrete[:, targets, :]
                V_cand = V_discrete[:, targets, :]

                q = self.ln_q(E_q_all + state)
                k = self.ln_k(K_cand)
                v = V_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals.view(1, 1, self.k_total)
                weights = F.softmax(scores, dim=-1)

                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                next_state = state + self.scale * context
                next_state = next_state + self.scale * self.mlp(self.ln_mlp(next_state))
                state = next_state

                if return_diagnostics:
                    diagnostics["offsets"].append(past_offsets.detach().cpu().tolist())
                    mean_decay = decay.mean().item()
                    diagnostics["wave_reaches"].append(round(1.0 / max(mean_decay, 1e-5), 1))
                    entropy = -(weights * (weights + 1e-9).log()).sum(dim=-1).mean().item()
                    diagnostics["attn_entropies"].append(round(entropy, 3))

                if hop < self.n_hops - 1:
                    curr_wave = curr_wave + 0.1 * self.wave_transition(curr_wave)

            pooled = state.mean(dim=1)
            logits = self.head(self.ln_f(pooled))

            if return_diagnostics:
                return logits, diagnostics
            return logits

    # 3. Training Loop
    model = SymmetricGlobalWaveViT().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)

    def get_lr(step):
        if step < warmup_steps:
            return 1e-3 * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))

    epoch_records = []
    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        ep_t0 = time.time()

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += bx.size(0)
            global_step += 1

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        diag = None
        with torch.no_grad():
            for i, (tx, ty) in enumerate(test_loader):
                tx, ty = tx.to(device), ty.to(device)
                if i == 0:
                    t_logits, diag = model(tx, return_diagnostics=True)
                else:
                    t_logits = model(tx)
                preds = t_logits.argmax(dim=-1)
                test_correct += (preds == ty).sum().item()
                test_total += tx.size(0)

        test_acc = 100.0 * test_correct / test_total
        ep_time = time.time() - ep_t0

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | Train: {train_loss:.4f} ({train_acc:.2f}%) | "
            f"Test Acc: {test_acc:.2f}% | Offsets: {diag['offsets'][0]} | Time: {ep_time:.1f}s",
            flush=True
        )

        epoch_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_acc": round(test_acc, 2),
            "offsets_hop1": diag["offsets"][0] if diag else [],
            "offsets_hop4": diag["offsets"][-1] if diag else [],
            "time_s": round(ep_time, 1),
        })

    total_time = time.time() - t0
    final_test_acc = epoch_records[-1]["test_acc"]
    best_test_acc = max(r["test_acc"] for r in epoch_records)

    print("=" * 95)
    print(f"  S3-032 COMPLETE: Final Top-1: {final_test_acc:.2f}% | Best: {best_test_acc:.2f}% | Total Time: {total_time:.1f}s")
    print("=" * 95)

    return {
        "study": "S3-032",
        "description": "True Symmetric Global Wave on High-Res CIFAR-100 (20 Epochs, Bilateral Radiation)",
        "seq_len": seq_len,
        "d_model": d_model,
        "n_hops": n_hops,
        "p_peaks": p_peaks,
        "k_total": k_total,
        "num_params": param_count,
        "epochs": epochs,
        "total_time_s": round(total_time, 1),
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "epoch_records": epoch_records,
        "diagnostics": diag,
    }


@app.local_entrypoint()
def main():
    print("Launching S3-032 True Symmetric Global Wave on CIFAR-100 (20 Epochs on Modal A10G)...")
    result = run_symmetric_global_wave_20ep_experiment.remote()

    os.makedirs("season3/results", exist_ok=True)
    out_path = "season3/results/s3_032_symmetric_global_wave_cifar100_20ep.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] Results saved to {out_path}")
