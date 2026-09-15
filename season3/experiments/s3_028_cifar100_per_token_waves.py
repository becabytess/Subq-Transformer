"""S3-028: State-Conditioned Per-Token Wave Router on High-Res CIFAR-100 (10 Epochs, T=4 Hops).

Scientific Hypothesis:
By shifting from a single bottleneck [CLS] token to Global Average Pooling (GAP),
image classification becomes a global shared responsibility:
1. Every single patch receives direct gradient backpropagation (\nabla_{s_i} Loss != 0).
2. Every patch's personal continuous 12-carrier harmonic wave router trains immediately.
3. The image sequence is a clean 16x16 = 256 grid, aligned with 2D image strides.

Benchmark:
- High-Res CIFAR-100 (L=256, 2x2 patches, 100 classes)
- Dim D = 128, Hops T = 4, Peaks K = 8, Epochs = 10, Batch Size = 128
- Comparison benchmarks:
  * S3-006 (10 Epochs, D=128, Global Wave): 37.36%
  * S3-007 (10 Epochs checkpoint, D=192, Global Wave): 45.08%
  * S3-017 (Inverted Global Wave, Epoch 10): 45.84% (20 Ep final: 49.56%)
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "torchvision>=0.17.0", "datasets>=2.18.0", "numpy", "pillow"
)
app = modal.App("season3-s3-028-cifar100-per-token-gap")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_cifar100_per_token_experiment(
    seed: int = 42,
    epochs: int = 10,
    n_hops: int = 4,
    k_peaks: int = 8,
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

    seq_len = 256  # 256 patches (16x16 grid, 2x2 patches on 32x32)
    num_classes = 100

    print("=" * 105)
    print("  S3-028: STATE-CONDITIONED PER-TOKEN WAVE ROUTER ON CIFAR-100 (GLOBAL AVERAGE POOLING)")
    print(f"  Seq Len L = {seq_len}, Dim D = {d_model}, Epochs = {epochs}, Hops T = {n_hops}, K = {k_peaks} Peaks")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        Cifar("test", test_transform),
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # 2. State-Conditioned Per-Token Wave ViT Model with Global Average Pooling
    class StateConditionedPerTokenGAPViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.n_hops = n_hops
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = seq_len

            # Patch projection (2x2 non-overlapping patches)
            self.patch = nn.Conv2d(3, d_model, kernel_size=2, stride=2)
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

            # Phase 1: Causal cuDNN GRU Scan -> Cumulative Query Escrow
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.q_proj = nn.Linear(d_model, d_model)

            # Phase 2: Discrete Keys and Values
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            # State-Conditioned Per-Token Wave Emission Head
            self.ln_wave = nn.LayerNorm(d_model)
            self.wave_head = nn.Linear(d_model, num_waves * 4)

            log_freqs = torch.linspace(0.0, -2.0, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves, 1))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, 1, self.max_d - 1))

            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            mlp_dim = d_model * mlp_ratio
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model),
            )

            # Global Output Head
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes, bias=False)

        def forward(self, x, return_diagnostics: bool = False):
            B = x.shape[0]
            tokens = self.patch(x).flatten(2).transpose(1, 2) + self.pos  # [B, 256, D]

            H, _ = self.scanner(tokens)
            G = torch.sigmoid(self.gate(H))
            Q_raw = self.q_proj(G * H)
            E_q_all = torch.cumsum(Q_raw, dim=1)

            state = H
            pos_grid = torch.arange(seq_len, device=x.device).view(1, seq_len, 1)

            diagnostics = {
                "offset_std_per_token": [],
                "mean_attended_distances": [],
                "attn_entropies": [],
            }

            for hop in range(self.n_hops):
                wave_params = self.wave_head(self.ln_wave(state)).view(B, seq_len, self.num_waves, 4)
                amp = torch.tanh(wave_params[..., 0]).unsqueeze(-1)
                omega = F.softplus(wave_params[..., 1]).unsqueeze(-1) * self.base_freqs
                phi = (wave_params[..., 2] * math.pi).unsqueeze(-1)
                decay = (F.softplus(wave_params[..., 3]) * 0.05).unsqueeze(-1)

                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=2)  # [B, L, max_d - 1]

                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros((B, seq_len, 1), dtype=torch.long, device=x.device)
                zero_val = torch.zeros((B, seq_len, 1), dtype=torch.float, device=x.device)

                active_offsets = torch.cat([zero_off, past_offsets], dim=-1)  # [B, L, K]
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)          # [B, L, K]

                # 2D Periodic patch grid wrapping
                targets = (pos_grid - active_offsets) % seq_len

                K_discrete = self.k_proj(state)
                V_discrete = self.v_proj(state)

                idx = targets.view(B, seq_len * self.k_peaks, 1).expand(-1, -1, d_model)
                K_cand = torch.gather(K_discrete, 1, idx).view(B, seq_len, self.k_peaks, d_model)
                V_cand = torch.gather(V_discrete, 1, idx).view(B, seq_len, self.k_peaks, d_model)

                q = self.ln_q(E_q_all + state)
                k = self.ln_k(K_cand)
                v = V_cand

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals
                weights = F.softmax(scores, dim=-1)

                context = (weights.unsqueeze(-1) * v).sum(dim=2)

                next_state = state + self.scale * context
                next_state = next_state + self.scale * self.mlp(self.ln_mlp(next_state))
                state = next_state

                if return_diagnostics:
                    std_off = active_offsets[:, :, 1:].float().std(dim=1).mean().item()
                    diagnostics["offset_std_per_token"].append(std_off)
                    mean_dist = (weights * active_offsets.float()).sum(dim=-1).mean().item()
                    diagnostics["mean_attended_distances"].append(mean_dist)
                    entropy = -(weights * (weights + 1e-9).log()).sum(dim=-1).mean().item()
                    diagnostics["attn_entropies"].append(round(entropy, 3))

            # Global Average Pooling across all 256 patches
            pooled = state.mean(dim=1)
            logits = self.head(self.ln_f(pooled))

            if return_diagnostics:
                return logits, diagnostics
            return logits

    model = StateConditionedPerTokenGAPViT().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    @torch.inference_mode()
    def evaluate():
        model.eval()
        correct, total = 0, 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(imgs).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / total

    epoch_records = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t_ep = time.time()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        test_acc = evaluate()
        ep_time = time.time() - t_ep

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"Time: {ep_time:.1f}s",
            flush=True
        )

        epoch_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_acc": round(test_acc, 2),
            "time_s": round(ep_time, 1),
        })

    # Validation diagnostics on a test batch
    with torch.no_grad():
        sample_imgs, _ = next(iter(test_loader))
        sample_imgs = sample_imgs.to(device)
        _, final_diag = model(sample_imgs, return_diagnostics=True)

    result = {
        "study": "S3-028",
        "description": "State-Conditioned Per-Token Wave Router on High-Res CIFAR-100 with GAP (10 Epochs, T=4)",
        "seq_len": seq_len,
        "d_model": d_model,
        "n_hops": n_hops,
        "k_peaks": k_peaks,
        "num_params": num_params,
        "epochs": epochs,
        "total_time_s": round(time.time() - t_start, 1),
        "final_test_acc": epoch_records[-1]["test_acc"],
        "best_test_acc": max(r["test_acc"] for r in epoch_records),
        "epoch_records": epoch_records,
        "diagnostics": final_diag,
    }

    return result


@app.local_entrypoint()
def main():
    print(">>> Launching S3-028: State-Conditioned Per-Token Waves on CIFAR-100 (GAP, 10 Epochs, T=4) on Modal A10G...")
    res = run_cifar100_per_token_experiment.remote()
    print("\n" + "=" * 105)
    print(f"  COMPLETED S3-028: Final Test Acc = {res['final_test_acc']:.2f}% (Best: {res['best_test_acc']:.2f}%)")
    print(f"  Total Time: {res['total_time_s']}s")
    print("=" * 105)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "s3_028_cifar100_per_token_waves.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Saved result to {out_path}]")
